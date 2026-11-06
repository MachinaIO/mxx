import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_keygen

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_10.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Unit) : Prop :=
  MxxRuntime.gaussianSampleAt tape (path ++ [0]) ((36955943884422039849686363886844743345501063507786308997710330263389608971746115995555182166571772460446609040073468199866370528568323731719438794004731564382523252973706112397296780774527489663820602174997088646797830840886459663010452528843218822143/71362384635297994052914298472474756819137331200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)) (3366109981) w_0_0 ∧
  outputs = (w_0_0, (w_2_0, ()))

def parallel_generatedRoot_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  ∃ (w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1),
    let w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 20 := (1 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 20)
    let w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 := MxxRuntime.matrixMul w_1_0 w_0_0
    parallel_generatedRoot_10.constraints_0 tape path w_0_0 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_14_8.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_6_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_7_0 : Int) (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_12_0 : Int) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [99, 50, 51, 100, 97, 98, 52, 57, 98, 52, 101, 50, 102, 49, 56, 100, 102, 54, 53, 99, 51, 49, 97, 98, 101, 51, 51, 53, 101, 52, 57, 98, 97, 53, 102, 100, 98, 57, 97, 54, 56, 57, 102, 99, 51, 101, 98, 100, 102, 56, 52, 99, 56, 97, 97, 52, 99, 97, 51, 99, 56, 49, 52, 100], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_1)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [99, 50, 51, 100, 97, 98, 52, 57, 98, 52, 101, 50, 102, 49, 56, 100, 102, 54, 53, 99, 51, 49, 97, 98, 101, 51, 51, 53, 101, 52, 57, 98, 97, 53, 102, 100, 98, 57, 97, 54, 56, 57, 102, 99, 51, 101, 98, 100, 102, 56, 52, 99, 56, 97, 97, 52, 99, 97, 51, 99, 56, 49, 52, 100], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_1)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.concatColumns w_2_0 w_3_0 w_4_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 2 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (8193)]) w_4_0 w_5_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_7_0 w_8_0 ∧
  0 ≤ (((Int.ofNat i_0) + (Int.ofNat i_1))) ∧
  (((Int.ofNat i_0) + (Int.ofNat i_1))) < 2 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_parallel_generatedRoot_14_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_6_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 := inputs.2.2.1
  let w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 := inputs.2.2.2.1
  ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
      ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
        ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
          let w_7_0 : Int := ((Int.ofNat i_0) + (Int.ofNat i_1))
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1),
            let w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_5_0 w_8_0
            let w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_0_0 w_9_0
            let w_12_0 : Int := ((Int.ofNat i_1) + MxxIR.selectInt (((Int.ofNat i_0) + (Int.ofNat i_1))) [(1), (0)])
            ∃ (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1),
              let w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_11_0 w_13_0
              let w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixSub w_10_0 w_14_0
              sequential_parallel_generatedRoot_14_8.constraints_0 hashModel i_0 i_1 w_1_0 w_6_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_14.Witness where
  w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1

abbrev parallel_generatedRoot_14.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_5_0 : ByteArray) (w_6_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 2 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) => sequential_parallel_generatedRoot_14_8 backend hashModel tape (path ++ [8, i]) params i_0 i (current, (w_5_0, (w_6_0, (w_7_0, ())))) next) (Int.toNat (1)) w_4_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 2 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (8193)]) w_9_0 w_10_0 ∧
  outputs = w_10_0

abbrev parallel_generatedRoot_14.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (witness : parallel_generatedRoot_14.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.1
  let w_5_0 : ByteArray := inputs.2.1
  let w_6_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 := inputs.2.2.1
  let w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 := inputs.2.2.2.1
  let w_0_0 : Int := MxxIR.selectInt ((Int.ofNat i_0)) [(1), (0)]
  let w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.liftInteger w_0_0
  let w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_2_0
  let w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := (0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1)
  let w_8_0 := witness.w_8_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixSub w_3_0 w_8_0
  let w_10_0 := witness.w_10_0
  parallel_generatedRoot_14.constraints_0 backend hashModel tape path params i_0 w_5_0 w_6_0 w_7_0 w_4_0 w_8_0 w_9_0 w_10_0 outputs

def parallel_generatedRoot_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) : Prop :=
  ∃ witness : parallel_generatedRoot_14.Witness,
    parallel_generatedRoot_14.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20
  w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1
  w_10_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1
  w_10_1 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1
  w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20
  w_14_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1
  w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_18_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit) (w_2_0 : ByteArray) (w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_10_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_10_1 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_14_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_18_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [49, 55, 98, 56, 98, 50, 100, 51, 100, 52, 51, 57, 99, 100, 101, 56, 100, 54, 102, 49, 54, 100, 53, 102, 51, 55, 48, 52, 55, 50, 48, 56, 52, 51, 53, 54, 57, 48, 48, 53, 49, 49, 48, 102, 57, 49, 102, 49, 53, 97, 48, 52, 101, 49, 101, 102, 97, 100, 50, 50, 55, 51, 52, 50], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (0)]) (w_2_0) w_3_0 ∧
  MxxRuntime.concatColumns w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [7]) ((36955943884422039849686363886844743345501063507786308997710330263389608971746115995555182166571772460446609040073468199866370528568323731719438794004731564382523252973706112397296780774527489663820602174997088646797830840886459663010452528843218822143/71362384635297994052914298472474756819137331200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)) (3366109981) w_7_0 ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_10 backend hashModel tape (path ++ [10, i.val]) params i () ((w_10_0 i), ((w_10_1 i), ()))) ∧
  MxxRuntime.gadgetMatrixRuns backend 32768 10 w_11_0 ∧
  MxxRuntime.concatColumns w_11_0 w_12_0 w_13_0 ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_14 backend hashModel tape (path ++ [14, i.val]) params i (w_9_0, (w_2_0, (w_10_1, (w_13_0, ())))) (w_14_0 i)) ∧
  MxxRuntime.familyGetStatic w_14_0 (0) w_15_0 ∧
  0 ≤ (3366109981) ∧
  MxxRuntime.preimageRunsDispatched backend w_1_0 w_0_0 w_15_0 (Int.toNat (3366109981)) w_16_0 ∧
  MxxRuntime.familyGetStatic w_14_0 (1) w_17_0 ∧
  0 ≤ (3366109981) ∧
  MxxRuntime.preimageRunsDispatched backend w_1_0 w_0_0 w_17_0 (Int.toNat (3366109981)) w_18_0 ∧
  outputs = (w_16_0, (w_18_0, (w_10_0, (w_7_0, ()))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := MxxRuntime.trapdoorPublic w_0_0
  let w_3_0 := witness.w_3_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_6_0 w_7_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixSub w_3_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_10_1 := witness.w_10_1
  let w_11_0 := witness.w_11_0
  let w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := (0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10)
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_2_0 w_4_0 w_5_0 w_1_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_10_1 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_keygen
