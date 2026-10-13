import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_keygen

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_10.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 × Unit) : Prop :=
  MxxRuntime.gaussianSampleAt tape (path ++ [0]) ((4389025337902376125660995805558403571735000606365759330490212352000200390911288650386268272995948791531926191579113733444167099470786667475975131864934951407850793966346594722185551921261089014531894904905967892780706350909928212818870220803113408526907/142724769270595988105828596944949513638274662400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)) (199885870141) w_0_0 ∧
  outputs = (w_0_0, (w_2_0, ()))

def parallel_generatedRoot_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  ∃ (w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
    let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 16 := (1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 16)
    let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 := MxxRuntime.matrixMul w_1_0 w_0_0
    parallel_generatedRoot_10.constraints_0 tape path w_0_0 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_14_8.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_6_0 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_7_0 : Int) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_12_0 : Int) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [51, 56, 57, 102, 52, 100, 53, 51, 51, 101, 56, 57, 55, 55, 98, 55, 54, 99, 100, 100, 52, 49, 51, 97, 49, 50, 48, 50, 51, 49, 48, 51, 100, 102, 48, 53, 49, 50, 54, 50, 54, 102, 53, 49, 53, 98, 51, 48, 53, 56, 57, 97, 50, 57, 57, 101, 100, 54, 97, 55, 53, 101, 48, 57], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_1)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [51, 56, 57, 102, 52, 100, 53, 51, 51, 101, 56, 57, 55, 55, 98, 55, 54, 99, 100, 100, 52, 49, 51, 97, 49, 50, 48, 50, 51, 49, 48, 51, 100, 102, 48, 53, 49, 50, 54, 50, 54, 102, 53, 49, 53, 98, 51, 48, 53, 56, 57, 97, 50, 57, 57, 101, 100, 54, 97, 55, 53, 101, 48, 57], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_1)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
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

def sequential_parallel_generatedRoot_14_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 × ByteArray × (Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_6_0 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 := inputs.2.2.1
  let w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.1
  ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8),
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8),
      ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16),
        ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16),
          let w_7_0 : Int := ((Int.ofNat i_0) + (Int.ofNat i_1))
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
            let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_5_0 w_8_0
            let w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixAdd w_0_0 w_9_0
            let w_12_0 : Int := ((Int.ofNat i_1) + MxxIR.selectInt (((Int.ofNat i_0) + (Int.ofNat i_1))) [(1), (0)])
            ∃ (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
              let w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_11_0 w_13_0
              let w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixSub w_10_0 w_14_0
              sequential_parallel_generatedRoot_14_8.constraints_0 hashModel i_0 i_1 w_1_0 w_6_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_14.Witness where
  w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1

abbrev parallel_generatedRoot_14.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_5_0 : ByteArray) (w_6_0 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 2 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) => sequential_parallel_generatedRoot_14_8 backend hashModel tape (path ++ [8, i]) params i_0 i (current, (w_5_0, (w_6_0, (w_7_0, ())))) next) (Int.toNat (1)) w_4_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 2 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (8193)]) w_9_0 w_10_0 ∧
  outputs = w_10_0

abbrev parallel_generatedRoot_14.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 × ByteArray × (Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (witness : parallel_generatedRoot_14.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := inputs.1
  let w_5_0 : ByteArray := inputs.2.1
  let w_6_0 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 := inputs.2.2.1
  let w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.1
  let w_0_0 : Int := MxxIR.selectInt ((Int.ofNat i_0)) [(1), (0)]
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.liftInteger w_0_0
  let w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_2_0
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_8_0 := witness.w_8_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixSub w_3_0 w_8_0
  let w_10_0 := witness.w_10_0
  parallel_generatedRoot_14.constraints_0 backend hashModel tape path params i_0 w_5_0 w_6_0 w_7_0 w_4_0 w_8_0 w_9_0 w_10_0 outputs

def parallel_generatedRoot_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 × ByteArray × (Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) : Prop :=
  ∃ witness : parallel_generatedRoot_14.Witness,
    parallel_generatedRoot_14.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1
  w_10_0 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1
  w_10_1 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1
  w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_14_0 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1
  w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) Unit) (w_2_0 : ByteArray) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_10_0 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_10_1 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_14_0 : Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1) (w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1 × (Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [98, 98, 98, 54, 54, 101, 53, 52, 102, 54, 54, 102, 49, 50, 50, 48, 52, 99, 51, 51, 98, 53, 49, 50, 99, 56, 102, 55, 102, 54, 100, 100, 102, 53, 54, 98, 53, 54, 49, 53, 102, 99, 102, 51, 97, 50, 101, 57, 49, 48, 49, 54, 51, 99, 99, 50, 50, 102, 55, 55, 99, 98, 57, 100], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (0)]) (w_2_0) w_3_0 ∧
  MxxRuntime.concatColumns w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [7]) ((4389025337902376125660995805558403571735000606365759330490212352000200390911288650386268272995948791531926191579113733444167099470786667475975131864934951407850793966346594722185551921261089014531894904905967892780706350909928212818870220803113408526907/142724769270595988105828596944949513638274662400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)) (199885870141) w_7_0 ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_10 backend hashModel tape (path ++ [10, i.val]) params i () ((w_10_0 i), ((w_10_1 i), ()))) ∧
  MxxRuntime.gadgetMatrixRuns backend 2097152 8 w_11_0 ∧
  MxxRuntime.concatColumns w_11_0 w_12_0 w_13_0 ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_14 backend hashModel tape (path ++ [14, i.val]) params i (w_9_0, (w_2_0, (w_10_1, (w_13_0, ())))) (w_14_0 i)) ∧
  MxxRuntime.familyGetStatic w_14_0 (0) w_15_0 ∧
  0 ≤ (199885870141) ∧
  MxxRuntime.preimageRunsDispatched backend w_1_0 w_0_0 w_15_0 (Int.toNat (199885870141)) w_16_0 ∧
  MxxRuntime.familyGetStatic w_14_0 (1) w_17_0 ∧
  0 ≤ (199885870141) ∧
  MxxRuntime.preimageRunsDispatched backend w_1_0 w_0_0 w_17_0 (Int.toNat (199885870141)) w_18_0 ∧
  outputs = (w_16_0, (w_18_0, (w_10_0, (w_7_0, ()))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) Unit × ByteArray × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1 × (Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) Unit := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := MxxRuntime.trapdoorPublic w_0_0
  let w_3_0 := witness.w_3_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_6_0 w_7_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixSub w_3_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_10_1 := witness.w_10_1
  let w_11_0 := witness.w_11_0
  let w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8)
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_2_0 w_4_0 w_5_0 w_1_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_10_1 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) Unit × ByteArray × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1 × (Fin 2 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_keygen
