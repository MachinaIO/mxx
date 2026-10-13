import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_prelude

structure Params where
  unit : Unit

abbrev sequential_parallel_generatedRoot_11_3.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (i_1 : Nat) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31) (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) : Prop :=
  MxxRuntime.gaussianSampleAt tape (path ++ [2]) ((319/100)) (20) w_2_0 ∧
  0 ≤ (15) ∧
  (15) < 31 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (15 + (-1 * (Int.ofNat i_1))) ∧
  (15 + (-1 * (Int.ofNat i_1))) < (31 + (-1 * (Int.ofNat i_1))) ∧
  (31 + (-1 * (Int.ofNat i_1))) ≤ 31 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (15 + (-1 * (Int.ofNat i_1))) (31 + (-1 * (Int.ofNat i_1))) w_5_0 ∧
  outputs = (w_7_0, (w_12_0, ()))

def sequential_parallel_generatedRoot_11_3 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.1
  let w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := inputs.2.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := inputs.2.2.2.1
  ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 10 1),
    let w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_1_0 w_2_0
    let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31 := (MxxRuntime.unitRow (15) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31)
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16),
      let w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixMulScalarLeft w_3_0 w_5_0
      let w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_0_0 w_6_0
      let w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_9_0 w_2_0
      let w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixMulScalarLeft w_10_0 w_5_0
      let w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_8_0 w_11_0
      sequential_parallel_generatedRoot_11_3.constraints_0 tape path i_1 w_2_0 w_4_0 w_5_0 w_7_0 w_12_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_11.Witness where
  w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_3_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8

abbrev parallel_generatedRoot_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) (w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_3_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) => sequential_parallel_generatedRoot_11_3 backend hashModel tape (path ++ [3, i]) params i_0 i (current.1, (current.2.1, (w_1_0, (w_2_0, ())))) next) (Int.toNat (16)) (w_0_0, (w_0_0, ())) (w_3_0, (w_3_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 0 8 w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 8 ∧
  8 < 16 ∧
  16 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 8 16 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_3_1 0 1 0 8 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 8 ∧
  8 < 16 ∧
  16 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_3_1 0 1 8 16 w_13_0 ∧
  outputs = (w_10_0, (w_11_0, (w_12_0, (w_14_0, ()))))

abbrev parallel_generatedRoot_11.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Int × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (witness : parallel_generatedRoot_11.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := inputs.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := inputs.2.1
  let w_6_0 : Int := inputs.2.2.1
  let w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.1
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16)
  let w_3_0 := witness.w_3_0
  let w_3_1 := witness.w_3_1
  let w_4_0 := witness.w_4_0
  let w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.multiplyMonomial w_5_0 w_6_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
  let w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixAdd w_4_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixAdd w_13_0 w_9_0
  parallel_generatedRoot_11.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_0_0 w_3_0 w_3_1 w_4_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 outputs

def parallel_generatedRoot_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Int × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_11.Witness,
    parallel_generatedRoot_11.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10
  w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10
  w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_11_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_11_1 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_11_2 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_11_3 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_20_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) (w_1_0 : ByteArray) (w_9_0 : Fin 1 → Int) (w_19_0 : Fin 8192 → Int) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_11_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_11_1 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_11_2 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_11_3 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_20_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_22_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (outputs : (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [98, 98, 98, 54, 54, 101, 53, 52, 102, 54, 54, 102, 49, 50, 50, 48, 52, 99, 51, 51, 98, 53, 49, 50, 99, 56, 102, 55, 102, 54, 100, 100, 102, 53, 54, 98, 53, 54, 49, 53, 102, 99, 102, 51, 97, 50, 101, 57, 49, 48, 49, 54, 51, 99, 99, 50, 50, 102, 55, 55, 99, 98, 57, 100], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) (w_1_0) w_2_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [6]) ((319/100)) (20) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 2097152 8 w_10_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_11 backend hashModel tape (path ++ [11, i.val]) params i (w_0_0, (w_8_0, ((w_9_0 i), (w_10_0, ())))) ((w_11_0 i), ((w_11_1 i), ((w_11_2 i), ((w_11_3 i), ()))))) ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [98, 98, 98, 54, 54, 101, 53, 52, 102, 54, 54, 102, 49, 50, 50, 48, 52, 99, 51, 51, 98, 53, 49, 50, 99, 56, 102, 55, 102, 54, 100, 100, 102, 53, 54, 98, 53, 54, 49, 53, 102, 99, 102, 51, 97, 50, 101, 57, 49, 48, 49, 54, 51, 99, 99, 50, 50, 102, 55, 55, 99, 98, 57, 100], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (0)]) (w_1_0) w_12_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [98, 98, 98, 54, 54, 101, 53, 52, 102, 54, 54, 102, 49, 50, 50, 48, 52, 99, 51, 51, 98, 53, 49, 50, 99, 56, 102, 55, 102, 54, 100, 100, 102, 53, 54, 98, 53, 54, 49, 53, 102, 99, 102, 51, 97, 50, 101, 57, 49, 48, 49, 54, 51, 99, 99, 50, 50, 102, 55, 55, 99, 98, 57, 100], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) (w_1_0) w_14_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [16]) ((319/100)) (20) w_16_0 ∧
  MxxRuntime.polynomialFromValues false w_19_0 w_20_0 ∧
  outputs = (w_11_0, (w_11_1, (w_11_2, (w_11_3, (w_7_0, (w_22_0, ()))))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × ByteArray × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 × (Fin 1 → Int) × (Fin 8192 → Int) × Unit) (outputs : (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := inputs.2.2.1
  let w_9_0 : Fin 1 → Int := inputs.2.2.2.1
  let w_19_0 : Fin 8192 → Int := inputs.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_3_0 w_0_0
  let w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := MxxRuntime.matrixAdd w_4_0 w_2_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := MxxRuntime.matrixAdd w_5_0 w_6_0
  let w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 := MxxRuntime.matrixSub w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_11_2 := witness.w_11_2
  let w_11_3 := witness.w_11_3
  let w_12_0 := witness.w_12_0
  let w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_3_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixAdd w_15_0 w_16_0
  let w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.matrixPolynomial [(187072069502007624630008562053914270362791364190208)] : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_20_0 := witness.w_20_0
  let w_21_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_18_0 w_20_0
  let w_22_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixAdd w_17_0 w_21_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_9_0 w_19_0 w_2_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_14_0 w_16_0 w_20_0 w_22_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × ByteArray × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 × (Fin 1 → Int) × (Fin 8192 → Int) × Unit) (outputs : (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 10 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_prelude
