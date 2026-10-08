import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_prelude

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_11.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 20) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  MxxRuntime.gaussianSampleAt tape (path ++ [1]) ((319/100)) (20) w_1_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 20 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 0 10 w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 10 ∧
  10 < 20 ∧
  20 ≤ 20 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 10 20 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 10 ∧
  10 ≤ 20 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 10 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 10 ∧
  10 < 20 ∧
  20 ≤ 20 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 10 20 w_14_0 ∧
  outputs = (w_9_0, (w_10_0, (w_13_0, (w_15_0, ()))))

def parallel_generatedRoot_11 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × Int × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := inputs.1
  let w_5_0 : Int := inputs.2.1
  let w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.1
  let w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := inputs.2.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 20),
    let w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 := MxxRuntime.matrixMul w_0_0 w_1_0
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
      let w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := (1 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1)
      let w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.multiplyMonomial w_4_0 w_5_0
      let w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_6_0 w_7_0
      let w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_3_0 w_8_0
      ∃ (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
        let w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 := MxxRuntime.matrixMul w_11_0 w_1_0
        ∃ (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
          ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
            let w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_14_0 w_8_0
            parallel_generatedRoot_11.constraints_0 tape path w_1_0 w_2_0 w_3_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12
  w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12
  w_11_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_11_1 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_11_2 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_11_3 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (w_1_0 : Fin 1 → Int) (w_3_0 : ByteArray) (w_19_0 : Fin 8192 → Int) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (w_11_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_11_1 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_11_2 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_11_3 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (outputs : (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × Unit) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 32768 10 w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [57, 100, 56, 97, 97, 51, 54, 55, 54, 51, 56, 57, 57, 97, 52, 49, 57, 97, 51, 49, 57, 100, 99, 99, 50, 99, 97, 52, 99, 48, 100, 101, 55, 99, 101, 48, 49, 100, 55, 53, 97, 97, 49, 99, 56, 99, 52, 53, 49, 101, 50, 52, 55, 53, 54, 54, 100, 49, 100, 49, 56, 54, 101, 54], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) (w_3_0) w_4_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [8]) ((319/100)) (20) w_8_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_11 backend hashModel tape (path ++ [11, i.val]) params i (w_0_0, ((w_1_0 i), (w_2_0, (w_10_0, ())))) ((w_11_0 i), ((w_11_1 i), ((w_11_2 i), ((w_11_3 i), ()))))) ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [57, 100, 56, 97, 97, 51, 54, 55, 54, 51, 56, 57, 57, 97, 52, 49, 57, 97, 51, 49, 57, 100, 99, 99, 50, 99, 97, 52, 99, 48, 100, 101, 55, 99, 101, 48, 49, 100, 55, 53, 97, 97, 49, 99, 56, 99, 52, 53, 49, 101, 50, 52, 55, 53, 54, 54, 100, 49, 100, 49, 56, 54, 101, 54], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (0)]) (w_3_0) w_12_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [57, 100, 56, 97, 97, 51, 54, 55, 54, 51, 56, 57, 57, 97, 52, 49, 57, 97, 51, 49, 57, 100, 99, 99, 50, 99, 97, 52, 99, 48, 100, 101, 55, 99, 101, 48, 49, 100, 55, 53, 97, 97, 49, 99, 56, 99, 52, 53, 49, 101, 50, 52, 55, 53, 54, 54, 100, 49, 100, 49, 56, 54, 101, 54], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) (w_3_0) w_14_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [16]) ((319/100)) (20) w_16_0 ∧
  MxxRuntime.polynomialFromValues false w_19_0 w_20_0 ∧
  outputs = (w_11_0, (w_11_1, (w_11_2, (w_11_3, (w_9_0, (w_22_0, ()))))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × (Fin 1 → Int) × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × (Fin 8192 → Int) × Unit) (outputs : (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := inputs.1
  let w_1_0 : Fin 1 → Int := inputs.2.1
  let w_3_0 : ByteArray := inputs.2.2.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.2.2.2.1
  let w_19_0 : Fin 8192 → Int := inputs.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_4_0 := witness.w_4_0
  let w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := MxxRuntime.matrixMulScalarLeft w_5_0 w_0_0
  let w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := MxxRuntime.matrixAdd w_6_0 w_4_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := MxxRuntime.matrixSub w_4_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_11_2 := witness.w_11_2
  let w_11_3 := witness.w_11_3
  let w_12_0 := witness.w_12_0
  let w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_15_0 w_16_0
  let w_18_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := (MxxRuntime.matrixPolynomial [(712851031313528105616668575764770600691113984)] : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1)
  let w_20_0 := witness.w_20_0
  let w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_18_0 w_20_0
  let w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_17_0 w_21_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_3_0 w_19_0 w_2_0 w_4_0 w_8_0 w_9_0 w_10_0 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_14_0 w_16_0 w_20_0 w_22_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × (Fin 1 → Int) × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × (Fin 8192 → Int) × Unit) (outputs : (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_prelude
