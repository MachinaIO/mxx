import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_encrypt

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_5.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_11_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_12_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1048576 4 w_12_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [16]) ((4/1)) (26) w_16_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87, 105, 114, 101, 73, 100, 40, 48, 41, 47, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_20_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [25]) ((4/1)) (26) w_25_0 ∧
  outputs = (w_17_0, (w_26_0, (w_11_0, ())))

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × ByteArray × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Int × Int × Unit) (outputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.2.2.1
  let w_8_0 : Int := inputs.2.2.2.1
  let w_10_0 : Int := inputs.2.2.2.2.1
  ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
    let w_3_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
      let w_6_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_9_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := MxxRuntime.liftInteger w_8_0
      let w_11_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := MxxRuntime.multiplyMonomial w_9_0 w_10_0
      ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
        let w_13_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarLeft w_11_0 w_12_0
        let w_14_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarLeft w_0_0 w_13_0
        let w_15_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixSub w_7_0 w_14_0
        ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
          let w_17_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixAdd w_15_0 w_16_0
          ∃ (w_18_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
            let w_19_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarLeft w_0_0 w_18_0
            ∃ (w_20_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
              let w_21_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarLeft w_4_0 w_20_0
              let w_22_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixAdd w_19_0 w_21_0
              let w_23_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixMulScalarLeft w_4_0 w_13_0
              let w_24_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixSub w_22_0 w_23_0
              ∃ (w_25_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4),
                let w_26_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 := MxxRuntime.matrixAdd w_24_0 w_25_0
                parallel_generatedRoot_5.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_5_0 w_11_0 w_12_0 w_16_0 w_17_0 w_18_0 w_20_0 w_25_0 w_26_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_5_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_5_1 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4
  w_5_2 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_1_0 : ByteArray) (w_3_0 : Fin 5 → Int) (w_4_0 : Fin 5 → Int) (w_2_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (w_5_0 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_5_1 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) (w_5_2 : Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) (outputs : (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) × Unit) : Prop :=
  (5) = 5 ∧
  (∀ i : Fin 5, parallel_generatedRoot_5 backend hashModel tape (path ++ [5, i.val]) params i (w_0_0, (w_1_0, (w_2_0, ((w_3_0 i), ((w_4_0 i), ()))))) ((w_5_0 i), ((w_5_1 i), ((w_5_2 i), ())))) ∧
  outputs = (w_5_0, (w_5_1, (w_5_2, ())))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × ByteArray × (Fin 5 → Int) × (Fin 5 → Int) × Unit) (outputs : (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_3_0 : Fin 5 → Int := inputs.2.2.1
  let w_4_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 := (1 : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1)
  let w_5_0 := witness.w_5_0
  let w_5_1 := witness.w_5_1
  let w_5_2 := witness.w_5_2
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_3_0 w_4_0 w_2_0 w_5_0 w_5_1 w_5_2 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × ByteArray × (Fin 5 → Int) × (Fin 5 → Int) × Unit) (outputs : (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_encrypt
