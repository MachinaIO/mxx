import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_online_eval

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 2 1) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 1048576 w_7_0 w_8_0 ∧
  0 ≤ (15) ∧
  (15) < 31 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (15 + (-1 * (Int.ofNat i_0))) ∧
  (15 + (-1 * (Int.ofNat i_0))) < (31 + (-1 * (Int.ofNat i_0))) ∧
  (31 + (-1 * (Int.ofNat i_0))) ≤ 31 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (15 + (-1 * (Int.ofNat i_0))) (31 + (-1 * (Int.ofNat i_0))) w_11_0 ∧
  outputs = w_13_0

def sequential_generatedRoot_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.1
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1),
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1),
      ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 2 1),
        ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
            let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31 := (MxxRuntime.unitRow (15) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31)
            ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16),
              let w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_generatedRoot_18.constraints_0 backend i_0 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_29_15.constraints_0 (i_1 : Nat) (w_12_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_15_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31) (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_13_0 : Int) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  0 ≤ (15) ∧
  (15) < 31 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (15 + (-1 * (Int.ofNat i_1))) ∧
  (15 + (-1 * (Int.ofNat i_1))) < (31 + (-1 * (Int.ofNat i_1))) ∧
  (31 + (-1 * (Int.ofNat i_1))) ≤ 31 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (15 + (-1 * (Int.ofNat i_1))) (31 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_13_0 w_16_0 ∧
  outputs = (w_11_0, (w_14_0, (w_16_0, ())))

def sequential_sequential_generatedRoot_29_15 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1 × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1 := inputs.2.2.2.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1 := inputs.2.2.2.2.1
  let w_12_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.1
  let w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31 := (MxxRuntime.unitRow (15) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31)
  ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16),
    let w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := ((16 * (Int.fmod ((Int.ofNat i_0)) (16))) + (Int.fmod ((1 + (Int.ofNat i_1))) (16)))
    ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8),
      ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8),
        sequential_sequential_generatedRoot_29_15.constraints_0 i_1 w_12_0 w_15_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_29.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1
  w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1
  w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_15_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_15_2 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16

abbrev sequential_generatedRoot_29.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_7_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1) (w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_15_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_15_2 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  (16) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (16)) ∧
  (Int.fmod ((Int.ofNat i_0)) (16)) < (1 + (Int.fmod ((Int.ofNat i_0)) (16))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (16))) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 (Int.fmod ((Int.ofNat i_0)) (16)) (1 + (Int.fmod ((Int.ofNat i_0)) (16))) w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_13_0 w_14_0 ∧
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) => sequential_sequential_generatedRoot_29_15 backend hashModel tape (path ++ [15, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_11_0, (w_14_0, (w_4_0, (w_7_0, ()))))))) next) (Int.toNat (16)) (w_3_0, (w_6_0, (w_8_0, ()))) (w_15_0, (w_15_1, (w_15_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_15_0 w_16_0 ∧
  outputs = w_17_0

abbrev sequential_generatedRoot_29.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_7_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_5_0 : Int) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1) (w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_15_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_15_2 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (16))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (16))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (16)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (16)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (16))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (16)))) w_2_0 ∧
  (16) ≠ 0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_5_0 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (16)) ∧
  (Int.fmod ((Int.ofNat i_0)) (16)) < (1 + (Int.fmod ((Int.ofNat i_0)) (16))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (16))) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (Int.fmod ((Int.ofNat i_0)) (16)) (1 + (Int.fmod ((Int.ofNat i_0)) (16))) w_10_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (16) ≠ 0 ∧
  sequential_generatedRoot_29.constraints_1 backend hashModel tape path params i_0 w_4_0 w_7_0 w_12_0 w_2_0 w_3_0 w_6_0 w_8_0 w_11_0 w_13_0 w_14_0 w_15_0 w_15_1 w_15_2 w_16_0 w_17_0 outputs

abbrev sequential_generatedRoot_29.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (witness : sequential_generatedRoot_29.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.1
  let w_4_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.1
  let w_7_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.1
  let w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.2.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.unitRow (0) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16)
  let w_5_0 : Int := (16 * (Int.fmod ((Int.ofNat i_0)) (16)))
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_15_1 := witness.w_15_1
  let w_15_2 := witness.w_15_2
  let w_16_0 := witness.w_16_0
  let w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_0_0 w_16_0
  sequential_generatedRoot_29.constraints_0 backend hashModel tape path params i_0 w_4_0 w_7_0 w_9_0 w_12_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_15_1 w_15_2 w_16_0 w_17_0 outputs

def sequential_generatedRoot_29 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  ∃ witness : sequential_generatedRoot_29.Witness,
    sequential_generatedRoot_29.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_63_17.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_15_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31) (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_13_0 : Int) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  0 ≤ (15) ∧
  (15) < 31 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (15 + (-1 * (Int.ofNat i_1))) ∧
  (15 + (-1 * (Int.ofNat i_1))) < (31 + (-1 * (Int.ofNat i_1))) ∧
  (31 + (-1 * (Int.ofNat i_1))) ≤ 31 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (15 + (-1 * (Int.ofNat i_1))) (31 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (16))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (16))) < 1 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_13_0 w_16_0 ∧
  outputs = (w_11_0, (w_14_0, (w_16_0, ())))

def sequential_sequential_generatedRoot_63_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1 × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1 := inputs.2.2.2.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1 := inputs.2.2.2.2.1
  let w_12_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.1
  let w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31 := (MxxRuntime.unitRow (15) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31)
  ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16),
    let w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((16 * (Int.fmod ((Int.ofNat i_0)) (16))) + (Int.fmod ((1 + (Int.ofNat i_1))) (16))) + (256 * MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (16))) [(0)]))
    ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8),
      ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8),
        sequential_sequential_generatedRoot_63_17.constraints_0 i_0 i_1 w_12_0 w_15_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_63.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1
  w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1
  w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_17_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_17_2 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16

abbrev sequential_generatedRoot_63.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_7_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1) (w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_17_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_17_2 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_19_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  ((Int.fdiv ((Int.ofNat i_0)) (16))) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (16))) [(8193)]) w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (16)) ∧
  (Int.fmod ((Int.ofNat i_0)) (16)) < (1 + (Int.fmod ((Int.ofNat i_0)) (16))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (16))) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (Int.fmod ((Int.ofNat i_0)) (16)) (1 + (Int.fmod ((Int.ofNat i_0)) (16))) w_14_0 ∧
  (16) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (16))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (16))) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (16))) [(8193)]) w_14_0 w_15_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_15_0 w_16_0 ∧
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) => sequential_sequential_generatedRoot_63_17 backend hashModel tape (path ++ [17, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_12_0, (w_16_0, (w_4_0, (w_7_0, ()))))))) next) (Int.toNat (16)) (w_3_0, (w_6_0, (w_8_0, ()))) (w_17_0, (w_17_1, (w_17_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_17_0 w_18_0 ∧
  outputs = w_19_0

abbrev sequential_generatedRoot_63.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_7_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_5_0 : Int) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 1) (w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_17_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_17_2 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_19_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (16))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (16))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (16)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (16)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (16))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (16)))) w_2_0 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (16))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (16))) < 1 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 256 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_5_0 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (16) ≠ 0 ∧
  (16) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (16)) ∧
  (Int.fmod ((Int.ofNat i_0)) (16)) < (1 + (Int.fmod ((Int.ofNat i_0)) (16))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (16))) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (Int.fmod ((Int.ofNat i_0)) (16)) (1 + (Int.fmod ((Int.ofNat i_0)) (16))) w_10_0 ∧
  (16) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (16))) ∧
  sequential_generatedRoot_63.constraints_1 backend hashModel tape path params i_0 w_4_0 w_7_0 w_13_0 w_2_0 w_3_0 w_6_0 w_8_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_18_0 w_19_0 outputs

abbrev sequential_generatedRoot_63.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (witness : sequential_generatedRoot_63.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.1
  let w_4_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.1
  let w_7_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.2.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.unitRow (0) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16)
  let w_5_0 : Int := ((16 * (Int.fmod ((Int.ofNat i_0)) (16))) + (256 * MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (16))) [(0)]))
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_17_1 := witness.w_17_1
  let w_17_2 := witness.w_17_2
  let w_18_0 := witness.w_18_0
  let w_19_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_0_0 w_18_0
  sequential_generatedRoot_63.constraints_0 backend hashModel tape path params i_0 w_4_0 w_7_0 w_9_0 w_13_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_18_0 w_19_0 outputs

def sequential_generatedRoot_63 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  ∃ witness : sequential_generatedRoot_63.Witness,
    sequential_generatedRoot_63.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_64_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 2 1) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 1048576 w_7_0 w_8_0 ∧
  0 ≤ (15) ∧
  (15) < 31 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (15 + (-1 * (Int.ofNat i_1))) ∧
  (15 + (-1 * (Int.ofNat i_1))) < (31 + (-1 * (Int.ofNat i_1))) ∧
  (31 + (-1 * (Int.ofNat i_1))) ≤ 31 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (15 + (-1 * (Int.ofNat i_1))) (31 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = w_13_0

def sequential_sequential_generatedRoot_64_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.1
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1),
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1),
      ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 2 1),
        ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
            let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31 := (MxxRuntime.unitRow (15) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31)
            ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16),
              let w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_64_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_64_44.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 2 1) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 1048576 w_7_0 w_8_0 ∧
  0 ≤ (15) ∧
  (15) < 31 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (15 + (-1 * (Int.ofNat i_1))) ∧
  (15 + (-1 * (Int.ofNat i_1))) < (31 + (-1 * (Int.ofNat i_1))) ∧
  (31 + (-1 * (Int.ofNat i_1))) ≤ 31 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (15 + (-1 * (Int.ofNat i_1))) (31 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = w_13_0

def sequential_sequential_generatedRoot_64_44 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.1
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1),
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1),
      ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 2 1),
        ∃ (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
          ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 16 1),
            let w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31 := (MxxRuntime.unitRow (15) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 31)
            ∃ (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16),
              let w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_64_44.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_64.Witness where
  w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_19_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_22_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_24_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_25_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_27_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_28_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_29_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_30_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_31_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_32_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_33_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_35_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_36_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_38_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_39_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_40_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_41_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_43_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_44_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_45_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_65_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 8
  w_69_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_70_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_71_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8

abbrev sequential_generatedRoot_64.constraints_5 (w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_67_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_71_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_75_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 8 ∧
  8 < 16 ∧
  16 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 8 16 w_71_0 ∧
  outputs = (w_67_0, (w_75_0, ()))

abbrev sequential_generatedRoot_64.constraints_4 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_44_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_65_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 8) (w_67_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_69_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_70_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_71_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_75_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 8 w_48_0 ∧
  0 ≤ (0) ∧
  (0) < 8192 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_53_0 w_54_0 ∧
  0 ≤ (8191) ∧
  (8191) < 8192 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_58_0 w_59_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 2097152 8 w_63_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_64_0 w_65_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 8 ∧
  8 < 16 ∧
  16 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 8 16 w_69_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 8 ∧
  8 < 16 ∧
  16 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 8 16 w_70_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_64.constraints_5 w_47_0 w_67_0 w_71_0 w_75_0 outputs

abbrev sequential_generatedRoot_64.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_5_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_26_0 : ByteArray) (w_46_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_20_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_25_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_31_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_34_0 : Int) (w_37_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_38_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_39_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_40_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_41_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_42_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_43_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_44_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_45_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_65_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 8) (w_67_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_69_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_70_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_71_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_75_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_26_0) w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_38_0 w_39_0 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_34_0 w_40_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_40_0 w_41_0 ∧
  MxxRuntime.concatColumns w_37_0 w_42_0 w_43_0 ∧
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) => sequential_sequential_generatedRoot_64_44 backend hashModel tape (path ++ [44, i]) params i_0 i (current, (w_25_0, (w_31_0, (w_43_0, ())))) next) (Int.toNat (16)) w_20_0 w_44_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 0 8 w_45_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (16 * (Int.ofNat i_0)) ∧
  (16 * (Int.ofNat i_0)) < (16 + (16 * (Int.ofNat i_0))) ∧
  (16 + (16 * (Int.ofNat i_0))) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_46_0 0 1 (16 * (Int.ofNat i_0)) (16 + (16 * (Int.ofNat i_0))) w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_64.constraints_4 backend i_0 w_18_0 w_44_0 w_47_0 w_48_0 w_53_0 w_54_0 w_58_0 w_59_0 w_63_0 w_64_0 w_65_0 w_67_0 w_69_0 w_70_0 w_71_0 w_75_0 outputs

abbrev sequential_generatedRoot_64.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_5_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_21_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_23_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_26_0 : ByteArray) (w_46_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_20_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_22_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_24_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_25_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_27_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_28_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_29_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_30_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_31_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_32_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_33_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_34_0 : Int) (w_35_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_36_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_37_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_38_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_39_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_40_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_41_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_42_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_43_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_44_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_45_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_65_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 8) (w_67_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_69_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_70_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_71_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_75_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_21_0 w_22_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_23_0 w_24_0 ∧
  MxxRuntime.concatColumns w_22_0 w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_26_0) w_27_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_26_0) w_29_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_26_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_32_0 w_33_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_34_0 w_35_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_35_0 w_36_0 ∧
  sequential_generatedRoot_64.constraints_3 backend hashModel tape path params i_0 w_5_0 w_26_0 w_46_0 w_18_0 w_20_0 w_25_0 w_31_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_53_0 w_54_0 w_58_0 w_59_0 w_63_0 w_64_0 w_65_0 w_67_0 w_69_0 w_70_0 w_71_0 w_75_0 outputs

abbrev sequential_generatedRoot_64.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_5_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_21_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_23_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_26_0 : ByteArray) (w_46_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_19_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_20_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_22_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_24_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_25_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_27_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_28_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_29_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_30_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_31_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_32_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_33_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_34_0 : Int) (w_35_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_36_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_37_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_38_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_39_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_40_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_41_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_42_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_43_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_44_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_45_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_65_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 8) (w_67_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_69_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_70_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_71_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_75_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (16 * (Int.ofNat i_0)) ∧
  (16 * (Int.ofNat i_0)) < (16 + (16 * (Int.ofNat i_0))) ∧
  (16 + (16 * (Int.ofNat i_0))) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (16 * (Int.ofNat i_0)) (16 + (16 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 8 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 8 ∧
  8 < 16 ∧
  16 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 8 16 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) => sequential_sequential_generatedRoot_64_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (16)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 8 w_19_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  sequential_generatedRoot_64.constraints_2 backend hashModel tape path params i_0 w_2_0 w_5_0 w_21_0 w_23_0 w_26_0 w_46_0 w_18_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_53_0 w_54_0 w_58_0 w_59_0 w_63_0 w_64_0 w_65_0 w_67_0 w_69_0 w_70_0 w_71_0 w_75_0 outputs

abbrev sequential_generatedRoot_64.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_5_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_21_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_23_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_26_0 : ByteArray) (w_46_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_3_0 : Int) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_10_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_15_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_19_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_20_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_22_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_24_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_25_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_27_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_28_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_29_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_30_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_31_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_32_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_33_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_34_0 : Int) (w_35_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_36_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_37_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_38_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_39_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_40_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_41_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_42_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_43_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_44_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_45_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_65_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 8) (w_67_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_69_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_70_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_71_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_75_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (16 * (Int.ofNat i_0)) ∧
  (16 * (Int.ofNat i_0)) < (16 + (16 * (Int.ofNat i_0))) ∧
  (16 + (16 * (Int.ofNat i_0))) ≤ 16 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (16 * (Int.ofNat i_0)) (16 + (16 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 8 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 8 ∧
  8 < 16 ∧
  16 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 8 16 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_64.constraints_1 backend hashModel tape path params i_0 w_2_0 w_5_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_53_0 w_54_0 w_58_0 w_59_0 w_63_0 w_64_0 w_65_0 w_67_0 w_69_0 w_70_0 w_71_0 w_75_0 outputs

abbrev sequential_generatedRoot_64.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × ByteArray × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (witness : sequential_generatedRoot_64.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.1
  let w_68_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.1
  let w_2_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.1
  let w_5_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.1
  let w_8_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.2.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.2.2.1
  let w_21_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.1
  let w_26_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.1
  let w_46_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16)
  let w_3_0 : Int := MxxIR.selectInt ((Int.ofNat i_0)) [(0)]
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
  let w_20_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16)
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
  let w_34_0 : Int := MxxIR.selectInt ((Int.ofNat i_0)) [(0)]
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixSub w_33_0 w_36_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixSub w_39_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixAdd w_45_0 w_48_0
  let w_50_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixAdd w_19_0 w_49_0
  let w_51_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.matrixPolynomial [(187072069502007624630008562053914270362791364190209)] : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_52_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.matrixPolynomial [(0)] : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.rotationPolynomial (0) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_54_0 := witness.w_54_0
  let w_55_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_52_0 w_54_0
  let w_56_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_57_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.rotationPolynomial (8191) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixNeg w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_56_0 w_59_0
  let w_61_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixAdd w_55_0 w_60_0
  let w_62_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_51_0 w_61_0
  let w_63_0 := witness.w_63_0
  let w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixMulScalarLeft w_62_0 w_63_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixMul w_50_0 w_65_0
  let w_67_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixAdd w_0_0 w_66_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixAdd w_70_0 w_71_0
  let w_73_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixAdd w_69_0 w_72_0
  let w_74_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixMul w_73_0 w_65_0
  let w_75_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixAdd w_68_0 w_74_0
  sequential_generatedRoot_64.constraints_0 backend hashModel tape path params i_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_53_0 w_54_0 w_58_0 w_59_0 w_63_0 w_64_0 w_65_0 w_67_0 w_69_0 w_70_0 w_71_0 w_75_0 outputs

def sequential_generatedRoot_64 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × ByteArray × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_64.Witness,
    sequential_generatedRoot_64.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_23_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_24_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_25_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_26_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_27_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_28_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_29_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_31_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_35_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_40_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1
  w_44_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_46_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 8
  w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_62_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16
  w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8
  w_64_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_22_0 : ByteArray) (w_50_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_51_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_52_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_55_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_56_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_57_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_60_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_61_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_30_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_31_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_49_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_62_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_64_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 8 ∧
  8 < 16 ∧
  16 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_30_0 0 1 8 16 w_48_0 ∧
  MxxRuntime.concatColumns w_57_0 w_58_0 w_59_0 ∧
  MxxRuntime.concatColumns w_60_0 w_61_0 w_62_0 ∧
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) => sequential_generatedRoot_63 backend hashModel tape (path ++ [63, i]) params i (current, (w_55_0, (w_56_0, (w_59_0, (w_62_0, ()))))) next) (Int.toNat (16)) w_54_0 w_63_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) => sequential_generatedRoot_64 backend hashModel tape (path ++ [64, i]) params i (current.1, (current.2.1, (w_50_0, (w_51_0, (w_52_0, (w_53_0, (w_31_0, (w_48_0, (w_22_0, (w_63_0, ())))))))))) next) (Int.toNat (1)) (w_47_0, (w_49_0, ())) (w_64_0, (w_64_1, ())) ∧
  outputs = (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_64_0, (w_64_1, ()))))))

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_8_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_10_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_13_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_15_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_20_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_21_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_22_0 : ByteArray) (w_50_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_51_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_52_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_55_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_56_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_57_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_60_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_61_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_7_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_9_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_11_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_12_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_14_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_16_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_17_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_18_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_19_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_23_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_24_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_25_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_26_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_27_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_28_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_29_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_30_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_31_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_34_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_35_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_39_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_40_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1) (w_44_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_45_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_46_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 8 8) (w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_48_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_49_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_59_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_62_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_63_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) (w_64_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (w_64_1 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  MxxRuntime.concatColumns w_5_0 w_6_0 w_7_0 ∧
  MxxRuntime.familyGetStatic w_8_0 (0) w_9_0 ∧
  MxxRuntime.familyGetStatic w_10_0 (0) w_11_0 ∧
  MxxRuntime.concatColumns w_9_0 w_11_0 w_12_0 ∧
  MxxRuntime.familyGetStatic w_13_0 (0) w_14_0 ∧
  MxxRuntime.familyGetStatic w_15_0 (0) w_16_0 ∧
  MxxRuntime.concatColumns w_14_0 w_16_0 w_17_0 ∧
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) => sequential_generatedRoot_18 backend hashModel tape (path ++ [18, i]) params i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (16)) w_4_0 w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_22_0) w_23_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_22_0) w_24_0 ∧
  MxxRuntime.concatColumns w_23_0 w_24_0 w_25_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_22_0) w_26_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [100, 51, 55, 54, 53, 97, 53, 48, 99, 56, 57, 48, 98, 100, 57, 49, 55, 102, 56, 98, 54, 55, 53, 53, 57, 100, 57, 54, 49, 99, 54, 53, 98, 54, 48, 49, 52, 56, 56, 53, 101, 99, 53, 54, 98, 48, 53, 49, 102, 99, 97, 102, 52, 97, 53, 101, 52, 57, 48, 97, 55, 50, 50, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_22_0) w_27_0 ∧
  MxxRuntime.concatColumns w_26_0 w_27_0 w_28_0 ∧
  0 ≤ (16) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16) => sequential_generatedRoot_29 backend hashModel tape (path ++ [29, i]) params i (current, (w_20_0, (w_21_0, (w_25_0, (w_28_0, ()))))) next) (Int.toNat (16)) w_19_0 w_29_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 8 ∧
  8 ≤ 16 ∧
  MxxRuntime.sliceMatrix w_30_0 0 1 0 8 w_31_0 ∧
  0 ≤ (0) ∧
  (0) < 8192 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_34_0 w_35_0 ∧
  0 ≤ (8191) ∧
  (8191) < 8192 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_39_0 w_40_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 2097152 8 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 2097152 8 w_45_0 w_46_0 ∧
  generatedRoot.constraints_1 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_22_0 w_50_0 w_51_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_30_0 w_31_0 w_47_0 w_48_0 w_49_0 w_54_0 w_59_0 w_62_0 w_63_0 w_64_0 w_64_1 outputs

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × ByteArray × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.1
  let w_3_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.1
  let w_6_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.1
  let w_8_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.1
  let w_10_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.1
  let w_13_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.1
  let w_15_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_52_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_56_0 : Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_57_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_58_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_60_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_61_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16)
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16)
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := MxxRuntime.matrixAdd w_18_0 w_29_0
  let w_31_0 := witness.w_31_0
  let w_32_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.matrixPolynomial [(187072069502007624630008562053914270362791364190209)] : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_33_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.matrixPolynomial [(0)] : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_34_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.rotationPolynomial (0) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_35_0 := witness.w_35_0
  let w_36_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_33_0 w_35_0
  let w_37_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_38_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := (MxxRuntime.rotationPolynomial (8191) : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1)
  let w_39_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixNeg w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_37_0 w_40_0
  let w_42_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixAdd w_36_0 w_41_0
  let w_43_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_32_0 w_42_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixMulScalarLeft w_43_0 w_44_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixMul w_31_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 := MxxRuntime.matrixMul w_48_0 w_46_0
  let w_54_0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 := (0 : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16)
  let w_59_0 := witness.w_59_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_64_1 := witness.w_64_1
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_10_0 w_13_0 w_15_0 w_20_0 w_21_0 w_22_0 w_50_0 w_51_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_4_0 w_7_0 w_9_0 w_11_0 w_12_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_34_0 w_35_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_54_0 w_59_0 w_62_0 w_63_0 w_64_0 w_64_1 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × ByteArray × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 1 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 16 × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × (Fin 256 → Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8) × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) (outputs : Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Mxx.Primitives.ExactMatrix 374144139004015249260017124107828540725582728380417 8192 1 8 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_online_eval
