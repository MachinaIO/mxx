import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_multiply

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 4 1
  w_3_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
  w_4_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
  w_5_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
  w_7_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
  w_8_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1
  w_8_concat_1 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1

abbrev generatedRoot.constraints_0 (w_0_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (w_1_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (w_2_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 4 1) (w_3_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1) (w_4_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1) (w_5_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1) (w_7_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1) (w_8_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1) (w_8_concat_1 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (outputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1) : Prop :=
  MxxRuntime.tensorRuns w_0_0 w_1_0 w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 0 1 w_3_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_2_0 1 2 0 1 w_4_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_2_0 2 3 0 1 w_5_0 ∧
  0 ≤ 3 ∧
  3 < 4 ∧
  4 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_2_0 3 4 0 1 w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  outputs = w_8_0

abbrev generatedRoot.body (params : Params) (inputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 × Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 := inputs.1
  let w_1_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 := inputs.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_6_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1 := MxxRuntime.matrixAdd w_4_0 w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_8_concat_1 := witness.w_8_concat_1
  generatedRoot.constraints_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 outputs

def generatedRoot (params : Params) (inputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 × Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Stage_multiply
