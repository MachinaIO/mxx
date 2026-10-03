import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_relinearize

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
  w_2_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
  w_3_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
  w_5_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
  w_6_0 : Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 3 1
  w_8_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1

abbrev generatedRoot.constraints_0 (w_0_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1) (w_1_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1) (w_2_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1) (w_3_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (w_5_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 3 1) (w_7_0 : Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 1) (w_8_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (w_9_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (outputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_0_0 1 2 0 1 w_1_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_0_0 2 3 0 1 w_2_0 ∧
  MxxRuntime.concatRows w_1_0 w_2_0 w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_0_0 0 1 0 1 w_5_0 ∧
  MxxRuntime.rnsModUpRuns [18014398507892737, 18014398508138497, 18014398508400641] 1 true w_5_0 w_6_0 ∧
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497, 18014398508400641, 72057594037616641] (1032193) w_7_0 w_8_0 ∧
  outputs = w_9_0

abbrev generatedRoot.body (params : Params) (inputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1 × Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 3 × Unit) (outputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1 := inputs.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 3 := inputs.2.1
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 1 := MxxRuntime.matrixMul w_4_0 w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 := MxxRuntime.matrixAdd w_3_0 w_8_0
  generatedRoot.constraints_0 w_0_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 outputs

def generatedRoot (params : Params) (inputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1 × Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 3 × Unit) (outputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Stage_relinearize
