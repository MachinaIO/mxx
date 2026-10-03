import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_modswitch

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 2 1

abbrev generatedRoot.constraints_0 (w_0_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (w_1_0 : Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 2 1) (outputs : Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 2 1) : Prop :=
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497, 18014398508400641] (1032193) w_0_0 w_1_0 ∧
  outputs = w_1_0

abbrev generatedRoot.body (params : Params) (inputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (outputs : Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 2 1) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 := inputs
  let w_1_0 := witness.w_1_0
  generatedRoot.constraints_0 w_0_0 w_1_0 outputs

def generatedRoot (params : Params) (inputs : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1) (outputs : Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 2 1) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Stage_modswitch
