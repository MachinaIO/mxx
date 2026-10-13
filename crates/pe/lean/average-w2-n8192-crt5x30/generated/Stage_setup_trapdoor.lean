import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_setup_trapdoor

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_1 : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit
  w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (w_0_1 : MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit) (w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit × Unit) : Prop :=
  MxxRuntime.trapdoorSampleAt backend tape (path ++ [0]) (5154369773525533/1125899906842624) 32768 10 3366109981 w_0_0 w_0_1 ∧
  outputs = (w_0_0, (w_0_1, ()))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_1 := witness.w_0_1
  let w_0_0 := witness.w_0_0
  generatedRoot.constraints_0 backend tape path w_0_1 w_0_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) Unit × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend tape path params () outputs witness


end Stage_setup_trapdoor
