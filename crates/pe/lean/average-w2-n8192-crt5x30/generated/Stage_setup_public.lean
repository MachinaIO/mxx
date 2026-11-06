import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_setup_public

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where

abbrev generatedRoot.constraints_0 (w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) : Prop :=
  outputs = w_0_0

abbrev generatedRoot.body (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (_witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := inputs
  generatedRoot.constraints_0 w_0_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Stage_setup_public
