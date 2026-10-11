import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_mask

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_0_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (outputs : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) : Prop :=
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [0]) (-1) (1) w_0_0 ∧
  outputs = w_0_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 := witness.w_0_0
  generatedRoot.constraints_0 tape path w_0_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params () outputs witness


end Stage_mask
