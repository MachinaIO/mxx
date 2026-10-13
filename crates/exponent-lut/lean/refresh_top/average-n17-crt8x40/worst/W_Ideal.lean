import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Ideal

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where

abbrev generatedRoot.constraints_0 (w_2_0 : Int) (outputs : Int) : Prop :=
  outputs = w_2_0

abbrev generatedRoot.body (params : Params) (inputs : Int) (outputs : Int) (_witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 * w_1_0)
  generatedRoot.constraints_0 w_2_0 outputs

def generatedRoot (params : Params) (inputs : Int) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end W_Ideal
