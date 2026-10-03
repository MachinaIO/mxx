import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Ideal

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where

abbrev generatedRoot.constraints_0 (w_4_0 : Int) (outputs : Int) : Prop :=
  outputs = w_4_0

abbrev generatedRoot.body (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (inputs : Int × Int × Unit) (outputs : Int) (_witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : Int := inputs.1
  let w_2_0 : Int := inputs.2.1
  let w_0_0 : Int := 1
  let w_3_0 : Int := (w_1_0 * w_2_0)
  let w_4_0 : Int := (w_0_0 - w_3_0)
  generatedRoot.constraints_0 w_4_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Int × Int × Unit) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Ideal
