import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Requirement_0

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where

abbrev generatedRoot.constraints_0 (w_14_0 : Bool) (outputs : Bool) : Prop :=
  outputs = w_14_0

abbrev generatedRoot.body (params : Params) (inputs : Int × Int × Int × Int × Unit) (outputs : Bool) (_witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Int := inputs.2.1
  let w_6_0 : Int := inputs.2.2.1
  let w_7_0 : Int := inputs.2.2.2.1
  let w_2_0 : Int := 8
  let w_3_0 : Int := (w_1_0 * w_2_0)
  let w_4_0 : Bool := decide (w_0_0 = w_3_0)
  let w_5_0 : Int := if w_4_0 then 1 else 0
  let w_8_0 : Int := 8
  let w_9_0 : Int := (w_7_0 * w_8_0)
  let w_10_0 : Bool := decide (w_6_0 = w_9_0)
  let w_11_0 : Int := if w_10_0 then 1 else 0
  let w_12_0 : Int := (w_5_0 + w_11_0)
  let w_13_0 : Int := 2
  let w_14_0 : Bool := decide (w_12_0 = w_13_0)
  generatedRoot.constraints_0 w_14_0 outputs

def generatedRoot (params : Params) (inputs : Int × Int × Int × Int × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Requirement_0
