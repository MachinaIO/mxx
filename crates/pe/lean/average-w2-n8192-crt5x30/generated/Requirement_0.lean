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
  w_1_0 : Int
  w_8_0 : Int
  w_16_0 : Int

abbrev generatedRoot.constraints_0 (w_0_0 : Fin 1 → Int) (w_1_0 : Int) (w_2_0 : Int) (w_8_0 : Int) (w_9_0 : Int) (w_12_0 : Int) (w_15_0 : Int) (w_16_0 : Int) (w_22_0 : Bool) (outputs : Bool) : Prop :=
  MxxRuntime.familyGetStatic w_0_0 (0) w_1_0 ∧
  w_2_0 ≠ 0 ∧
  MxxRuntime.familyGetStatic w_0_0 (0) w_8_0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_9_0 [w_12_0, w_15_0] w_16_0 ∧
  outputs = w_22_0

abbrev generatedRoot.body (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (inputs : Fin 1 → Int) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Fin 1 → Int := inputs
  let w_1_0 := witness.w_1_0
  let w_2_0 : Int := 1
  let w_3_0 : Int := (w_1_0 % w_2_0)
  let w_4_0 : Int := 0
  let w_5_0 : Bool := decide (w_3_0 = w_4_0)
  let w_6_0 : Int := if w_5_0 then 1 else 0
  let w_7_0 : Int := 0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Int := (w_7_0 + w_8_0)
  let w_10_0 : Int := 0
  let w_11_0 : Int := 0
  let w_12_0 : Int := (w_10_0 + w_11_0)
  let w_13_0 : Int := 1
  let w_14_0 : Int := 0
  let w_15_0 : Int := (w_13_0 + w_14_0)
  let w_16_0 := witness.w_16_0
  let w_17_0 : Int := 0
  let w_18_0 : Bool := decide (w_16_0 = w_17_0)
  let w_19_0 : Int := if w_18_0 then 1 else 0
  let w_20_0 : Int := (w_6_0 + w_19_0)
  let w_21_0 : Int := 2
  let w_22_0 : Bool := decide (w_20_0 = w_21_0)
  generatedRoot.constraints_0 w_0_0 w_1_0 w_2_0 w_8_0 w_9_0 w_12_0 w_15_0 w_16_0 w_22_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Fin 1 → Int) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Requirement_0
