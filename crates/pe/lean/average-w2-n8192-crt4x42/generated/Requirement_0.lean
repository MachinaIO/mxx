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
  w_27_0 : Int
  w_28_0 : Int
  w_31_0 : Int

abbrev generatedRoot.constraints_0 (w_0_0 : Fin 1 → Int) (w_7_0 : Fin 2 → Int) (w_26_0 : Fin 1 → Int) (w_1_0 : Int) (w_2_0 : Int) (w_8_0 : Int) (w_16_0 : Int) (w_27_0 : Int) (w_28_0 : Int) (w_30_0 : Int) (w_31_0 : Int) (w_37_0 : Bool) (outputs : Bool) : Prop :=
  MxxRuntime.familyGetStatic w_0_0 (0) w_1_0 ∧
  w_2_0 ≠ 0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (1) w_16_0 ∧
  MxxRuntime.familyGetStatic w_26_0 (0) w_27_0 ∧
  0 ≤ w_27_0 ∧
  w_27_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_27_0 w_28_0 ∧
  0 ≤ w_30_0 ∧
  w_30_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_30_0 w_31_0 ∧
  outputs = w_37_0

abbrev generatedRoot.body (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (inputs : (Fin 1 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Fin 1 → Int := inputs.1
  let w_7_0 : Fin 2 → Int := inputs.2.1
  let w_26_0 : Fin 1 → Int := inputs.2.2.1
  let w_1_0 := witness.w_1_0
  let w_2_0 : Int := 1
  let w_3_0 : Int := (w_1_0 % w_2_0)
  let w_4_0 : Int := 0
  let w_5_0 : Bool := decide (w_3_0 = w_4_0)
  let w_6_0 : Int := if w_5_0 then 1 else 0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Int := 2
  let w_10_0 : Bool := decide (w_8_0 < w_9_0)
  let w_11_0 : Int := if w_10_0 then 1 else 0
  let w_12_0 : Int := (w_6_0 + w_11_0)
  let w_13_0 : Int := 2
  let w_14_0 : Bool := decide (w_12_0 = w_13_0)
  let w_15_0 : Int := if w_14_0 then 1 else 0
  let w_16_0 := witness.w_16_0
  let w_17_0 : Int := 2
  let w_18_0 : Bool := decide (w_16_0 < w_17_0)
  let w_19_0 : Int := if w_18_0 then 1 else 0
  let w_20_0 : Int := (w_15_0 + w_19_0)
  let w_21_0 : Int := 2
  let w_22_0 : Bool := decide (w_20_0 = w_21_0)
  let w_23_0 : Int := if w_22_0 then 1 else 0
  let w_24_0 : Int := 0
  let w_25_0 : Int := 0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : Int := (w_25_0 + w_28_0)
  let w_30_0 : Int := (w_24_0 + w_29_0)
  let w_31_0 := witness.w_31_0
  let w_32_0 : Int := 0
  let w_33_0 : Bool := decide (w_31_0 = w_32_0)
  let w_34_0 : Int := if w_33_0 then 1 else 0
  let w_35_0 : Int := (w_23_0 + w_34_0)
  let w_36_0 : Int := 2
  let w_37_0 : Bool := decide (w_35_0 = w_36_0)
  generatedRoot.constraints_0 w_0_0 w_7_0 w_26_0 w_1_0 w_2_0 w_8_0 w_16_0 w_27_0 w_28_0 w_30_0 w_31_0 w_37_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 1 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Requirement_0
