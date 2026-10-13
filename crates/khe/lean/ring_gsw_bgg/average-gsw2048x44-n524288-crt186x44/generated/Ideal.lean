import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Ideal

structure Params where
  «instance_width» : Int
  «witness_width» : Int
  «depth» : Int
  «max_layer_width» : Int

abbrev parallel_generatedRoot_1.constraints_0 (w_2_0 : Bool) (outputs : Bool) : Prop :=
  outputs = w_2_0

def parallel_generatedRoot_1 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : Int) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs
  let w_1_0 : Int := 1
  let w_2_0 : Bool := decide (w_0_0 = w_1_0)
  parallel_generatedRoot_1.constraints_0 w_2_0 outputs

abbrev parallel_sequential_generatedRoot_6_7.constraints_0 (w_10_0 : Fin 4817449462805760 → Int) (w_24_0 : Fin 20072706095024 → Bool) (w_25_0 : Fin 4817449462805760 → Int) (w_32_0 : Fin 4817449462805760 → Int) (w_5_0 : Int) (w_9_0 : Bool) (w_14_0 : Int) (w_15_0 : Int) (w_19_0 : Bool) (w_23_0 : Bool) (w_26_0 : Int) (w_27_0 : Bool) (w_30_0 : Bool) (w_33_0 : Int) (w_34_0 : Bool) (w_38_0 : Bool) (w_43_0 : Bool) (w_44_0 : Bool) (w_45_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_14_0 ∧
  w_14_0 < 4817449462805760 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_14_0 w_15_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4817449462805760 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_14_0 w_26_0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_26_0 w_27_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4817449462805760 ∧
  MxxRuntime.familyGetDynamic w_32_0 w_14_0 w_33_0 ∧
  0 ≤ w_33_0 ∧
  w_33_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_33_0 w_34_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 6 ∧
  6 = 6 ∧
  MxxRuntime.select w_15_0 [w_19_0, w_23_0, w_27_0, w_30_0, w_38_0, w_43_0] w_44_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_5_0 [w_9_0, w_44_0] w_45_0 ∧
  outputs = w_45_0

def parallel_sequential_generatedRoot_6_7 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 4817449462805760 → Int) × Int × (Fin 20072706095024 → Bool) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_1_0 : Int := inputs.1
  let w_10_0 : Fin 4817449462805760 → Int := inputs.2.1
  let w_11_0 : Int := inputs.2.2.1
  let w_24_0 : Fin 20072706095024 → Bool := inputs.2.2.2.1
  let w_25_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.1
  let w_32_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.2.1
  let w_0_0 : Int := (Int.ofNat i_1)
  let w_2_0 : Int := 1
  let w_3_0 : Int := (w_1_0 - w_2_0)
  let w_4_0 : Bool := decide (w_0_0 ≤ w_3_0)
  let w_5_0 : Int := if w_4_0 then 1 else 0
  let w_6_0 : Bool := false
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_8_0 : Int := 1
  let w_9_0 : Bool := decide (w_7_0 = w_8_0)
  let w_12_0 : Int := params.«max_layer_width»
  let w_13_0 : Int := (w_11_0 * w_12_0)
  let w_14_0 : Int := (w_13_0 + w_0_0)
  ∃ (w_15_0 : Int),
    let w_16_0 : Bool := false
    let w_17_0 : Int := if w_16_0 then 1 else 0
    let w_18_0 : Int := 1
    let w_19_0 : Bool := decide (w_17_0 = w_18_0)
    let w_20_0 : Bool := true
    let w_21_0 : Int := if w_20_0 then 1 else 0
    let w_22_0 : Int := 1
    let w_23_0 : Bool := decide (w_21_0 = w_22_0)
    ∃ (w_26_0 : Int),
      ∃ (w_27_0 : Bool),
        let w_28_0 : Int := if w_27_0 then 1 else 0
        let w_29_0 : Int := 0
        let w_30_0 : Bool := decide (w_28_0 = w_29_0)
        let w_31_0 : Int := if w_27_0 then 1 else 0
        ∃ (w_33_0 : Int),
          ∃ (w_34_0 : Bool),
            let w_35_0 : Int := if w_34_0 then 1 else 0
            let w_36_0 : Int := (w_31_0 + w_35_0)
            let w_37_0 : Int := 2
            let w_38_0 : Bool := decide (w_36_0 = w_37_0)
            let w_39_0 : Int := if w_27_0 then 1 else 0
            let w_40_0 : Int := if w_34_0 then 1 else 0
            let w_41_0 : Int := (w_39_0 + w_40_0)
            let w_42_0 : Int := 1
            let w_43_0 : Bool := decide (w_41_0 = w_42_0)
            ∃ (w_44_0 : Bool),
              ∃ (w_45_0 : Bool),
                parallel_sequential_generatedRoot_6_7.constraints_0 w_10_0 w_24_0 w_25_0 w_32_0 w_5_0 w_9_0 w_14_0 w_15_0 w_19_0 w_23_0 w_26_0 w_27_0 w_30_0 w_33_0 w_34_0 w_38_0 w_43_0 w_44_0 w_45_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_6.Witness where
  w_2_0 : Int
  w_7_0 : Fin 20072706095024 → Bool

abbrev sequential_generatedRoot_6.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 20072706095024 → Bool) (w_0_0 : Fin 240 → Int) (w_3_0 : Fin 4817449462805760 → Int) (w_5_0 : Fin 4817449462805760 → Int) (w_6_0 : Fin 4817449462805760 → Int) (w_1_0 : Int) (w_2_0 : Int) (w_7_0 : Fin 20072706095024 → Bool) (outputs : Fin 20072706095024 → Bool) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 240 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_sequential_generatedRoot_6_7 tape (path ++ [7, i.val]) params i_0 i (w_2_0, (w_3_0, (w_1_0, (w_4_0, (w_5_0, (w_6_0, ())))))) (w_7_0 i)) ∧
  outputs = w_7_0

abbrev sequential_generatedRoot_6.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 20072706095024 → Bool) × (Fin 240 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × Unit) (outputs : Fin 20072706095024 → Bool) (witness : sequential_generatedRoot_6.Witness) : Prop :=
  let _params := params
  let w_4_0 : Fin 20072706095024 → Bool := inputs.1
  let w_0_0 : Fin 240 → Int := inputs.2.1
  let w_3_0 : Fin 4817449462805760 → Int := inputs.2.2.1
  let w_5_0 : Fin 4817449462805760 → Int := inputs.2.2.2.1
  let w_6_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 := witness.w_2_0
  let w_7_0 := witness.w_7_0
  sequential_generatedRoot_6.constraints_0 tape path params i_0 w_4_0 w_0_0 w_3_0 w_5_0 w_6_0 w_1_0 w_2_0 w_7_0 outputs

def sequential_generatedRoot_6 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 20072706095024 → Bool) × (Fin 240 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × Unit) (outputs : Fin 20072706095024 → Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_6.Witness,
    sequential_generatedRoot_6.body tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : Fin 20072706095024 → Bool
  w_6_0 : Fin 20072706095024 → Bool
  w_8_0 : Int
  w_9_0 : Bool

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 20072706095024 → Int) (w_2_0 : Fin 240 → Int) (w_3_0 : Fin 4817449462805760 → Int) (w_4_0 : Fin 4817449462805760 → Int) (w_5_0 : Fin 4817449462805760 → Int) (w_7_0 : Fin 1 → Int) (w_1_0 : Fin 20072706095024 → Bool) (w_6_0 : Fin 20072706095024 → Bool) (w_8_0 : Int) (w_9_0 : Bool) (outputs : Bool) : Prop :=
  (params.«max_layer_width») = 20072706095024 ∧
  (∀ i : Fin 20072706095024, parallel_generatedRoot_1 tape (path ++ [1, i.val]) params i (w_0_0 i) (w_1_0 i)) ∧
  0 ≤ (params.«depth») ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Fin 20072706095024 → Bool) => sequential_generatedRoot_6 tape (path ++ [6, i]) params i (current, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ()))))) next) (Int.toNat (params.«depth»)) w_1_0 w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ w_8_0 ∧
  w_8_0 < 20072706095024 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_8_0 w_9_0 ∧
  outputs = w_9_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 20072706095024 → Int) × (Fin 240 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 1 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Fin 20072706095024 → Int := inputs.1
  let w_2_0 : Fin 240 → Int := inputs.2.1
  let w_3_0 : Fin 4817449462805760 → Int := inputs.2.2.1
  let w_4_0 : Fin 4817449462805760 → Int := inputs.2.2.2.1
  let w_5_0 : Fin 4817449462805760 → Int := inputs.2.2.2.2.1
  let w_7_0 : Fin 1 → Int := inputs.2.2.2.2.2.1
  let w_1_0 := witness.w_1_0
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  generatedRoot.constraints_0 tape path params w_0_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_1_0 w_6_0 w_8_0 w_9_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 20072706095024 → Int) × (Fin 240 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 4817449462805760 → Int) × (Fin 1 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Ideal
