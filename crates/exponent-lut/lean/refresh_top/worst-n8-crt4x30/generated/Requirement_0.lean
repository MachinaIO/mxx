import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Requirement_0

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_6.constraints_0 (w_2_0 : Fin 5 → Int) (w_27_0 : Fin 5 → Int) (w_37_0 : Fin 5 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_28_0 : Int) (w_38_0 : Int) (w_46_0 : Int) (w_53_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  5 = 5 ∧
  MxxRuntime.select w_3_0 [w_7_0, w_10_0, w_13_0, w_16_0, w_19_0] w_20_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_27_0 w_3_0 w_28_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_37_0 w_3_0 w_38_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_46_0 ∧
  outputs = w_53_0

def sequential_generatedRoot_6 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 5 → Int := inputs.2.1
  let w_27_0 : Fin 5 → Int := inputs.2.2.1
  let w_37_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : Int),
    let w_5_0 : Int := 3
    let w_6_0 : Int := 0
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_8_0 : Int := 3
    let w_9_0 : Int := 0
    let w_10_0 : Int := (w_8_0 + w_9_0)
    let w_11_0 : Int := 3
    let w_12_0 : Int := 0
    let w_13_0 : Int := (w_11_0 + w_12_0)
    let w_14_0 : Int := 6
    let w_15_0 : Int := 0
    let w_16_0 : Int := (w_14_0 + w_15_0)
    let w_17_0 : Int := 0
    let w_18_0 : Int := 0
    let w_19_0 : Int := (w_17_0 + w_18_0)
    ∃ (w_20_0 : Int),
      let w_21_0 : Bool := decide (w_4_0 = w_20_0)
      let w_22_0 : Int := if w_21_0 then 1 else 0
      let w_23_0 : Int := (w_1_0 + w_22_0)
      let w_24_0 : Int := 2
      let w_25_0 : Bool := decide (w_23_0 = w_24_0)
      let w_26_0 : Int := if w_25_0 then 1 else 0
      ∃ (w_28_0 : Int),
        let w_29_0 : Int := (w_28_0 * w_28_0)
        let w_30_0 : Int := 1
        let w_31_0 : Bool := decide (w_29_0 = w_30_0)
        let w_32_0 : Int := if w_31_0 then 1 else 0
        let w_33_0 : Int := (w_26_0 + w_32_0)
        let w_34_0 : Int := 2
        let w_35_0 : Bool := decide (w_33_0 = w_34_0)
        let w_36_0 : Int := if w_35_0 then 1 else 0
        ∃ (w_38_0 : Int),
          let w_39_0 : Int := 2
          let w_40_0 : Int := (w_38_0 * w_39_0)
          let w_41_0 : Int := 1
          let w_42_0 : Int := (w_41_0 - w_28_0)
          let w_43_0 : Int := 8
          let w_44_0 : Int := (w_42_0 * w_43_0)
          let w_45_0 : Int := (w_40_0 + w_44_0)
          ∃ (w_46_0 : Int),
            let w_47_0 : Int := 4
            let w_48_0 : Int := (w_46_0 * w_47_0)
            let w_49_0 : Bool := decide (w_45_0 = w_48_0)
            let w_50_0 : Int := if w_49_0 then 1 else 0
            let w_51_0 : Int := (w_36_0 + w_50_0)
            let w_52_0 : Int := 2
            let w_53_0 : Bool := decide (w_51_0 = w_52_0)
            sequential_generatedRoot_6.constraints_0 w_2_0 w_27_0 w_37_0 w_3_0 w_4_0 w_7_0 w_10_0 w_13_0 w_16_0 w_19_0 w_20_0 w_28_0 w_38_0 w_46_0 w_53_0 outputs

abbrev sequential_sequential_generatedRoot_8_5.constraints_0 (w_1_0 : Fin 240 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 240 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_8_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 240 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 240 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_8_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_8.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_8.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 240 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_8_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_8.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 240 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_8.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 240 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_8.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_8 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 240 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_8.Witness,
    sequential_generatedRoot_8.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_10_5.constraints_0 (w_1_0 : Fin 240 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 240 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_10_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 240 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 240 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_10_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_10.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_10.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 240 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_10_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_10.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 240 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_10.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 240 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_10.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_10 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 240 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_10.Witness,
    sequential_generatedRoot_10.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_12_5.constraints_0 (w_1_0 : Fin 240 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 240 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_12_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 240 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 240 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_12_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_12.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_12.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 240 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_12_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_12.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 240 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_12.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 240 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_12.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_12 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 240 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_12.Witness,
    sequential_generatedRoot_12.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_14_5.constraints_0 (w_1_0 : Fin 240 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 240 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_14_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 240 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 240 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_14_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_14.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_14.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 240 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_14_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_14.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 240 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_14.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 240 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_14.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_14 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 240 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_14.Witness,
    sequential_generatedRoot_14.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_16_5.constraints_0 (w_1_0 : Fin 40 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 40 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_16_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 40 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 40 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_16_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_16.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_16.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 40 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_16_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_16.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 40 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_16.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 40 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_16.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_16 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 40 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_16.Witness,
    sequential_generatedRoot_16.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_18_5.constraints_0 (w_1_0 : Fin 40 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 40 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_18_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 40 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 40 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_18_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_18.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_18.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 40 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_18_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_18.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 40 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_18.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 40 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_18.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_18 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 40 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_18.Witness,
    sequential_generatedRoot_18.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_20_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_20_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_20_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_20.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_20.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_20_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_20.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_20.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_20.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_20 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_20.Witness,
    sequential_generatedRoot_20.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_22_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_22_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_22_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_22.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_22.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_22_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_22.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_22.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_22.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_22 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_22.Witness,
    sequential_generatedRoot_22.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_24_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_24_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_24_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_24.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_24.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_24_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_24.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_24.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_24.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_24 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_24.Witness,
    sequential_generatedRoot_24.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_26_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_26_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_26_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_26.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_26.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_26_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_26.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_26.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_26.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_26 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_26.Witness,
    sequential_generatedRoot_26.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_28_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_28_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_28_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_28.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_28.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_28_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_28.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_28.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_28.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_28 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_28.Witness,
    sequential_generatedRoot_28.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_30_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_30_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_30_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_30.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_30.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_30_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_30.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_30.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_30.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_30 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_30.Witness,
    sequential_generatedRoot_30.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_32_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_32_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_32_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_32.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_32.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_32_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_32.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_32.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_32.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_32 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_32.Witness,
    sequential_generatedRoot_32.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_34_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_34_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_34_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_34.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_34.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_34_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_34.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_34.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_34.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_34 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_34.Witness,
    sequential_generatedRoot_34.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_36_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_36_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_36_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_36.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_36.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_36_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_36.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_36.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_36.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_36 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_36.Witness,
    sequential_generatedRoot_36.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_38_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_38_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_38_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_38.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_38.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_38_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_38.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_38.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_38.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_38 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_38.Witness,
    sequential_generatedRoot_38.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_40_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_40_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_40_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_40.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_40.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_40_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_40.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_40.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_40.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_40 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_40.Witness,
    sequential_generatedRoot_40.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_42_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_42_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_42_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_42.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_42.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_42_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_42.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_42.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_42.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_42 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_42.Witness,
    sequential_generatedRoot_42.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_44_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_44_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_44_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_44.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_44.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_44_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_44.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_44.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_44.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_44 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_44.Witness,
    sequential_generatedRoot_44.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_46_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_46_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_46_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_46.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_46.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_46_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_46.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_46.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_46.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_46 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_46.Witness,
    sequential_generatedRoot_46.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_48_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_48_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_48_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_48.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_48.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_48_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_48.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_48.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_48.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_48 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_48.Witness,
    sequential_generatedRoot_48.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_50_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_50_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_50_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_50.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_50.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_50_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_50.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_50.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_50.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_50 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_50.Witness,
    sequential_generatedRoot_50.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_52_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_52_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_52_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_52.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_52.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_52_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_52.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_52.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_52.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_52 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_52.Witness,
    sequential_generatedRoot_52.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_54_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_54_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_54_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_54.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_54.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_54_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_54.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_54.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_54.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_54 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_54.Witness,
    sequential_generatedRoot_54.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_56_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_56_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_56_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_56.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_56.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_56_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_56.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_56.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_56.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_56 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_56.Witness,
    sequential_generatedRoot_56.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_58_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_58_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_58_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_58.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_58.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_58_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_58.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_58.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_58.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_58 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_58.Witness,
    sequential_generatedRoot_58.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_60_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_60_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_60_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_60.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_60.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_60_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_60.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_60.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_60.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_60 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_60.Witness,
    sequential_generatedRoot_60.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_62_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_62_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_62_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_62.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_62.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_62_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_62.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_62.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_62.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_62 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_62.Witness,
    sequential_generatedRoot_62.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_64_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_64_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_64_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_64.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_64.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_64_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_64.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_64.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_64.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_64 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_64.Witness,
    sequential_generatedRoot_64.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_66_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_66_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 800 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    sequential_sequential_generatedRoot_66_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_66.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_66.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_66_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_66.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_66.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 800 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_2_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := 1
  let w_7_0 : Bool := decide (w_5_0 = w_6_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_1_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  sequential_generatedRoot_66.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_66 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_66.Witness,
    sequential_generatedRoot_66.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_78_10.constraints_0 (w_1_0 : Fin 40 → Int) (w_8_0 : Fin 5 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_11_0 : Int) (w_14_0 : Int) (w_17_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_26_0 : Int) (w_29_0 : Int) (w_32_0 : Int) (w_35_0 : Int) (w_38_0 : Int) (w_41_0 : Int) (w_44_0 : Int) (w_47_0 : Int) (w_50_0 : Int) (w_53_0 : Int) (w_56_0 : Int) (w_59_0 : Int) (w_62_0 : Int) (w_65_0 : Int) (w_68_0 : Int) (w_71_0 : Int) (w_74_0 : Int) (w_77_0 : Int) (w_80_0 : Int) (w_83_0 : Int) (w_86_0 : Int) (w_89_0 : Int) (w_92_0 : Int) (w_95_0 : Int) (w_98_0 : Int) (w_101_0 : Int) (w_104_0 : Int) (w_107_0 : Int) (w_110_0 : Int) (w_113_0 : Int) (w_116_0 : Int) (w_119_0 : Int) (w_122_0 : Int) (w_125_0 : Int) (w_128_0 : Int) (w_129_0 : Int) (w_130_0 : Int) (w_132_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 40 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 40 ∧
  40 = 40 ∧
  MxxRuntime.select w_6_0 [w_11_0, w_14_0, w_17_0, w_20_0, w_23_0, w_26_0, w_29_0, w_32_0, w_35_0, w_38_0, w_41_0, w_44_0, w_47_0, w_50_0, w_53_0, w_56_0, w_59_0, w_62_0, w_65_0, w_68_0, w_71_0, w_74_0, w_77_0, w_80_0, w_83_0, w_86_0, w_89_0, w_92_0, w_95_0, w_98_0, w_101_0, w_104_0, w_107_0, w_110_0, w_113_0, w_116_0, w_119_0, w_122_0, w_125_0, w_128_0] w_129_0 ∧
  0 ≤ w_129_0 ∧
  w_129_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_129_0 w_130_0 ∧
  outputs = w_132_0

def sequential_sequential_generatedRoot_78_10 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 40 → Int) × Int × (Fin 5 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 40 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_8_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_9_0 : Int := 0
    let w_10_0 : Int := 0
    let w_11_0 : Int := (w_9_0 + w_10_0)
    let w_12_0 : Int := 1
    let w_13_0 : Int := 0
    let w_14_0 : Int := (w_12_0 + w_13_0)
    let w_15_0 : Int := 2
    let w_16_0 : Int := 0
    let w_17_0 : Int := (w_15_0 + w_16_0)
    let w_18_0 : Int := 3
    let w_19_0 : Int := 0
    let w_20_0 : Int := (w_18_0 + w_19_0)
    let w_21_0 : Int := 4
    let w_22_0 : Int := 0
    let w_23_0 : Int := (w_21_0 + w_22_0)
    let w_24_0 : Int := 0
    let w_25_0 : Int := 0
    let w_26_0 : Int := (w_24_0 + w_25_0)
    let w_27_0 : Int := 1
    let w_28_0 : Int := 0
    let w_29_0 : Int := (w_27_0 + w_28_0)
    let w_30_0 : Int := 2
    let w_31_0 : Int := 0
    let w_32_0 : Int := (w_30_0 + w_31_0)
    let w_33_0 : Int := 3
    let w_34_0 : Int := 0
    let w_35_0 : Int := (w_33_0 + w_34_0)
    let w_36_0 : Int := 4
    let w_37_0 : Int := 0
    let w_38_0 : Int := (w_36_0 + w_37_0)
    let w_39_0 : Int := 0
    let w_40_0 : Int := 0
    let w_41_0 : Int := (w_39_0 + w_40_0)
    let w_42_0 : Int := 1
    let w_43_0 : Int := 0
    let w_44_0 : Int := (w_42_0 + w_43_0)
    let w_45_0 : Int := 2
    let w_46_0 : Int := 0
    let w_47_0 : Int := (w_45_0 + w_46_0)
    let w_48_0 : Int := 3
    let w_49_0 : Int := 0
    let w_50_0 : Int := (w_48_0 + w_49_0)
    let w_51_0 : Int := 4
    let w_52_0 : Int := 0
    let w_53_0 : Int := (w_51_0 + w_52_0)
    let w_54_0 : Int := 0
    let w_55_0 : Int := 0
    let w_56_0 : Int := (w_54_0 + w_55_0)
    let w_57_0 : Int := 1
    let w_58_0 : Int := 0
    let w_59_0 : Int := (w_57_0 + w_58_0)
    let w_60_0 : Int := 2
    let w_61_0 : Int := 0
    let w_62_0 : Int := (w_60_0 + w_61_0)
    let w_63_0 : Int := 3
    let w_64_0 : Int := 0
    let w_65_0 : Int := (w_63_0 + w_64_0)
    let w_66_0 : Int := 4
    let w_67_0 : Int := 0
    let w_68_0 : Int := (w_66_0 + w_67_0)
    let w_69_0 : Int := 0
    let w_70_0 : Int := 0
    let w_71_0 : Int := (w_69_0 + w_70_0)
    let w_72_0 : Int := 1
    let w_73_0 : Int := 0
    let w_74_0 : Int := (w_72_0 + w_73_0)
    let w_75_0 : Int := 2
    let w_76_0 : Int := 0
    let w_77_0 : Int := (w_75_0 + w_76_0)
    let w_78_0 : Int := 3
    let w_79_0 : Int := 0
    let w_80_0 : Int := (w_78_0 + w_79_0)
    let w_81_0 : Int := 4
    let w_82_0 : Int := 0
    let w_83_0 : Int := (w_81_0 + w_82_0)
    let w_84_0 : Int := 0
    let w_85_0 : Int := 0
    let w_86_0 : Int := (w_84_0 + w_85_0)
    let w_87_0 : Int := 1
    let w_88_0 : Int := 0
    let w_89_0 : Int := (w_87_0 + w_88_0)
    let w_90_0 : Int := 2
    let w_91_0 : Int := 0
    let w_92_0 : Int := (w_90_0 + w_91_0)
    let w_93_0 : Int := 3
    let w_94_0 : Int := 0
    let w_95_0 : Int := (w_93_0 + w_94_0)
    let w_96_0 : Int := 4
    let w_97_0 : Int := 0
    let w_98_0 : Int := (w_96_0 + w_97_0)
    let w_99_0 : Int := 0
    let w_100_0 : Int := 0
    let w_101_0 : Int := (w_99_0 + w_100_0)
    let w_102_0 : Int := 1
    let w_103_0 : Int := 0
    let w_104_0 : Int := (w_102_0 + w_103_0)
    let w_105_0 : Int := 2
    let w_106_0 : Int := 0
    let w_107_0 : Int := (w_105_0 + w_106_0)
    let w_108_0 : Int := 3
    let w_109_0 : Int := 0
    let w_110_0 : Int := (w_108_0 + w_109_0)
    let w_111_0 : Int := 4
    let w_112_0 : Int := 0
    let w_113_0 : Int := (w_111_0 + w_112_0)
    let w_114_0 : Int := 0
    let w_115_0 : Int := 0
    let w_116_0 : Int := (w_114_0 + w_115_0)
    let w_117_0 : Int := 1
    let w_118_0 : Int := 0
    let w_119_0 : Int := (w_117_0 + w_118_0)
    let w_120_0 : Int := 2
    let w_121_0 : Int := 0
    let w_122_0 : Int := (w_120_0 + w_121_0)
    let w_123_0 : Int := 3
    let w_124_0 : Int := 0
    let w_125_0 : Int := (w_123_0 + w_124_0)
    let w_126_0 : Int := 4
    let w_127_0 : Int := 0
    let w_128_0 : Int := (w_126_0 + w_127_0)
    ∃ (w_129_0 : Int),
      ∃ (w_130_0 : Int),
        let w_131_0 : Int := (w_7_0 * w_130_0)
        let w_132_0 : Int := (w_0_0 + w_131_0)
        sequential_sequential_generatedRoot_78_10.constraints_0 w_1_0 w_8_0 w_6_0 w_7_0 w_11_0 w_14_0 w_17_0 w_20_0 w_23_0 w_26_0 w_29_0 w_32_0 w_35_0 w_38_0 w_41_0 w_44_0 w_47_0 w_50_0 w_53_0 w_56_0 w_59_0 w_62_0 w_65_0 w_68_0 w_71_0 w_74_0 w_77_0 w_80_0 w_83_0 w_86_0 w_89_0 w_92_0 w_95_0 w_98_0 w_101_0 w_104_0 w_107_0 w_110_0 w_113_0 w_116_0 w_119_0 w_122_0 w_125_0 w_128_0 w_129_0 w_130_0 w_132_0 outputs

abbrev sequential_sequential_generatedRoot_78_14.constraints_0 (w_1_0 : Fin 40 → Int) (w_8_0 : Fin 5 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_11_0 : Int) (w_14_0 : Int) (w_17_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_26_0 : Int) (w_29_0 : Int) (w_32_0 : Int) (w_35_0 : Int) (w_38_0 : Int) (w_41_0 : Int) (w_44_0 : Int) (w_47_0 : Int) (w_50_0 : Int) (w_53_0 : Int) (w_56_0 : Int) (w_59_0 : Int) (w_62_0 : Int) (w_65_0 : Int) (w_68_0 : Int) (w_71_0 : Int) (w_74_0 : Int) (w_77_0 : Int) (w_80_0 : Int) (w_83_0 : Int) (w_86_0 : Int) (w_89_0 : Int) (w_92_0 : Int) (w_95_0 : Int) (w_98_0 : Int) (w_101_0 : Int) (w_104_0 : Int) (w_107_0 : Int) (w_110_0 : Int) (w_113_0 : Int) (w_116_0 : Int) (w_119_0 : Int) (w_122_0 : Int) (w_125_0 : Int) (w_128_0 : Int) (w_129_0 : Int) (w_130_0 : Int) (w_132_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 40 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 40 ∧
  40 = 40 ∧
  MxxRuntime.select w_6_0 [w_11_0, w_14_0, w_17_0, w_20_0, w_23_0, w_26_0, w_29_0, w_32_0, w_35_0, w_38_0, w_41_0, w_44_0, w_47_0, w_50_0, w_53_0, w_56_0, w_59_0, w_62_0, w_65_0, w_68_0, w_71_0, w_74_0, w_77_0, w_80_0, w_83_0, w_86_0, w_89_0, w_92_0, w_95_0, w_98_0, w_101_0, w_104_0, w_107_0, w_110_0, w_113_0, w_116_0, w_119_0, w_122_0, w_125_0, w_128_0] w_129_0 ∧
  0 ≤ w_129_0 ∧
  w_129_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_129_0 w_130_0 ∧
  outputs = w_132_0

def sequential_sequential_generatedRoot_78_14 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 40 → Int) × Int × (Fin 5 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 40 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_8_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_9_0 : Int := 0
    let w_10_0 : Int := 0
    let w_11_0 : Int := (w_9_0 + w_10_0)
    let w_12_0 : Int := 1
    let w_13_0 : Int := 0
    let w_14_0 : Int := (w_12_0 + w_13_0)
    let w_15_0 : Int := 2
    let w_16_0 : Int := 0
    let w_17_0 : Int := (w_15_0 + w_16_0)
    let w_18_0 : Int := 3
    let w_19_0 : Int := 0
    let w_20_0 : Int := (w_18_0 + w_19_0)
    let w_21_0 : Int := 4
    let w_22_0 : Int := 0
    let w_23_0 : Int := (w_21_0 + w_22_0)
    let w_24_0 : Int := 0
    let w_25_0 : Int := 0
    let w_26_0 : Int := (w_24_0 + w_25_0)
    let w_27_0 : Int := 1
    let w_28_0 : Int := 0
    let w_29_0 : Int := (w_27_0 + w_28_0)
    let w_30_0 : Int := 2
    let w_31_0 : Int := 0
    let w_32_0 : Int := (w_30_0 + w_31_0)
    let w_33_0 : Int := 3
    let w_34_0 : Int := 0
    let w_35_0 : Int := (w_33_0 + w_34_0)
    let w_36_0 : Int := 4
    let w_37_0 : Int := 0
    let w_38_0 : Int := (w_36_0 + w_37_0)
    let w_39_0 : Int := 0
    let w_40_0 : Int := 0
    let w_41_0 : Int := (w_39_0 + w_40_0)
    let w_42_0 : Int := 1
    let w_43_0 : Int := 0
    let w_44_0 : Int := (w_42_0 + w_43_0)
    let w_45_0 : Int := 2
    let w_46_0 : Int := 0
    let w_47_0 : Int := (w_45_0 + w_46_0)
    let w_48_0 : Int := 3
    let w_49_0 : Int := 0
    let w_50_0 : Int := (w_48_0 + w_49_0)
    let w_51_0 : Int := 4
    let w_52_0 : Int := 0
    let w_53_0 : Int := (w_51_0 + w_52_0)
    let w_54_0 : Int := 0
    let w_55_0 : Int := 0
    let w_56_0 : Int := (w_54_0 + w_55_0)
    let w_57_0 : Int := 1
    let w_58_0 : Int := 0
    let w_59_0 : Int := (w_57_0 + w_58_0)
    let w_60_0 : Int := 2
    let w_61_0 : Int := 0
    let w_62_0 : Int := (w_60_0 + w_61_0)
    let w_63_0 : Int := 3
    let w_64_0 : Int := 0
    let w_65_0 : Int := (w_63_0 + w_64_0)
    let w_66_0 : Int := 4
    let w_67_0 : Int := 0
    let w_68_0 : Int := (w_66_0 + w_67_0)
    let w_69_0 : Int := 0
    let w_70_0 : Int := 0
    let w_71_0 : Int := (w_69_0 + w_70_0)
    let w_72_0 : Int := 1
    let w_73_0 : Int := 0
    let w_74_0 : Int := (w_72_0 + w_73_0)
    let w_75_0 : Int := 2
    let w_76_0 : Int := 0
    let w_77_0 : Int := (w_75_0 + w_76_0)
    let w_78_0 : Int := 3
    let w_79_0 : Int := 0
    let w_80_0 : Int := (w_78_0 + w_79_0)
    let w_81_0 : Int := 4
    let w_82_0 : Int := 0
    let w_83_0 : Int := (w_81_0 + w_82_0)
    let w_84_0 : Int := 0
    let w_85_0 : Int := 0
    let w_86_0 : Int := (w_84_0 + w_85_0)
    let w_87_0 : Int := 1
    let w_88_0 : Int := 0
    let w_89_0 : Int := (w_87_0 + w_88_0)
    let w_90_0 : Int := 2
    let w_91_0 : Int := 0
    let w_92_0 : Int := (w_90_0 + w_91_0)
    let w_93_0 : Int := 3
    let w_94_0 : Int := 0
    let w_95_0 : Int := (w_93_0 + w_94_0)
    let w_96_0 : Int := 4
    let w_97_0 : Int := 0
    let w_98_0 : Int := (w_96_0 + w_97_0)
    let w_99_0 : Int := 0
    let w_100_0 : Int := 0
    let w_101_0 : Int := (w_99_0 + w_100_0)
    let w_102_0 : Int := 1
    let w_103_0 : Int := 0
    let w_104_0 : Int := (w_102_0 + w_103_0)
    let w_105_0 : Int := 2
    let w_106_0 : Int := 0
    let w_107_0 : Int := (w_105_0 + w_106_0)
    let w_108_0 : Int := 3
    let w_109_0 : Int := 0
    let w_110_0 : Int := (w_108_0 + w_109_0)
    let w_111_0 : Int := 4
    let w_112_0 : Int := 0
    let w_113_0 : Int := (w_111_0 + w_112_0)
    let w_114_0 : Int := 0
    let w_115_0 : Int := 0
    let w_116_0 : Int := (w_114_0 + w_115_0)
    let w_117_0 : Int := 1
    let w_118_0 : Int := 0
    let w_119_0 : Int := (w_117_0 + w_118_0)
    let w_120_0 : Int := 2
    let w_121_0 : Int := 0
    let w_122_0 : Int := (w_120_0 + w_121_0)
    let w_123_0 : Int := 3
    let w_124_0 : Int := 0
    let w_125_0 : Int := (w_123_0 + w_124_0)
    let w_126_0 : Int := 4
    let w_127_0 : Int := 0
    let w_128_0 : Int := (w_126_0 + w_127_0)
    ∃ (w_129_0 : Int),
      ∃ (w_130_0 : Int),
        let w_131_0 : Int := (w_7_0 * w_130_0)
        let w_132_0 : Int := (w_0_0 + w_131_0)
        sequential_sequential_generatedRoot_78_14.constraints_0 w_1_0 w_8_0 w_6_0 w_7_0 w_11_0 w_14_0 w_17_0 w_20_0 w_23_0 w_26_0 w_29_0 w_32_0 w_35_0 w_38_0 w_41_0 w_44_0 w_47_0 w_50_0 w_53_0 w_56_0 w_59_0 w_62_0 w_65_0 w_68_0 w_71_0 w_74_0 w_77_0 w_80_0 w_83_0 w_86_0 w_89_0 w_92_0 w_95_0 w_98_0 w_101_0 w_104_0 w_107_0 w_110_0 w_113_0 w_116_0 w_119_0 w_122_0 w_125_0 w_128_0 w_129_0 w_130_0 w_132_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_78.Witness where
  w_4_0 : Int
  w_10_0 : Int
  w_14_0 : Int

abbrev sequential_generatedRoot_78.constraints_0 (params : Params) (i_0 : Nat) (w_2_0 : Fin 8 → Int) (w_8_0 : Fin 40 → Int) (w_9_0 : Fin 5 → Int) (w_13_0 : Fin 40 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_12_0 : Int) (w_14_0 : Int) (w_16_0 : Int) (w_18_0 : Int) (w_25_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_78_10 params i_0 i (current, (w_8_0, (w_3_0, (w_9_0, ())))) next) (Int.toNat (5)) w_7_0 w_10_0 ∧
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_78_14 params i_0 i (current, (w_13_0, (w_3_0, (w_9_0, ())))) next) (Int.toNat (5)) w_12_0 w_14_0 ∧
  w_16_0 ≠ 0 ∧
  w_18_0 ≠ 0 ∧
  outputs = w_25_0

abbrev sequential_generatedRoot_78.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8 → Int) × (Fin 40 → Int) × (Fin 5 → Int) × (Fin 40 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_78.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 8 → Int := inputs.2.1
  let w_8_0 : Fin 40 → Int := inputs.2.2.1
  let w_9_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_13_0 : Fin 40 → Int := inputs.2.2.2.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := 1
  let w_6_0 : Int := 0
  let w_7_0 : Int := 0
  let w_10_0 := witness.w_10_0
  let w_11_0 : Int := (w_6_0 + w_10_0)
  let w_12_0 : Int := 0
  let w_14_0 := witness.w_14_0
  let w_15_0 : Int := (w_11_0 + w_14_0)
  let w_16_0 : Int := 8
  let w_17_0 : Int := (w_15_0 % w_16_0)
  let w_18_0 : Int := 4
  let w_19_0 : Int := (w_17_0 / w_18_0)
  let w_20_0 : Int := (w_5_0 - w_19_0)
  let w_21_0 : Bool := decide (w_4_0 = w_20_0)
  let w_22_0 : Int := if w_21_0 then 1 else 0
  let w_23_0 : Int := (w_1_0 + w_22_0)
  let w_24_0 : Int := 2
  let w_25_0 : Bool := decide (w_23_0 = w_24_0)
  sequential_generatedRoot_78.constraints_0 params i_0 w_2_0 w_8_0 w_9_0 w_13_0 w_3_0 w_4_0 w_7_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_25_0 outputs

def sequential_generatedRoot_78 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8 → Int) × (Fin 40 → Int) × (Fin 5 → Int) × (Fin 40 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_78.Witness,
    sequential_generatedRoot_78.body params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_6_0 : Bool
  w_8_0 : Bool
  w_10_0 : Bool
  w_12_0 : Bool
  w_14_0 : Bool
  w_16_0 : Bool
  w_18_0 : Bool
  w_20_0 : Bool
  w_22_0 : Bool
  w_24_0 : Bool
  w_26_0 : Bool
  w_28_0 : Bool
  w_30_0 : Bool
  w_32_0 : Bool
  w_34_0 : Bool
  w_36_0 : Bool
  w_38_0 : Bool
  w_40_0 : Bool
  w_42_0 : Bool
  w_44_0 : Bool
  w_46_0 : Bool
  w_48_0 : Bool
  w_50_0 : Bool
  w_52_0 : Bool
  w_54_0 : Bool
  w_56_0 : Bool
  w_58_0 : Bool
  w_60_0 : Bool
  w_62_0 : Bool
  w_64_0 : Bool
  w_66_0 : Bool
  w_78_0 : Bool

abbrev generatedRoot.constraints_1 (params : Params) (w_3_0 : Fin 5 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_37_0 : Fin 800 → Int) (w_39_0 : Fin 800 → Int) (w_41_0 : Fin 800 → Int) (w_43_0 : Fin 800 → Int) (w_45_0 : Fin 800 → Int) (w_47_0 : Fin 800 → Int) (w_49_0 : Fin 800 → Int) (w_51_0 : Fin 800 → Int) (w_53_0 : Fin 800 → Int) (w_55_0 : Fin 800 → Int) (w_57_0 : Fin 800 → Int) (w_59_0 : Fin 800 → Int) (w_61_0 : Fin 800 → Int) (w_63_0 : Fin 800 → Int) (w_65_0 : Fin 800 → Int) (w_77_0 : Fin 8 → Int) (w_36_0 : Bool) (w_38_0 : Bool) (w_40_0 : Bool) (w_42_0 : Bool) (w_44_0 : Bool) (w_46_0 : Bool) (w_48_0 : Bool) (w_50_0 : Bool) (w_52_0 : Bool) (w_54_0 : Bool) (w_56_0 : Bool) (w_58_0 : Bool) (w_60_0 : Bool) (w_62_0 : Bool) (w_64_0 : Bool) (w_66_0 : Bool) (w_76_0 : Bool) (w_78_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_38 params i (current, (w_37_0, ())) next) (Int.toNat (160)) w_36_0 w_38_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_40 params i (current, (w_39_0, ())) next) (Int.toNat (160)) w_38_0 w_40_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_42 params i (current, (w_41_0, ())) next) (Int.toNat (160)) w_40_0 w_42_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_44 params i (current, (w_43_0, ())) next) (Int.toNat (160)) w_42_0 w_44_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_46 params i (current, (w_45_0, ())) next) (Int.toNat (160)) w_44_0 w_46_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_48 params i (current, (w_47_0, ())) next) (Int.toNat (160)) w_46_0 w_48_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_50 params i (current, (w_49_0, ())) next) (Int.toNat (160)) w_48_0 w_50_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_52 params i (current, (w_51_0, ())) next) (Int.toNat (160)) w_50_0 w_52_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_54 params i (current, (w_53_0, ())) next) (Int.toNat (160)) w_52_0 w_54_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_56 params i (current, (w_55_0, ())) next) (Int.toNat (160)) w_54_0 w_56_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_58 params i (current, (w_57_0, ())) next) (Int.toNat (160)) w_56_0 w_58_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_60 params i (current, (w_59_0, ())) next) (Int.toNat (160)) w_58_0 w_60_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_62 params i (current, (w_61_0, ())) next) (Int.toNat (160)) w_60_0 w_62_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_64 params i (current, (w_63_0, ())) next) (Int.toNat (160)) w_62_0 w_64_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_66 params i (current, (w_65_0, ())) next) (Int.toNat (160)) w_64_0 w_66_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_78 params i (current, (w_77_0, (w_15_0, (w_3_0, (w_17_0, ()))))) next) (Int.toNat (8)) w_76_0 w_78_0 ∧
  outputs = w_78_0

abbrev generatedRoot.constraints_0 (params : Params) (w_3_0 : Fin 5 → Int) (w_4_0 : Fin 5 → Int) (w_5_0 : Fin 5 → Int) (w_7_0 : Fin 240 → Int) (w_9_0 : Fin 240 → Int) (w_11_0 : Fin 240 → Int) (w_13_0 : Fin 240 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_19_0 : Fin 800 → Int) (w_21_0 : Fin 800 → Int) (w_23_0 : Fin 800 → Int) (w_25_0 : Fin 800 → Int) (w_27_0 : Fin 800 → Int) (w_29_0 : Fin 800 → Int) (w_31_0 : Fin 800 → Int) (w_33_0 : Fin 800 → Int) (w_35_0 : Fin 800 → Int) (w_37_0 : Fin 800 → Int) (w_39_0 : Fin 800 → Int) (w_41_0 : Fin 800 → Int) (w_43_0 : Fin 800 → Int) (w_45_0 : Fin 800 → Int) (w_47_0 : Fin 800 → Int) (w_49_0 : Fin 800 → Int) (w_51_0 : Fin 800 → Int) (w_53_0 : Fin 800 → Int) (w_55_0 : Fin 800 → Int) (w_57_0 : Fin 800 → Int) (w_59_0 : Fin 800 → Int) (w_61_0 : Fin 800 → Int) (w_63_0 : Fin 800 → Int) (w_65_0 : Fin 800 → Int) (w_77_0 : Fin 8 → Int) (w_2_0 : Bool) (w_6_0 : Bool) (w_8_0 : Bool) (w_10_0 : Bool) (w_12_0 : Bool) (w_14_0 : Bool) (w_16_0 : Bool) (w_18_0 : Bool) (w_20_0 : Bool) (w_22_0 : Bool) (w_24_0 : Bool) (w_26_0 : Bool) (w_28_0 : Bool) (w_30_0 : Bool) (w_32_0 : Bool) (w_34_0 : Bool) (w_36_0 : Bool) (w_38_0 : Bool) (w_40_0 : Bool) (w_42_0 : Bool) (w_44_0 : Bool) (w_46_0 : Bool) (w_48_0 : Bool) (w_50_0 : Bool) (w_52_0 : Bool) (w_54_0 : Bool) (w_56_0 : Bool) (w_58_0 : Bool) (w_60_0 : Bool) (w_62_0 : Bool) (w_64_0 : Bool) (w_66_0 : Bool) (w_76_0 : Bool) (w_78_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 params i (current, (w_3_0, (w_4_0, (w_5_0, ())))) next) (Int.toNat (5)) w_2_0 w_6_0 ∧
  0 ≤ (48) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_8 params i (current, (w_7_0, ())) next) (Int.toNat (48)) w_6_0 w_8_0 ∧
  0 ≤ (48) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_10 params i (current, (w_9_0, ())) next) (Int.toNat (48)) w_8_0 w_10_0 ∧
  0 ≤ (48) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_12 params i (current, (w_11_0, ())) next) (Int.toNat (48)) w_10_0 w_12_0 ∧
  0 ≤ (48) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_14 params i (current, (w_13_0, ())) next) (Int.toNat (48)) w_12_0 w_14_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_16 params i (current, (w_15_0, ())) next) (Int.toNat (8)) w_14_0 w_16_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_18 params i (current, (w_17_0, ())) next) (Int.toNat (8)) w_16_0 w_18_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_20 params i (current, (w_19_0, ())) next) (Int.toNat (160)) w_18_0 w_20_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_22 params i (current, (w_21_0, ())) next) (Int.toNat (160)) w_20_0 w_22_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_24 params i (current, (w_23_0, ())) next) (Int.toNat (160)) w_22_0 w_24_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_26 params i (current, (w_25_0, ())) next) (Int.toNat (160)) w_24_0 w_26_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_28 params i (current, (w_27_0, ())) next) (Int.toNat (160)) w_26_0 w_28_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_30 params i (current, (w_29_0, ())) next) (Int.toNat (160)) w_28_0 w_30_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_32 params i (current, (w_31_0, ())) next) (Int.toNat (160)) w_30_0 w_32_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_34 params i (current, (w_33_0, ())) next) (Int.toNat (160)) w_32_0 w_34_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_36 params i (current, (w_35_0, ())) next) (Int.toNat (160)) w_34_0 w_36_0 ∧
  generatedRoot.constraints_1 params w_3_0 w_15_0 w_17_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_77_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_76_0 w_78_0 outputs

abbrev generatedRoot.body (params : Params) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 40 → Int) × (Fin 40 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × Int × Int × (Fin 8 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_3_0 : Fin 5 → Int := inputs.1
  let w_4_0 : Fin 5 → Int := inputs.2.1
  let w_5_0 : Fin 5 → Int := inputs.2.2.1
  let w_7_0 : Fin 240 → Int := inputs.2.2.2.1
  let w_9_0 : Fin 240 → Int := inputs.2.2.2.2.1
  let w_11_0 : Fin 240 → Int := inputs.2.2.2.2.2.1
  let w_13_0 : Fin 240 → Int := inputs.2.2.2.2.2.2.1
  let w_15_0 : Fin 40 → Int := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : Fin 40 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_23_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_27_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_33_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_35_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_37_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_45_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_47_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_49_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_57_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_59_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_61_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_63_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_65_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_68_0 : Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_70_0 : Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_77_0 : Fin 8 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : Int := 0
  let w_1_0 : Int := 0
  let w_2_0 : Bool := decide (w_0_0 = w_1_0)
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_20_0 := witness.w_20_0
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_32_0 := witness.w_32_0
  let w_34_0 := witness.w_34_0
  let w_36_0 := witness.w_36_0
  let w_38_0 := witness.w_38_0
  let w_40_0 := witness.w_40_0
  let w_42_0 := witness.w_42_0
  let w_44_0 := witness.w_44_0
  let w_46_0 := witness.w_46_0
  let w_48_0 := witness.w_48_0
  let w_50_0 := witness.w_50_0
  let w_52_0 := witness.w_52_0
  let w_54_0 := witness.w_54_0
  let w_56_0 := witness.w_56_0
  let w_58_0 := witness.w_58_0
  let w_60_0 := witness.w_60_0
  let w_62_0 := witness.w_62_0
  let w_64_0 := witness.w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 : Int := if w_66_0 then 1 else 0
  let w_69_0 : Int := 1
  let w_71_0 : Int := (w_69_0 - w_70_0)
  let w_72_0 : Bool := decide (w_68_0 = w_71_0)
  let w_73_0 : Int := if w_72_0 then 1 else 0
  let w_74_0 : Int := (w_67_0 + w_73_0)
  let w_75_0 : Int := 2
  let w_76_0 : Bool := decide (w_74_0 = w_75_0)
  let w_78_0 := witness.w_78_0
  generatedRoot.constraints_0 params w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_77_0 w_2_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_76_0 w_78_0 outputs

def generatedRoot (params : Params) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 40 → Int) × (Fin 40 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × Int × Int × (Fin 8 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Requirement_0
