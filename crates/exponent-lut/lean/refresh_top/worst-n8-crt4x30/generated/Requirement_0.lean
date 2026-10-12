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
    let w_5_0 : Int := 7
    let w_6_0 : Int := 0
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_8_0 : Int := 5
    let w_9_0 : Int := 0
    let w_10_0 : Int := (w_8_0 + w_9_0)
    let w_11_0 : Int := 2
    let w_12_0 : Int := 0
    let w_13_0 : Int := (w_11_0 + w_12_0)
    let w_14_0 : Int := 3
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

abbrev sequential_sequential_generatedRoot_68_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_68_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_68_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_68.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_68.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_68_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_68.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_68.Witness) : Prop :=
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
  sequential_generatedRoot_68.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_68 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_68.Witness,
    sequential_generatedRoot_68.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_70_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_70_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_70_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_70.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_70.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_70_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_70.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_70.Witness) : Prop :=
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
  sequential_generatedRoot_70.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_70 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_70.Witness,
    sequential_generatedRoot_70.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_72_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_72_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_72_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_72.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_72.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_72_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_72.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_72.Witness) : Prop :=
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
  sequential_generatedRoot_72.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_72 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_72.Witness,
    sequential_generatedRoot_72.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_74_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_74_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_74_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_74.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_74.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_74_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_74.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_74.Witness) : Prop :=
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
  sequential_generatedRoot_74.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_74 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_74.Witness,
    sequential_generatedRoot_74.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_76_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_76_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_76_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_76.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_76.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_76_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_76.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_76.Witness) : Prop :=
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
  sequential_generatedRoot_76.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_76 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_76.Witness,
    sequential_generatedRoot_76.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_78_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_78_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_78_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_78.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_78.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_78_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_78.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_78.Witness) : Prop :=
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
  sequential_generatedRoot_78.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_78 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_78.Witness,
    sequential_generatedRoot_78.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_80_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_80_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_80_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_80.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_80.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_80_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_80.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_80.Witness) : Prop :=
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
  sequential_generatedRoot_80.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_80 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_80.Witness,
    sequential_generatedRoot_80.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_82_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_82_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_82_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_82.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_82.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_82_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_82.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_82.Witness) : Prop :=
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
  sequential_generatedRoot_82.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_82 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_82.Witness,
    sequential_generatedRoot_82.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_84_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_84_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_84_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_84.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_84.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_84_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_84.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_84.Witness) : Prop :=
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
  sequential_generatedRoot_84.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_84 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_84.Witness,
    sequential_generatedRoot_84.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_86_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_86_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_86_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_86.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_86.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_86_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_86.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_86.Witness) : Prop :=
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
  sequential_generatedRoot_86.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_86 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_86.Witness,
    sequential_generatedRoot_86.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_88_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_88_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_88_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_88.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_88.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_88_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_88.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_88.Witness) : Prop :=
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
  sequential_generatedRoot_88.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_88 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_88.Witness,
    sequential_generatedRoot_88.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_90_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_90_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_90_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_90.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_90.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_90_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_90.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_90.Witness) : Prop :=
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
  sequential_generatedRoot_90.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_90 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_90.Witness,
    sequential_generatedRoot_90.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_92_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_92_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_92_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_92.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_92.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_92_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_92.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_92.Witness) : Prop :=
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
  sequential_generatedRoot_92.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_92 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_92.Witness,
    sequential_generatedRoot_92.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_94_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_94_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_94_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_94.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_94.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_94_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_94.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_94.Witness) : Prop :=
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
  sequential_generatedRoot_94.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_94 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_94.Witness,
    sequential_generatedRoot_94.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_96_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_96_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_96_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_96.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_96.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_96_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_96.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_96.Witness) : Prop :=
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
  sequential_generatedRoot_96.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_96 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_96.Witness,
    sequential_generatedRoot_96.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_98_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_98_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_98_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_98.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_98.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_98_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_98.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_98.Witness) : Prop :=
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
  sequential_generatedRoot_98.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_98 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_98.Witness,
    sequential_generatedRoot_98.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_100_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_100_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_100_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_100.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_100.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_100_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_100.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_100.Witness) : Prop :=
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
  sequential_generatedRoot_100.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_100 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_100.Witness,
    sequential_generatedRoot_100.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_102_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_102_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_102_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_102.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_102.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_102_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_102.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_102.Witness) : Prop :=
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
  sequential_generatedRoot_102.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_102 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_102.Witness,
    sequential_generatedRoot_102.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_104_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_104_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_104_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_104.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_104.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_104_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_104.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_104.Witness) : Prop :=
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
  sequential_generatedRoot_104.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_104 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_104.Witness,
    sequential_generatedRoot_104.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_106_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_106_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_106_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_106.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_106.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_106_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_106.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_106.Witness) : Prop :=
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
  sequential_generatedRoot_106.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_106 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_106.Witness,
    sequential_generatedRoot_106.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_108_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_108_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_108_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_108.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_108.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_108_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_108.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_108.Witness) : Prop :=
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
  sequential_generatedRoot_108.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_108 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_108.Witness,
    sequential_generatedRoot_108.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_110_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_110_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_110_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_110.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_110.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_110_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_110.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_110.Witness) : Prop :=
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
  sequential_generatedRoot_110.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_110 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_110.Witness,
    sequential_generatedRoot_110.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_112_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_112_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_112_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_112.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_112.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_112_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_112.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_112.Witness) : Prop :=
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
  sequential_generatedRoot_112.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_112 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_112.Witness,
    sequential_generatedRoot_112.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_114_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_114_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_114_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_114.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_114.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_114_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_114.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_114.Witness) : Prop :=
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
  sequential_generatedRoot_114.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_114 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_114.Witness,
    sequential_generatedRoot_114.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_116_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_116_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_116_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_116.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_116.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_116_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_116.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_116.Witness) : Prop :=
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
  sequential_generatedRoot_116.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_116 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_116.Witness,
    sequential_generatedRoot_116.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_118_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_118_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_118_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_118.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_118.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_118_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_118.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_118.Witness) : Prop :=
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
  sequential_generatedRoot_118.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_118 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_118.Witness,
    sequential_generatedRoot_118.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_120_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_120_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_120_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_120.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_120.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_120_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_120.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_120.Witness) : Prop :=
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
  sequential_generatedRoot_120.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_120 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_120.Witness,
    sequential_generatedRoot_120.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_122_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_122_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_122_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_122.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_122.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_122_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_122.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_122.Witness) : Prop :=
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
  sequential_generatedRoot_122.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_122 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_122.Witness,
    sequential_generatedRoot_122.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_124_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_124_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_124_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_124.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_124.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_124_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_124.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_124.Witness) : Prop :=
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
  sequential_generatedRoot_124.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_124 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_124.Witness,
    sequential_generatedRoot_124.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_126_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_126_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_126_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_126.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_126.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_126_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_126.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_126.Witness) : Prop :=
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
  sequential_generatedRoot_126.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_126 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_126.Witness,
    sequential_generatedRoot_126.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_128_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_128_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_128_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_128.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_128.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_128_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_128.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_128.Witness) : Prop :=
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
  sequential_generatedRoot_128.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_128 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_128.Witness,
    sequential_generatedRoot_128.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_130_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_130_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_130_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_130.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_130.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_130_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_130.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_130.Witness) : Prop :=
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
  sequential_generatedRoot_130.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_130 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_130.Witness,
    sequential_generatedRoot_130.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_132_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_132_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_132_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_132.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_132.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_132_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_132.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_132.Witness) : Prop :=
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
  sequential_generatedRoot_132.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_132 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_132.Witness,
    sequential_generatedRoot_132.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_134_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_134_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_134_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_134.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_134.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_134_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_134.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_134.Witness) : Prop :=
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
  sequential_generatedRoot_134.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_134 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_134.Witness,
    sequential_generatedRoot_134.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_136_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_136_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_136_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_136.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_136.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_136_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_136.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_136.Witness) : Prop :=
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
  sequential_generatedRoot_136.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_136 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_136.Witness,
    sequential_generatedRoot_136.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_138_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_138_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_138_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_138.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_138.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_138_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_138.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_138.Witness) : Prop :=
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
  sequential_generatedRoot_138.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_138 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_138.Witness,
    sequential_generatedRoot_138.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_140_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_140_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_140_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_140.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_140.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_140_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_140.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_140.Witness) : Prop :=
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
  sequential_generatedRoot_140.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_140 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_140.Witness,
    sequential_generatedRoot_140.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_142_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_142_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_142_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_142.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_142.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_142_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_142.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_142.Witness) : Prop :=
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
  sequential_generatedRoot_142.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_142 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_142.Witness,
    sequential_generatedRoot_142.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_144_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_144_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_144_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_144.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_144.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_144_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_144.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_144.Witness) : Prop :=
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
  sequential_generatedRoot_144.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_144 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_144.Witness,
    sequential_generatedRoot_144.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_146_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_146_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_146_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_146.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_146.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_146_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_146.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_146.Witness) : Prop :=
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
  sequential_generatedRoot_146.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_146 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_146.Witness,
    sequential_generatedRoot_146.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_148_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_148_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_148_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_148.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_148.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_148_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_148.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_148.Witness) : Prop :=
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
  sequential_generatedRoot_148.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_148 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_148.Witness,
    sequential_generatedRoot_148.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_150_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_150_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_150_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_150.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_150.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_150_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_150.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_150.Witness) : Prop :=
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
  sequential_generatedRoot_150.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_150 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_150.Witness,
    sequential_generatedRoot_150.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_152_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_152_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_152_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_152.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_152.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_152_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_152.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_152.Witness) : Prop :=
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
  sequential_generatedRoot_152.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_152 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_152.Witness,
    sequential_generatedRoot_152.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_154_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_154_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_154_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_154.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_154.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_154_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_154.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_154.Witness) : Prop :=
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
  sequential_generatedRoot_154.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_154 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_154.Witness,
    sequential_generatedRoot_154.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_156_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_156_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_156_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_156.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_156.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_156_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_156.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_156.Witness) : Prop :=
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
  sequential_generatedRoot_156.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_156 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_156.Witness,
    sequential_generatedRoot_156.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_158_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_158_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_158_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_158.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_158.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_158_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_158.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_158.Witness) : Prop :=
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
  sequential_generatedRoot_158.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_158 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_158.Witness,
    sequential_generatedRoot_158.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_160_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_160_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_160_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_160.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_160.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_160_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_160.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_160.Witness) : Prop :=
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
  sequential_generatedRoot_160.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_160 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_160.Witness,
    sequential_generatedRoot_160.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_162_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_162_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_162_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_162.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_162.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_162_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_162.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_162.Witness) : Prop :=
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
  sequential_generatedRoot_162.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_162 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_162.Witness,
    sequential_generatedRoot_162.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_164_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_164_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_164_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_164.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_164.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_164_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_164.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_164.Witness) : Prop :=
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
  sequential_generatedRoot_164.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_164 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_164.Witness,
    sequential_generatedRoot_164.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_166_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_166_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_166_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_166.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_166.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_166_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_166.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_166.Witness) : Prop :=
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
  sequential_generatedRoot_166.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_166 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_166.Witness,
    sequential_generatedRoot_166.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_168_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_168_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_168_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_168.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_168.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_168_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_168.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_168.Witness) : Prop :=
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
  sequential_generatedRoot_168.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_168 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_168.Witness,
    sequential_generatedRoot_168.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_170_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_170_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_170_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_170.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_170.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_170_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_170.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_170.Witness) : Prop :=
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
  sequential_generatedRoot_170.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_170 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_170.Witness,
    sequential_generatedRoot_170.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_172_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_172_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_172_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_172.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_172.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_172_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_172.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_172.Witness) : Prop :=
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
  sequential_generatedRoot_172.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_172 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_172.Witness,
    sequential_generatedRoot_172.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_174_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_174_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_174_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_174.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_174.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_174_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_174.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_174.Witness) : Prop :=
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
  sequential_generatedRoot_174.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_174 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_174.Witness,
    sequential_generatedRoot_174.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_176_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_176_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_176_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_176.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_176.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_176_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_176.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_176.Witness) : Prop :=
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
  sequential_generatedRoot_176.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_176 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_176.Witness,
    sequential_generatedRoot_176.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_178_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_178_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_178_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_178.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_178.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_178_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_178.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_178.Witness) : Prop :=
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
  sequential_generatedRoot_178.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_178 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_178.Witness,
    sequential_generatedRoot_178.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_180_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_180_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_180_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_180.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_180.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_180_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_180.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_180.Witness) : Prop :=
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
  sequential_generatedRoot_180.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_180 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_180.Witness,
    sequential_generatedRoot_180.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_182_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_182_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_182_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_182.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_182.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_182_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_182.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_182.Witness) : Prop :=
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
  sequential_generatedRoot_182.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_182 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_182.Witness,
    sequential_generatedRoot_182.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_184_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_184_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_184_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_184.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_184.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_184_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_184.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_184.Witness) : Prop :=
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
  sequential_generatedRoot_184.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_184 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_184.Witness,
    sequential_generatedRoot_184.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_186_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_186_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_186_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_186.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_186.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_186_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_186.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_186.Witness) : Prop :=
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
  sequential_generatedRoot_186.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_186 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_186.Witness,
    sequential_generatedRoot_186.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_188_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_188_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_188_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_188.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_188.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_188_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_188.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_188.Witness) : Prop :=
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
  sequential_generatedRoot_188.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_188 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_188.Witness,
    sequential_generatedRoot_188.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_190_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_190_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_190_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_190.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_190.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_190_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_190.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_190.Witness) : Prop :=
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
  sequential_generatedRoot_190.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_190 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_190.Witness,
    sequential_generatedRoot_190.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_192_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_192_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_192_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_192.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_192.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_192_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_192.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_192.Witness) : Prop :=
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
  sequential_generatedRoot_192.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_192 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_192.Witness,
    sequential_generatedRoot_192.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_194_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_194_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_194_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_194.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_194.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_194_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_194.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_194.Witness) : Prop :=
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
  sequential_generatedRoot_194.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_194 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_194.Witness,
    sequential_generatedRoot_194.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_196_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_196_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_196_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_196.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_196.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_196_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_196.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_196.Witness) : Prop :=
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
  sequential_generatedRoot_196.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_196 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_196.Witness,
    sequential_generatedRoot_196.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_198_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_198_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_198_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_198.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_198.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_198_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_198.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_198.Witness) : Prop :=
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
  sequential_generatedRoot_198.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_198 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_198.Witness,
    sequential_generatedRoot_198.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_200_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_200_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_200_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_200.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_200.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_200_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_200.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_200.Witness) : Prop :=
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
  sequential_generatedRoot_200.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_200 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_200.Witness,
    sequential_generatedRoot_200.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_202_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_202_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_202_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_202.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_202.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_202_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_202.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_202.Witness) : Prop :=
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
  sequential_generatedRoot_202.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_202 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_202.Witness,
    sequential_generatedRoot_202.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_204_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_204_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_204_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_204.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_204.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_204_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_204.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_204.Witness) : Prop :=
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
  sequential_generatedRoot_204.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_204 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_204.Witness,
    sequential_generatedRoot_204.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_206_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_206_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_206_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_206.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_206.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_206_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_206.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_206.Witness) : Prop :=
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
  sequential_generatedRoot_206.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_206 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_206.Witness,
    sequential_generatedRoot_206.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_208_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_208_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_208_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_208.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_208.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_208_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_208.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_208.Witness) : Prop :=
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
  sequential_generatedRoot_208.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_208 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_208.Witness,
    sequential_generatedRoot_208.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_210_5.constraints_0 (w_1_0 : Fin 800 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 800 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_210_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 800 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_210_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_210.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_210.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 800 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_210_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (5)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_210.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_210.Witness) : Prop :=
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
  sequential_generatedRoot_210.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_210 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 800 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_210.Witness,
    sequential_generatedRoot_210.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_222_10.constraints_0 (w_1_0 : Fin 40 → Int) (w_8_0 : Fin 5 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_11_0 : Int) (w_14_0 : Int) (w_17_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_26_0 : Int) (w_29_0 : Int) (w_32_0 : Int) (w_35_0 : Int) (w_38_0 : Int) (w_41_0 : Int) (w_44_0 : Int) (w_47_0 : Int) (w_50_0 : Int) (w_53_0 : Int) (w_56_0 : Int) (w_59_0 : Int) (w_62_0 : Int) (w_65_0 : Int) (w_68_0 : Int) (w_71_0 : Int) (w_74_0 : Int) (w_77_0 : Int) (w_80_0 : Int) (w_83_0 : Int) (w_86_0 : Int) (w_89_0 : Int) (w_92_0 : Int) (w_95_0 : Int) (w_98_0 : Int) (w_101_0 : Int) (w_104_0 : Int) (w_107_0 : Int) (w_110_0 : Int) (w_113_0 : Int) (w_116_0 : Int) (w_119_0 : Int) (w_122_0 : Int) (w_125_0 : Int) (w_128_0 : Int) (w_129_0 : Int) (w_130_0 : Int) (w_132_0 : Int) (outputs : Int) : Prop :=
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

def sequential_sequential_generatedRoot_222_10 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 40 → Int) × Int × (Fin 5 → Int) × Unit) (outputs : Int) : Prop :=
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
        sequential_sequential_generatedRoot_222_10.constraints_0 w_1_0 w_8_0 w_6_0 w_7_0 w_11_0 w_14_0 w_17_0 w_20_0 w_23_0 w_26_0 w_29_0 w_32_0 w_35_0 w_38_0 w_41_0 w_44_0 w_47_0 w_50_0 w_53_0 w_56_0 w_59_0 w_62_0 w_65_0 w_68_0 w_71_0 w_74_0 w_77_0 w_80_0 w_83_0 w_86_0 w_89_0 w_92_0 w_95_0 w_98_0 w_101_0 w_104_0 w_107_0 w_110_0 w_113_0 w_116_0 w_119_0 w_122_0 w_125_0 w_128_0 w_129_0 w_130_0 w_132_0 outputs

abbrev sequential_sequential_generatedRoot_222_14.constraints_0 (w_1_0 : Fin 40 → Int) (w_8_0 : Fin 5 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_11_0 : Int) (w_14_0 : Int) (w_17_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_26_0 : Int) (w_29_0 : Int) (w_32_0 : Int) (w_35_0 : Int) (w_38_0 : Int) (w_41_0 : Int) (w_44_0 : Int) (w_47_0 : Int) (w_50_0 : Int) (w_53_0 : Int) (w_56_0 : Int) (w_59_0 : Int) (w_62_0 : Int) (w_65_0 : Int) (w_68_0 : Int) (w_71_0 : Int) (w_74_0 : Int) (w_77_0 : Int) (w_80_0 : Int) (w_83_0 : Int) (w_86_0 : Int) (w_89_0 : Int) (w_92_0 : Int) (w_95_0 : Int) (w_98_0 : Int) (w_101_0 : Int) (w_104_0 : Int) (w_107_0 : Int) (w_110_0 : Int) (w_113_0 : Int) (w_116_0 : Int) (w_119_0 : Int) (w_122_0 : Int) (w_125_0 : Int) (w_128_0 : Int) (w_129_0 : Int) (w_130_0 : Int) (w_132_0 : Int) (outputs : Int) : Prop :=
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

def sequential_sequential_generatedRoot_222_14 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 40 → Int) × Int × (Fin 5 → Int) × Unit) (outputs : Int) : Prop :=
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
        sequential_sequential_generatedRoot_222_14.constraints_0 w_1_0 w_8_0 w_6_0 w_7_0 w_11_0 w_14_0 w_17_0 w_20_0 w_23_0 w_26_0 w_29_0 w_32_0 w_35_0 w_38_0 w_41_0 w_44_0 w_47_0 w_50_0 w_53_0 w_56_0 w_59_0 w_62_0 w_65_0 w_68_0 w_71_0 w_74_0 w_77_0 w_80_0 w_83_0 w_86_0 w_89_0 w_92_0 w_95_0 w_98_0 w_101_0 w_104_0 w_107_0 w_110_0 w_113_0 w_116_0 w_119_0 w_122_0 w_125_0 w_128_0 w_129_0 w_130_0 w_132_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_222.Witness where
  w_4_0 : Int
  w_10_0 : Int
  w_14_0 : Int

abbrev sequential_generatedRoot_222.constraints_0 (params : Params) (i_0 : Nat) (w_2_0 : Fin 8 → Int) (w_8_0 : Fin 40 → Int) (w_9_0 : Fin 5 → Int) (w_13_0 : Fin 40 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_12_0 : Int) (w_14_0 : Int) (w_16_0 : Int) (w_18_0 : Int) (w_25_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_222_10 params i_0 i (current, (w_8_0, (w_3_0, (w_9_0, ())))) next) (Int.toNat (5)) w_7_0 w_10_0 ∧
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_222_14 params i_0 i (current, (w_13_0, (w_3_0, (w_9_0, ())))) next) (Int.toNat (5)) w_12_0 w_14_0 ∧
  w_16_0 ≠ 0 ∧
  w_18_0 ≠ 0 ∧
  outputs = w_25_0

abbrev sequential_generatedRoot_222.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8 → Int) × (Fin 40 → Int) × (Fin 5 → Int) × (Fin 40 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_222.Witness) : Prop :=
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
  sequential_generatedRoot_222.constraints_0 params i_0 w_2_0 w_8_0 w_9_0 w_13_0 w_3_0 w_4_0 w_7_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_25_0 outputs

def sequential_generatedRoot_222 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8 → Int) × (Fin 40 → Int) × (Fin 5 → Int) × (Fin 40 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_222.Witness,
    sequential_generatedRoot_222.body params i_0 inputs outputs witness

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
  w_68_0 : Bool
  w_70_0 : Bool
  w_72_0 : Bool
  w_74_0 : Bool
  w_76_0 : Bool
  w_78_0 : Bool
  w_80_0 : Bool
  w_82_0 : Bool
  w_84_0 : Bool
  w_86_0 : Bool
  w_88_0 : Bool
  w_90_0 : Bool
  w_92_0 : Bool
  w_94_0 : Bool
  w_96_0 : Bool
  w_98_0 : Bool
  w_100_0 : Bool
  w_102_0 : Bool
  w_104_0 : Bool
  w_106_0 : Bool
  w_108_0 : Bool
  w_110_0 : Bool
  w_112_0 : Bool
  w_114_0 : Bool
  w_116_0 : Bool
  w_118_0 : Bool
  w_120_0 : Bool
  w_122_0 : Bool
  w_124_0 : Bool
  w_126_0 : Bool
  w_128_0 : Bool
  w_130_0 : Bool
  w_132_0 : Bool
  w_134_0 : Bool
  w_136_0 : Bool
  w_138_0 : Bool
  w_140_0 : Bool
  w_142_0 : Bool
  w_144_0 : Bool
  w_146_0 : Bool
  w_148_0 : Bool
  w_150_0 : Bool
  w_152_0 : Bool
  w_154_0 : Bool
  w_156_0 : Bool
  w_158_0 : Bool
  w_160_0 : Bool
  w_162_0 : Bool
  w_164_0 : Bool
  w_166_0 : Bool
  w_168_0 : Bool
  w_170_0 : Bool
  w_172_0 : Bool
  w_174_0 : Bool
  w_176_0 : Bool
  w_178_0 : Bool
  w_180_0 : Bool
  w_182_0 : Bool
  w_184_0 : Bool
  w_186_0 : Bool
  w_188_0 : Bool
  w_190_0 : Bool
  w_192_0 : Bool
  w_194_0 : Bool
  w_196_0 : Bool
  w_198_0 : Bool
  w_200_0 : Bool
  w_202_0 : Bool
  w_204_0 : Bool
  w_206_0 : Bool
  w_208_0 : Bool
  w_210_0 : Bool
  w_222_0 : Bool

abbrev generatedRoot.constraints_6 (params : Params) (w_3_0 : Fin 5 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_197_0 : Fin 800 → Int) (w_199_0 : Fin 800 → Int) (w_201_0 : Fin 800 → Int) (w_203_0 : Fin 800 → Int) (w_205_0 : Fin 800 → Int) (w_207_0 : Fin 800 → Int) (w_209_0 : Fin 800 → Int) (w_221_0 : Fin 8 → Int) (w_196_0 : Bool) (w_198_0 : Bool) (w_200_0 : Bool) (w_202_0 : Bool) (w_204_0 : Bool) (w_206_0 : Bool) (w_208_0 : Bool) (w_210_0 : Bool) (w_220_0 : Bool) (w_222_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_198 params i (current, (w_197_0, ())) next) (Int.toNat (160)) w_196_0 w_198_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_200 params i (current, (w_199_0, ())) next) (Int.toNat (160)) w_198_0 w_200_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_202 params i (current, (w_201_0, ())) next) (Int.toNat (160)) w_200_0 w_202_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_204 params i (current, (w_203_0, ())) next) (Int.toNat (160)) w_202_0 w_204_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_206 params i (current, (w_205_0, ())) next) (Int.toNat (160)) w_204_0 w_206_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_208 params i (current, (w_207_0, ())) next) (Int.toNat (160)) w_206_0 w_208_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_210 params i (current, (w_209_0, ())) next) (Int.toNat (160)) w_208_0 w_210_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_222 params i (current, (w_221_0, (w_15_0, (w_3_0, (w_17_0, ()))))) next) (Int.toNat (8)) w_220_0 w_222_0 ∧
  outputs = w_222_0

abbrev generatedRoot.constraints_5 (params : Params) (w_3_0 : Fin 5 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_165_0 : Fin 800 → Int) (w_167_0 : Fin 800 → Int) (w_169_0 : Fin 800 → Int) (w_171_0 : Fin 800 → Int) (w_173_0 : Fin 800 → Int) (w_175_0 : Fin 800 → Int) (w_177_0 : Fin 800 → Int) (w_179_0 : Fin 800 → Int) (w_181_0 : Fin 800 → Int) (w_183_0 : Fin 800 → Int) (w_185_0 : Fin 800 → Int) (w_187_0 : Fin 800 → Int) (w_189_0 : Fin 800 → Int) (w_191_0 : Fin 800 → Int) (w_193_0 : Fin 800 → Int) (w_195_0 : Fin 800 → Int) (w_197_0 : Fin 800 → Int) (w_199_0 : Fin 800 → Int) (w_201_0 : Fin 800 → Int) (w_203_0 : Fin 800 → Int) (w_205_0 : Fin 800 → Int) (w_207_0 : Fin 800 → Int) (w_209_0 : Fin 800 → Int) (w_221_0 : Fin 8 → Int) (w_164_0 : Bool) (w_166_0 : Bool) (w_168_0 : Bool) (w_170_0 : Bool) (w_172_0 : Bool) (w_174_0 : Bool) (w_176_0 : Bool) (w_178_0 : Bool) (w_180_0 : Bool) (w_182_0 : Bool) (w_184_0 : Bool) (w_186_0 : Bool) (w_188_0 : Bool) (w_190_0 : Bool) (w_192_0 : Bool) (w_194_0 : Bool) (w_196_0 : Bool) (w_198_0 : Bool) (w_200_0 : Bool) (w_202_0 : Bool) (w_204_0 : Bool) (w_206_0 : Bool) (w_208_0 : Bool) (w_210_0 : Bool) (w_220_0 : Bool) (w_222_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_166 params i (current, (w_165_0, ())) next) (Int.toNat (160)) w_164_0 w_166_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_168 params i (current, (w_167_0, ())) next) (Int.toNat (160)) w_166_0 w_168_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_170 params i (current, (w_169_0, ())) next) (Int.toNat (160)) w_168_0 w_170_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_172 params i (current, (w_171_0, ())) next) (Int.toNat (160)) w_170_0 w_172_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_174 params i (current, (w_173_0, ())) next) (Int.toNat (160)) w_172_0 w_174_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_176 params i (current, (w_175_0, ())) next) (Int.toNat (160)) w_174_0 w_176_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_178 params i (current, (w_177_0, ())) next) (Int.toNat (160)) w_176_0 w_178_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_180 params i (current, (w_179_0, ())) next) (Int.toNat (160)) w_178_0 w_180_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_182 params i (current, (w_181_0, ())) next) (Int.toNat (160)) w_180_0 w_182_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_184 params i (current, (w_183_0, ())) next) (Int.toNat (160)) w_182_0 w_184_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_186 params i (current, (w_185_0, ())) next) (Int.toNat (160)) w_184_0 w_186_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_188 params i (current, (w_187_0, ())) next) (Int.toNat (160)) w_186_0 w_188_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_190 params i (current, (w_189_0, ())) next) (Int.toNat (160)) w_188_0 w_190_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_192 params i (current, (w_191_0, ())) next) (Int.toNat (160)) w_190_0 w_192_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_194 params i (current, (w_193_0, ())) next) (Int.toNat (160)) w_192_0 w_194_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_196 params i (current, (w_195_0, ())) next) (Int.toNat (160)) w_194_0 w_196_0 ∧
  generatedRoot.constraints_6 params w_3_0 w_15_0 w_17_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_207_0 w_209_0 w_221_0 w_196_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_208_0 w_210_0 w_220_0 w_222_0 outputs

abbrev generatedRoot.constraints_4 (params : Params) (w_3_0 : Fin 5 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_133_0 : Fin 800 → Int) (w_135_0 : Fin 800 → Int) (w_137_0 : Fin 800 → Int) (w_139_0 : Fin 800 → Int) (w_141_0 : Fin 800 → Int) (w_143_0 : Fin 800 → Int) (w_145_0 : Fin 800 → Int) (w_147_0 : Fin 800 → Int) (w_149_0 : Fin 800 → Int) (w_151_0 : Fin 800 → Int) (w_153_0 : Fin 800 → Int) (w_155_0 : Fin 800 → Int) (w_157_0 : Fin 800 → Int) (w_159_0 : Fin 800 → Int) (w_161_0 : Fin 800 → Int) (w_163_0 : Fin 800 → Int) (w_165_0 : Fin 800 → Int) (w_167_0 : Fin 800 → Int) (w_169_0 : Fin 800 → Int) (w_171_0 : Fin 800 → Int) (w_173_0 : Fin 800 → Int) (w_175_0 : Fin 800 → Int) (w_177_0 : Fin 800 → Int) (w_179_0 : Fin 800 → Int) (w_181_0 : Fin 800 → Int) (w_183_0 : Fin 800 → Int) (w_185_0 : Fin 800 → Int) (w_187_0 : Fin 800 → Int) (w_189_0 : Fin 800 → Int) (w_191_0 : Fin 800 → Int) (w_193_0 : Fin 800 → Int) (w_195_0 : Fin 800 → Int) (w_197_0 : Fin 800 → Int) (w_199_0 : Fin 800 → Int) (w_201_0 : Fin 800 → Int) (w_203_0 : Fin 800 → Int) (w_205_0 : Fin 800 → Int) (w_207_0 : Fin 800 → Int) (w_209_0 : Fin 800 → Int) (w_221_0 : Fin 8 → Int) (w_132_0 : Bool) (w_134_0 : Bool) (w_136_0 : Bool) (w_138_0 : Bool) (w_140_0 : Bool) (w_142_0 : Bool) (w_144_0 : Bool) (w_146_0 : Bool) (w_148_0 : Bool) (w_150_0 : Bool) (w_152_0 : Bool) (w_154_0 : Bool) (w_156_0 : Bool) (w_158_0 : Bool) (w_160_0 : Bool) (w_162_0 : Bool) (w_164_0 : Bool) (w_166_0 : Bool) (w_168_0 : Bool) (w_170_0 : Bool) (w_172_0 : Bool) (w_174_0 : Bool) (w_176_0 : Bool) (w_178_0 : Bool) (w_180_0 : Bool) (w_182_0 : Bool) (w_184_0 : Bool) (w_186_0 : Bool) (w_188_0 : Bool) (w_190_0 : Bool) (w_192_0 : Bool) (w_194_0 : Bool) (w_196_0 : Bool) (w_198_0 : Bool) (w_200_0 : Bool) (w_202_0 : Bool) (w_204_0 : Bool) (w_206_0 : Bool) (w_208_0 : Bool) (w_210_0 : Bool) (w_220_0 : Bool) (w_222_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_134 params i (current, (w_133_0, ())) next) (Int.toNat (160)) w_132_0 w_134_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_136 params i (current, (w_135_0, ())) next) (Int.toNat (160)) w_134_0 w_136_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_138 params i (current, (w_137_0, ())) next) (Int.toNat (160)) w_136_0 w_138_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_140 params i (current, (w_139_0, ())) next) (Int.toNat (160)) w_138_0 w_140_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_142 params i (current, (w_141_0, ())) next) (Int.toNat (160)) w_140_0 w_142_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_144 params i (current, (w_143_0, ())) next) (Int.toNat (160)) w_142_0 w_144_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_146 params i (current, (w_145_0, ())) next) (Int.toNat (160)) w_144_0 w_146_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_148 params i (current, (w_147_0, ())) next) (Int.toNat (160)) w_146_0 w_148_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_150 params i (current, (w_149_0, ())) next) (Int.toNat (160)) w_148_0 w_150_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_152 params i (current, (w_151_0, ())) next) (Int.toNat (160)) w_150_0 w_152_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_154 params i (current, (w_153_0, ())) next) (Int.toNat (160)) w_152_0 w_154_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_156 params i (current, (w_155_0, ())) next) (Int.toNat (160)) w_154_0 w_156_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_158 params i (current, (w_157_0, ())) next) (Int.toNat (160)) w_156_0 w_158_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_160 params i (current, (w_159_0, ())) next) (Int.toNat (160)) w_158_0 w_160_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_162 params i (current, (w_161_0, ())) next) (Int.toNat (160)) w_160_0 w_162_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_164 params i (current, (w_163_0, ())) next) (Int.toNat (160)) w_162_0 w_164_0 ∧
  generatedRoot.constraints_5 params w_3_0 w_15_0 w_17_0 w_165_0 w_167_0 w_169_0 w_171_0 w_173_0 w_175_0 w_177_0 w_179_0 w_181_0 w_183_0 w_185_0 w_187_0 w_189_0 w_191_0 w_193_0 w_195_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_207_0 w_209_0 w_221_0 w_164_0 w_166_0 w_168_0 w_170_0 w_172_0 w_174_0 w_176_0 w_178_0 w_180_0 w_182_0 w_184_0 w_186_0 w_188_0 w_190_0 w_192_0 w_194_0 w_196_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_208_0 w_210_0 w_220_0 w_222_0 outputs

abbrev generatedRoot.constraints_3 (params : Params) (w_3_0 : Fin 5 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_101_0 : Fin 800 → Int) (w_103_0 : Fin 800 → Int) (w_105_0 : Fin 800 → Int) (w_107_0 : Fin 800 → Int) (w_109_0 : Fin 800 → Int) (w_111_0 : Fin 800 → Int) (w_113_0 : Fin 800 → Int) (w_115_0 : Fin 800 → Int) (w_117_0 : Fin 800 → Int) (w_119_0 : Fin 800 → Int) (w_121_0 : Fin 800 → Int) (w_123_0 : Fin 800 → Int) (w_125_0 : Fin 800 → Int) (w_127_0 : Fin 800 → Int) (w_129_0 : Fin 800 → Int) (w_131_0 : Fin 800 → Int) (w_133_0 : Fin 800 → Int) (w_135_0 : Fin 800 → Int) (w_137_0 : Fin 800 → Int) (w_139_0 : Fin 800 → Int) (w_141_0 : Fin 800 → Int) (w_143_0 : Fin 800 → Int) (w_145_0 : Fin 800 → Int) (w_147_0 : Fin 800 → Int) (w_149_0 : Fin 800 → Int) (w_151_0 : Fin 800 → Int) (w_153_0 : Fin 800 → Int) (w_155_0 : Fin 800 → Int) (w_157_0 : Fin 800 → Int) (w_159_0 : Fin 800 → Int) (w_161_0 : Fin 800 → Int) (w_163_0 : Fin 800 → Int) (w_165_0 : Fin 800 → Int) (w_167_0 : Fin 800 → Int) (w_169_0 : Fin 800 → Int) (w_171_0 : Fin 800 → Int) (w_173_0 : Fin 800 → Int) (w_175_0 : Fin 800 → Int) (w_177_0 : Fin 800 → Int) (w_179_0 : Fin 800 → Int) (w_181_0 : Fin 800 → Int) (w_183_0 : Fin 800 → Int) (w_185_0 : Fin 800 → Int) (w_187_0 : Fin 800 → Int) (w_189_0 : Fin 800 → Int) (w_191_0 : Fin 800 → Int) (w_193_0 : Fin 800 → Int) (w_195_0 : Fin 800 → Int) (w_197_0 : Fin 800 → Int) (w_199_0 : Fin 800 → Int) (w_201_0 : Fin 800 → Int) (w_203_0 : Fin 800 → Int) (w_205_0 : Fin 800 → Int) (w_207_0 : Fin 800 → Int) (w_209_0 : Fin 800 → Int) (w_221_0 : Fin 8 → Int) (w_100_0 : Bool) (w_102_0 : Bool) (w_104_0 : Bool) (w_106_0 : Bool) (w_108_0 : Bool) (w_110_0 : Bool) (w_112_0 : Bool) (w_114_0 : Bool) (w_116_0 : Bool) (w_118_0 : Bool) (w_120_0 : Bool) (w_122_0 : Bool) (w_124_0 : Bool) (w_126_0 : Bool) (w_128_0 : Bool) (w_130_0 : Bool) (w_132_0 : Bool) (w_134_0 : Bool) (w_136_0 : Bool) (w_138_0 : Bool) (w_140_0 : Bool) (w_142_0 : Bool) (w_144_0 : Bool) (w_146_0 : Bool) (w_148_0 : Bool) (w_150_0 : Bool) (w_152_0 : Bool) (w_154_0 : Bool) (w_156_0 : Bool) (w_158_0 : Bool) (w_160_0 : Bool) (w_162_0 : Bool) (w_164_0 : Bool) (w_166_0 : Bool) (w_168_0 : Bool) (w_170_0 : Bool) (w_172_0 : Bool) (w_174_0 : Bool) (w_176_0 : Bool) (w_178_0 : Bool) (w_180_0 : Bool) (w_182_0 : Bool) (w_184_0 : Bool) (w_186_0 : Bool) (w_188_0 : Bool) (w_190_0 : Bool) (w_192_0 : Bool) (w_194_0 : Bool) (w_196_0 : Bool) (w_198_0 : Bool) (w_200_0 : Bool) (w_202_0 : Bool) (w_204_0 : Bool) (w_206_0 : Bool) (w_208_0 : Bool) (w_210_0 : Bool) (w_220_0 : Bool) (w_222_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_102 params i (current, (w_101_0, ())) next) (Int.toNat (160)) w_100_0 w_102_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_104 params i (current, (w_103_0, ())) next) (Int.toNat (160)) w_102_0 w_104_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_106 params i (current, (w_105_0, ())) next) (Int.toNat (160)) w_104_0 w_106_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_108 params i (current, (w_107_0, ())) next) (Int.toNat (160)) w_106_0 w_108_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_110 params i (current, (w_109_0, ())) next) (Int.toNat (160)) w_108_0 w_110_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_112 params i (current, (w_111_0, ())) next) (Int.toNat (160)) w_110_0 w_112_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_114 params i (current, (w_113_0, ())) next) (Int.toNat (160)) w_112_0 w_114_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_116 params i (current, (w_115_0, ())) next) (Int.toNat (160)) w_114_0 w_116_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_118 params i (current, (w_117_0, ())) next) (Int.toNat (160)) w_116_0 w_118_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_120 params i (current, (w_119_0, ())) next) (Int.toNat (160)) w_118_0 w_120_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_122 params i (current, (w_121_0, ())) next) (Int.toNat (160)) w_120_0 w_122_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_124 params i (current, (w_123_0, ())) next) (Int.toNat (160)) w_122_0 w_124_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_126 params i (current, (w_125_0, ())) next) (Int.toNat (160)) w_124_0 w_126_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_128 params i (current, (w_127_0, ())) next) (Int.toNat (160)) w_126_0 w_128_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_130 params i (current, (w_129_0, ())) next) (Int.toNat (160)) w_128_0 w_130_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_132 params i (current, (w_131_0, ())) next) (Int.toNat (160)) w_130_0 w_132_0 ∧
  generatedRoot.constraints_4 params w_3_0 w_15_0 w_17_0 w_133_0 w_135_0 w_137_0 w_139_0 w_141_0 w_143_0 w_145_0 w_147_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_159_0 w_161_0 w_163_0 w_165_0 w_167_0 w_169_0 w_171_0 w_173_0 w_175_0 w_177_0 w_179_0 w_181_0 w_183_0 w_185_0 w_187_0 w_189_0 w_191_0 w_193_0 w_195_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_207_0 w_209_0 w_221_0 w_132_0 w_134_0 w_136_0 w_138_0 w_140_0 w_142_0 w_144_0 w_146_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_158_0 w_160_0 w_162_0 w_164_0 w_166_0 w_168_0 w_170_0 w_172_0 w_174_0 w_176_0 w_178_0 w_180_0 w_182_0 w_184_0 w_186_0 w_188_0 w_190_0 w_192_0 w_194_0 w_196_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_208_0 w_210_0 w_220_0 w_222_0 outputs

abbrev generatedRoot.constraints_2 (params : Params) (w_3_0 : Fin 5 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_69_0 : Fin 800 → Int) (w_71_0 : Fin 800 → Int) (w_73_0 : Fin 800 → Int) (w_75_0 : Fin 800 → Int) (w_77_0 : Fin 800 → Int) (w_79_0 : Fin 800 → Int) (w_81_0 : Fin 800 → Int) (w_83_0 : Fin 800 → Int) (w_85_0 : Fin 800 → Int) (w_87_0 : Fin 800 → Int) (w_89_0 : Fin 800 → Int) (w_91_0 : Fin 800 → Int) (w_93_0 : Fin 800 → Int) (w_95_0 : Fin 800 → Int) (w_97_0 : Fin 800 → Int) (w_99_0 : Fin 800 → Int) (w_101_0 : Fin 800 → Int) (w_103_0 : Fin 800 → Int) (w_105_0 : Fin 800 → Int) (w_107_0 : Fin 800 → Int) (w_109_0 : Fin 800 → Int) (w_111_0 : Fin 800 → Int) (w_113_0 : Fin 800 → Int) (w_115_0 : Fin 800 → Int) (w_117_0 : Fin 800 → Int) (w_119_0 : Fin 800 → Int) (w_121_0 : Fin 800 → Int) (w_123_0 : Fin 800 → Int) (w_125_0 : Fin 800 → Int) (w_127_0 : Fin 800 → Int) (w_129_0 : Fin 800 → Int) (w_131_0 : Fin 800 → Int) (w_133_0 : Fin 800 → Int) (w_135_0 : Fin 800 → Int) (w_137_0 : Fin 800 → Int) (w_139_0 : Fin 800 → Int) (w_141_0 : Fin 800 → Int) (w_143_0 : Fin 800 → Int) (w_145_0 : Fin 800 → Int) (w_147_0 : Fin 800 → Int) (w_149_0 : Fin 800 → Int) (w_151_0 : Fin 800 → Int) (w_153_0 : Fin 800 → Int) (w_155_0 : Fin 800 → Int) (w_157_0 : Fin 800 → Int) (w_159_0 : Fin 800 → Int) (w_161_0 : Fin 800 → Int) (w_163_0 : Fin 800 → Int) (w_165_0 : Fin 800 → Int) (w_167_0 : Fin 800 → Int) (w_169_0 : Fin 800 → Int) (w_171_0 : Fin 800 → Int) (w_173_0 : Fin 800 → Int) (w_175_0 : Fin 800 → Int) (w_177_0 : Fin 800 → Int) (w_179_0 : Fin 800 → Int) (w_181_0 : Fin 800 → Int) (w_183_0 : Fin 800 → Int) (w_185_0 : Fin 800 → Int) (w_187_0 : Fin 800 → Int) (w_189_0 : Fin 800 → Int) (w_191_0 : Fin 800 → Int) (w_193_0 : Fin 800 → Int) (w_195_0 : Fin 800 → Int) (w_197_0 : Fin 800 → Int) (w_199_0 : Fin 800 → Int) (w_201_0 : Fin 800 → Int) (w_203_0 : Fin 800 → Int) (w_205_0 : Fin 800 → Int) (w_207_0 : Fin 800 → Int) (w_209_0 : Fin 800 → Int) (w_221_0 : Fin 8 → Int) (w_68_0 : Bool) (w_70_0 : Bool) (w_72_0 : Bool) (w_74_0 : Bool) (w_76_0 : Bool) (w_78_0 : Bool) (w_80_0 : Bool) (w_82_0 : Bool) (w_84_0 : Bool) (w_86_0 : Bool) (w_88_0 : Bool) (w_90_0 : Bool) (w_92_0 : Bool) (w_94_0 : Bool) (w_96_0 : Bool) (w_98_0 : Bool) (w_100_0 : Bool) (w_102_0 : Bool) (w_104_0 : Bool) (w_106_0 : Bool) (w_108_0 : Bool) (w_110_0 : Bool) (w_112_0 : Bool) (w_114_0 : Bool) (w_116_0 : Bool) (w_118_0 : Bool) (w_120_0 : Bool) (w_122_0 : Bool) (w_124_0 : Bool) (w_126_0 : Bool) (w_128_0 : Bool) (w_130_0 : Bool) (w_132_0 : Bool) (w_134_0 : Bool) (w_136_0 : Bool) (w_138_0 : Bool) (w_140_0 : Bool) (w_142_0 : Bool) (w_144_0 : Bool) (w_146_0 : Bool) (w_148_0 : Bool) (w_150_0 : Bool) (w_152_0 : Bool) (w_154_0 : Bool) (w_156_0 : Bool) (w_158_0 : Bool) (w_160_0 : Bool) (w_162_0 : Bool) (w_164_0 : Bool) (w_166_0 : Bool) (w_168_0 : Bool) (w_170_0 : Bool) (w_172_0 : Bool) (w_174_0 : Bool) (w_176_0 : Bool) (w_178_0 : Bool) (w_180_0 : Bool) (w_182_0 : Bool) (w_184_0 : Bool) (w_186_0 : Bool) (w_188_0 : Bool) (w_190_0 : Bool) (w_192_0 : Bool) (w_194_0 : Bool) (w_196_0 : Bool) (w_198_0 : Bool) (w_200_0 : Bool) (w_202_0 : Bool) (w_204_0 : Bool) (w_206_0 : Bool) (w_208_0 : Bool) (w_210_0 : Bool) (w_220_0 : Bool) (w_222_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_70 params i (current, (w_69_0, ())) next) (Int.toNat (160)) w_68_0 w_70_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_72 params i (current, (w_71_0, ())) next) (Int.toNat (160)) w_70_0 w_72_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_74 params i (current, (w_73_0, ())) next) (Int.toNat (160)) w_72_0 w_74_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_76 params i (current, (w_75_0, ())) next) (Int.toNat (160)) w_74_0 w_76_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_78 params i (current, (w_77_0, ())) next) (Int.toNat (160)) w_76_0 w_78_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_80 params i (current, (w_79_0, ())) next) (Int.toNat (160)) w_78_0 w_80_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_82 params i (current, (w_81_0, ())) next) (Int.toNat (160)) w_80_0 w_82_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_84 params i (current, (w_83_0, ())) next) (Int.toNat (160)) w_82_0 w_84_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_86 params i (current, (w_85_0, ())) next) (Int.toNat (160)) w_84_0 w_86_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_88 params i (current, (w_87_0, ())) next) (Int.toNat (160)) w_86_0 w_88_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_90 params i (current, (w_89_0, ())) next) (Int.toNat (160)) w_88_0 w_90_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_92 params i (current, (w_91_0, ())) next) (Int.toNat (160)) w_90_0 w_92_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_94 params i (current, (w_93_0, ())) next) (Int.toNat (160)) w_92_0 w_94_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_96 params i (current, (w_95_0, ())) next) (Int.toNat (160)) w_94_0 w_96_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_98 params i (current, (w_97_0, ())) next) (Int.toNat (160)) w_96_0 w_98_0 ∧
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_100 params i (current, (w_99_0, ())) next) (Int.toNat (160)) w_98_0 w_100_0 ∧
  generatedRoot.constraints_3 params w_3_0 w_15_0 w_17_0 w_101_0 w_103_0 w_105_0 w_107_0 w_109_0 w_111_0 w_113_0 w_115_0 w_117_0 w_119_0 w_121_0 w_123_0 w_125_0 w_127_0 w_129_0 w_131_0 w_133_0 w_135_0 w_137_0 w_139_0 w_141_0 w_143_0 w_145_0 w_147_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_159_0 w_161_0 w_163_0 w_165_0 w_167_0 w_169_0 w_171_0 w_173_0 w_175_0 w_177_0 w_179_0 w_181_0 w_183_0 w_185_0 w_187_0 w_189_0 w_191_0 w_193_0 w_195_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_207_0 w_209_0 w_221_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_110_0 w_112_0 w_114_0 w_116_0 w_118_0 w_120_0 w_122_0 w_124_0 w_126_0 w_128_0 w_130_0 w_132_0 w_134_0 w_136_0 w_138_0 w_140_0 w_142_0 w_144_0 w_146_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_158_0 w_160_0 w_162_0 w_164_0 w_166_0 w_168_0 w_170_0 w_172_0 w_174_0 w_176_0 w_178_0 w_180_0 w_182_0 w_184_0 w_186_0 w_188_0 w_190_0 w_192_0 w_194_0 w_196_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_208_0 w_210_0 w_220_0 w_222_0 outputs

abbrev generatedRoot.constraints_1 (params : Params) (w_3_0 : Fin 5 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_37_0 : Fin 800 → Int) (w_39_0 : Fin 800 → Int) (w_41_0 : Fin 800 → Int) (w_43_0 : Fin 800 → Int) (w_45_0 : Fin 800 → Int) (w_47_0 : Fin 800 → Int) (w_49_0 : Fin 800 → Int) (w_51_0 : Fin 800 → Int) (w_53_0 : Fin 800 → Int) (w_55_0 : Fin 800 → Int) (w_57_0 : Fin 800 → Int) (w_59_0 : Fin 800 → Int) (w_61_0 : Fin 800 → Int) (w_63_0 : Fin 800 → Int) (w_65_0 : Fin 800 → Int) (w_67_0 : Fin 800 → Int) (w_69_0 : Fin 800 → Int) (w_71_0 : Fin 800 → Int) (w_73_0 : Fin 800 → Int) (w_75_0 : Fin 800 → Int) (w_77_0 : Fin 800 → Int) (w_79_0 : Fin 800 → Int) (w_81_0 : Fin 800 → Int) (w_83_0 : Fin 800 → Int) (w_85_0 : Fin 800 → Int) (w_87_0 : Fin 800 → Int) (w_89_0 : Fin 800 → Int) (w_91_0 : Fin 800 → Int) (w_93_0 : Fin 800 → Int) (w_95_0 : Fin 800 → Int) (w_97_0 : Fin 800 → Int) (w_99_0 : Fin 800 → Int) (w_101_0 : Fin 800 → Int) (w_103_0 : Fin 800 → Int) (w_105_0 : Fin 800 → Int) (w_107_0 : Fin 800 → Int) (w_109_0 : Fin 800 → Int) (w_111_0 : Fin 800 → Int) (w_113_0 : Fin 800 → Int) (w_115_0 : Fin 800 → Int) (w_117_0 : Fin 800 → Int) (w_119_0 : Fin 800 → Int) (w_121_0 : Fin 800 → Int) (w_123_0 : Fin 800 → Int) (w_125_0 : Fin 800 → Int) (w_127_0 : Fin 800 → Int) (w_129_0 : Fin 800 → Int) (w_131_0 : Fin 800 → Int) (w_133_0 : Fin 800 → Int) (w_135_0 : Fin 800 → Int) (w_137_0 : Fin 800 → Int) (w_139_0 : Fin 800 → Int) (w_141_0 : Fin 800 → Int) (w_143_0 : Fin 800 → Int) (w_145_0 : Fin 800 → Int) (w_147_0 : Fin 800 → Int) (w_149_0 : Fin 800 → Int) (w_151_0 : Fin 800 → Int) (w_153_0 : Fin 800 → Int) (w_155_0 : Fin 800 → Int) (w_157_0 : Fin 800 → Int) (w_159_0 : Fin 800 → Int) (w_161_0 : Fin 800 → Int) (w_163_0 : Fin 800 → Int) (w_165_0 : Fin 800 → Int) (w_167_0 : Fin 800 → Int) (w_169_0 : Fin 800 → Int) (w_171_0 : Fin 800 → Int) (w_173_0 : Fin 800 → Int) (w_175_0 : Fin 800 → Int) (w_177_0 : Fin 800 → Int) (w_179_0 : Fin 800 → Int) (w_181_0 : Fin 800 → Int) (w_183_0 : Fin 800 → Int) (w_185_0 : Fin 800 → Int) (w_187_0 : Fin 800 → Int) (w_189_0 : Fin 800 → Int) (w_191_0 : Fin 800 → Int) (w_193_0 : Fin 800 → Int) (w_195_0 : Fin 800 → Int) (w_197_0 : Fin 800 → Int) (w_199_0 : Fin 800 → Int) (w_201_0 : Fin 800 → Int) (w_203_0 : Fin 800 → Int) (w_205_0 : Fin 800 → Int) (w_207_0 : Fin 800 → Int) (w_209_0 : Fin 800 → Int) (w_221_0 : Fin 8 → Int) (w_36_0 : Bool) (w_38_0 : Bool) (w_40_0 : Bool) (w_42_0 : Bool) (w_44_0 : Bool) (w_46_0 : Bool) (w_48_0 : Bool) (w_50_0 : Bool) (w_52_0 : Bool) (w_54_0 : Bool) (w_56_0 : Bool) (w_58_0 : Bool) (w_60_0 : Bool) (w_62_0 : Bool) (w_64_0 : Bool) (w_66_0 : Bool) (w_68_0 : Bool) (w_70_0 : Bool) (w_72_0 : Bool) (w_74_0 : Bool) (w_76_0 : Bool) (w_78_0 : Bool) (w_80_0 : Bool) (w_82_0 : Bool) (w_84_0 : Bool) (w_86_0 : Bool) (w_88_0 : Bool) (w_90_0 : Bool) (w_92_0 : Bool) (w_94_0 : Bool) (w_96_0 : Bool) (w_98_0 : Bool) (w_100_0 : Bool) (w_102_0 : Bool) (w_104_0 : Bool) (w_106_0 : Bool) (w_108_0 : Bool) (w_110_0 : Bool) (w_112_0 : Bool) (w_114_0 : Bool) (w_116_0 : Bool) (w_118_0 : Bool) (w_120_0 : Bool) (w_122_0 : Bool) (w_124_0 : Bool) (w_126_0 : Bool) (w_128_0 : Bool) (w_130_0 : Bool) (w_132_0 : Bool) (w_134_0 : Bool) (w_136_0 : Bool) (w_138_0 : Bool) (w_140_0 : Bool) (w_142_0 : Bool) (w_144_0 : Bool) (w_146_0 : Bool) (w_148_0 : Bool) (w_150_0 : Bool) (w_152_0 : Bool) (w_154_0 : Bool) (w_156_0 : Bool) (w_158_0 : Bool) (w_160_0 : Bool) (w_162_0 : Bool) (w_164_0 : Bool) (w_166_0 : Bool) (w_168_0 : Bool) (w_170_0 : Bool) (w_172_0 : Bool) (w_174_0 : Bool) (w_176_0 : Bool) (w_178_0 : Bool) (w_180_0 : Bool) (w_182_0 : Bool) (w_184_0 : Bool) (w_186_0 : Bool) (w_188_0 : Bool) (w_190_0 : Bool) (w_192_0 : Bool) (w_194_0 : Bool) (w_196_0 : Bool) (w_198_0 : Bool) (w_200_0 : Bool) (w_202_0 : Bool) (w_204_0 : Bool) (w_206_0 : Bool) (w_208_0 : Bool) (w_210_0 : Bool) (w_220_0 : Bool) (w_222_0 : Bool) (outputs : Bool) : Prop :=
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
  0 ≤ (160) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_68 params i (current, (w_67_0, ())) next) (Int.toNat (160)) w_66_0 w_68_0 ∧
  generatedRoot.constraints_2 params w_3_0 w_15_0 w_17_0 w_69_0 w_71_0 w_73_0 w_75_0 w_77_0 w_79_0 w_81_0 w_83_0 w_85_0 w_87_0 w_89_0 w_91_0 w_93_0 w_95_0 w_97_0 w_99_0 w_101_0 w_103_0 w_105_0 w_107_0 w_109_0 w_111_0 w_113_0 w_115_0 w_117_0 w_119_0 w_121_0 w_123_0 w_125_0 w_127_0 w_129_0 w_131_0 w_133_0 w_135_0 w_137_0 w_139_0 w_141_0 w_143_0 w_145_0 w_147_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_159_0 w_161_0 w_163_0 w_165_0 w_167_0 w_169_0 w_171_0 w_173_0 w_175_0 w_177_0 w_179_0 w_181_0 w_183_0 w_185_0 w_187_0 w_189_0 w_191_0 w_193_0 w_195_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_207_0 w_209_0 w_221_0 w_68_0 w_70_0 w_72_0 w_74_0 w_76_0 w_78_0 w_80_0 w_82_0 w_84_0 w_86_0 w_88_0 w_90_0 w_92_0 w_94_0 w_96_0 w_98_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_110_0 w_112_0 w_114_0 w_116_0 w_118_0 w_120_0 w_122_0 w_124_0 w_126_0 w_128_0 w_130_0 w_132_0 w_134_0 w_136_0 w_138_0 w_140_0 w_142_0 w_144_0 w_146_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_158_0 w_160_0 w_162_0 w_164_0 w_166_0 w_168_0 w_170_0 w_172_0 w_174_0 w_176_0 w_178_0 w_180_0 w_182_0 w_184_0 w_186_0 w_188_0 w_190_0 w_192_0 w_194_0 w_196_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_208_0 w_210_0 w_220_0 w_222_0 outputs

abbrev generatedRoot.constraints_0 (params : Params) (w_3_0 : Fin 5 → Int) (w_4_0 : Fin 5 → Int) (w_5_0 : Fin 5 → Int) (w_7_0 : Fin 240 → Int) (w_9_0 : Fin 240 → Int) (w_11_0 : Fin 240 → Int) (w_13_0 : Fin 240 → Int) (w_15_0 : Fin 40 → Int) (w_17_0 : Fin 40 → Int) (w_19_0 : Fin 800 → Int) (w_21_0 : Fin 800 → Int) (w_23_0 : Fin 800 → Int) (w_25_0 : Fin 800 → Int) (w_27_0 : Fin 800 → Int) (w_29_0 : Fin 800 → Int) (w_31_0 : Fin 800 → Int) (w_33_0 : Fin 800 → Int) (w_35_0 : Fin 800 → Int) (w_37_0 : Fin 800 → Int) (w_39_0 : Fin 800 → Int) (w_41_0 : Fin 800 → Int) (w_43_0 : Fin 800 → Int) (w_45_0 : Fin 800 → Int) (w_47_0 : Fin 800 → Int) (w_49_0 : Fin 800 → Int) (w_51_0 : Fin 800 → Int) (w_53_0 : Fin 800 → Int) (w_55_0 : Fin 800 → Int) (w_57_0 : Fin 800 → Int) (w_59_0 : Fin 800 → Int) (w_61_0 : Fin 800 → Int) (w_63_0 : Fin 800 → Int) (w_65_0 : Fin 800 → Int) (w_67_0 : Fin 800 → Int) (w_69_0 : Fin 800 → Int) (w_71_0 : Fin 800 → Int) (w_73_0 : Fin 800 → Int) (w_75_0 : Fin 800 → Int) (w_77_0 : Fin 800 → Int) (w_79_0 : Fin 800 → Int) (w_81_0 : Fin 800 → Int) (w_83_0 : Fin 800 → Int) (w_85_0 : Fin 800 → Int) (w_87_0 : Fin 800 → Int) (w_89_0 : Fin 800 → Int) (w_91_0 : Fin 800 → Int) (w_93_0 : Fin 800 → Int) (w_95_0 : Fin 800 → Int) (w_97_0 : Fin 800 → Int) (w_99_0 : Fin 800 → Int) (w_101_0 : Fin 800 → Int) (w_103_0 : Fin 800 → Int) (w_105_0 : Fin 800 → Int) (w_107_0 : Fin 800 → Int) (w_109_0 : Fin 800 → Int) (w_111_0 : Fin 800 → Int) (w_113_0 : Fin 800 → Int) (w_115_0 : Fin 800 → Int) (w_117_0 : Fin 800 → Int) (w_119_0 : Fin 800 → Int) (w_121_0 : Fin 800 → Int) (w_123_0 : Fin 800 → Int) (w_125_0 : Fin 800 → Int) (w_127_0 : Fin 800 → Int) (w_129_0 : Fin 800 → Int) (w_131_0 : Fin 800 → Int) (w_133_0 : Fin 800 → Int) (w_135_0 : Fin 800 → Int) (w_137_0 : Fin 800 → Int) (w_139_0 : Fin 800 → Int) (w_141_0 : Fin 800 → Int) (w_143_0 : Fin 800 → Int) (w_145_0 : Fin 800 → Int) (w_147_0 : Fin 800 → Int) (w_149_0 : Fin 800 → Int) (w_151_0 : Fin 800 → Int) (w_153_0 : Fin 800 → Int) (w_155_0 : Fin 800 → Int) (w_157_0 : Fin 800 → Int) (w_159_0 : Fin 800 → Int) (w_161_0 : Fin 800 → Int) (w_163_0 : Fin 800 → Int) (w_165_0 : Fin 800 → Int) (w_167_0 : Fin 800 → Int) (w_169_0 : Fin 800 → Int) (w_171_0 : Fin 800 → Int) (w_173_0 : Fin 800 → Int) (w_175_0 : Fin 800 → Int) (w_177_0 : Fin 800 → Int) (w_179_0 : Fin 800 → Int) (w_181_0 : Fin 800 → Int) (w_183_0 : Fin 800 → Int) (w_185_0 : Fin 800 → Int) (w_187_0 : Fin 800 → Int) (w_189_0 : Fin 800 → Int) (w_191_0 : Fin 800 → Int) (w_193_0 : Fin 800 → Int) (w_195_0 : Fin 800 → Int) (w_197_0 : Fin 800 → Int) (w_199_0 : Fin 800 → Int) (w_201_0 : Fin 800 → Int) (w_203_0 : Fin 800 → Int) (w_205_0 : Fin 800 → Int) (w_207_0 : Fin 800 → Int) (w_209_0 : Fin 800 → Int) (w_221_0 : Fin 8 → Int) (w_2_0 : Bool) (w_6_0 : Bool) (w_8_0 : Bool) (w_10_0 : Bool) (w_12_0 : Bool) (w_14_0 : Bool) (w_16_0 : Bool) (w_18_0 : Bool) (w_20_0 : Bool) (w_22_0 : Bool) (w_24_0 : Bool) (w_26_0 : Bool) (w_28_0 : Bool) (w_30_0 : Bool) (w_32_0 : Bool) (w_34_0 : Bool) (w_36_0 : Bool) (w_38_0 : Bool) (w_40_0 : Bool) (w_42_0 : Bool) (w_44_0 : Bool) (w_46_0 : Bool) (w_48_0 : Bool) (w_50_0 : Bool) (w_52_0 : Bool) (w_54_0 : Bool) (w_56_0 : Bool) (w_58_0 : Bool) (w_60_0 : Bool) (w_62_0 : Bool) (w_64_0 : Bool) (w_66_0 : Bool) (w_68_0 : Bool) (w_70_0 : Bool) (w_72_0 : Bool) (w_74_0 : Bool) (w_76_0 : Bool) (w_78_0 : Bool) (w_80_0 : Bool) (w_82_0 : Bool) (w_84_0 : Bool) (w_86_0 : Bool) (w_88_0 : Bool) (w_90_0 : Bool) (w_92_0 : Bool) (w_94_0 : Bool) (w_96_0 : Bool) (w_98_0 : Bool) (w_100_0 : Bool) (w_102_0 : Bool) (w_104_0 : Bool) (w_106_0 : Bool) (w_108_0 : Bool) (w_110_0 : Bool) (w_112_0 : Bool) (w_114_0 : Bool) (w_116_0 : Bool) (w_118_0 : Bool) (w_120_0 : Bool) (w_122_0 : Bool) (w_124_0 : Bool) (w_126_0 : Bool) (w_128_0 : Bool) (w_130_0 : Bool) (w_132_0 : Bool) (w_134_0 : Bool) (w_136_0 : Bool) (w_138_0 : Bool) (w_140_0 : Bool) (w_142_0 : Bool) (w_144_0 : Bool) (w_146_0 : Bool) (w_148_0 : Bool) (w_150_0 : Bool) (w_152_0 : Bool) (w_154_0 : Bool) (w_156_0 : Bool) (w_158_0 : Bool) (w_160_0 : Bool) (w_162_0 : Bool) (w_164_0 : Bool) (w_166_0 : Bool) (w_168_0 : Bool) (w_170_0 : Bool) (w_172_0 : Bool) (w_174_0 : Bool) (w_176_0 : Bool) (w_178_0 : Bool) (w_180_0 : Bool) (w_182_0 : Bool) (w_184_0 : Bool) (w_186_0 : Bool) (w_188_0 : Bool) (w_190_0 : Bool) (w_192_0 : Bool) (w_194_0 : Bool) (w_196_0 : Bool) (w_198_0 : Bool) (w_200_0 : Bool) (w_202_0 : Bool) (w_204_0 : Bool) (w_206_0 : Bool) (w_208_0 : Bool) (w_210_0 : Bool) (w_220_0 : Bool) (w_222_0 : Bool) (outputs : Bool) : Prop :=
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
  generatedRoot.constraints_1 params w_3_0 w_15_0 w_17_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_67_0 w_69_0 w_71_0 w_73_0 w_75_0 w_77_0 w_79_0 w_81_0 w_83_0 w_85_0 w_87_0 w_89_0 w_91_0 w_93_0 w_95_0 w_97_0 w_99_0 w_101_0 w_103_0 w_105_0 w_107_0 w_109_0 w_111_0 w_113_0 w_115_0 w_117_0 w_119_0 w_121_0 w_123_0 w_125_0 w_127_0 w_129_0 w_131_0 w_133_0 w_135_0 w_137_0 w_139_0 w_141_0 w_143_0 w_145_0 w_147_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_159_0 w_161_0 w_163_0 w_165_0 w_167_0 w_169_0 w_171_0 w_173_0 w_175_0 w_177_0 w_179_0 w_181_0 w_183_0 w_185_0 w_187_0 w_189_0 w_191_0 w_193_0 w_195_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_207_0 w_209_0 w_221_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_68_0 w_70_0 w_72_0 w_74_0 w_76_0 w_78_0 w_80_0 w_82_0 w_84_0 w_86_0 w_88_0 w_90_0 w_92_0 w_94_0 w_96_0 w_98_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_110_0 w_112_0 w_114_0 w_116_0 w_118_0 w_120_0 w_122_0 w_124_0 w_126_0 w_128_0 w_130_0 w_132_0 w_134_0 w_136_0 w_138_0 w_140_0 w_142_0 w_144_0 w_146_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_158_0 w_160_0 w_162_0 w_164_0 w_166_0 w_168_0 w_170_0 w_172_0 w_174_0 w_176_0 w_178_0 w_180_0 w_182_0 w_184_0 w_186_0 w_188_0 w_190_0 w_192_0 w_194_0 w_196_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_208_0 w_210_0 w_220_0 w_222_0 outputs

abbrev generatedRoot.body (params : Params) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 40 → Int) × (Fin 40 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × Int × Int × (Fin 8 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
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
  let w_67_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_69_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_71_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_73_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_77_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_85_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_87_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_89_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_91_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_93_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_95_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_97_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_99_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_101_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_103_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_105_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_107_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_109_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_111_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_113_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_115_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_117_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_119_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_121_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_123_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_125_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_127_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_129_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_131_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_133_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_135_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_137_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_139_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_141_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_143_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_145_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_147_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_149_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_151_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_153_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_155_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_157_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_159_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_161_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_163_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_165_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_167_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_169_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_171_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_173_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_175_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_177_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_179_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_181_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_183_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_185_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_187_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_189_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_191_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_193_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_195_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_197_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_199_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_201_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_203_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_205_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_207_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_209_0 : Fin 800 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_212_0 : Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_214_0 : Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_221_0 : Fin 8 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  let w_68_0 := witness.w_68_0
  let w_70_0 := witness.w_70_0
  let w_72_0 := witness.w_72_0
  let w_74_0 := witness.w_74_0
  let w_76_0 := witness.w_76_0
  let w_78_0 := witness.w_78_0
  let w_80_0 := witness.w_80_0
  let w_82_0 := witness.w_82_0
  let w_84_0 := witness.w_84_0
  let w_86_0 := witness.w_86_0
  let w_88_0 := witness.w_88_0
  let w_90_0 := witness.w_90_0
  let w_92_0 := witness.w_92_0
  let w_94_0 := witness.w_94_0
  let w_96_0 := witness.w_96_0
  let w_98_0 := witness.w_98_0
  let w_100_0 := witness.w_100_0
  let w_102_0 := witness.w_102_0
  let w_104_0 := witness.w_104_0
  let w_106_0 := witness.w_106_0
  let w_108_0 := witness.w_108_0
  let w_110_0 := witness.w_110_0
  let w_112_0 := witness.w_112_0
  let w_114_0 := witness.w_114_0
  let w_116_0 := witness.w_116_0
  let w_118_0 := witness.w_118_0
  let w_120_0 := witness.w_120_0
  let w_122_0 := witness.w_122_0
  let w_124_0 := witness.w_124_0
  let w_126_0 := witness.w_126_0
  let w_128_0 := witness.w_128_0
  let w_130_0 := witness.w_130_0
  let w_132_0 := witness.w_132_0
  let w_134_0 := witness.w_134_0
  let w_136_0 := witness.w_136_0
  let w_138_0 := witness.w_138_0
  let w_140_0 := witness.w_140_0
  let w_142_0 := witness.w_142_0
  let w_144_0 := witness.w_144_0
  let w_146_0 := witness.w_146_0
  let w_148_0 := witness.w_148_0
  let w_150_0 := witness.w_150_0
  let w_152_0 := witness.w_152_0
  let w_154_0 := witness.w_154_0
  let w_156_0 := witness.w_156_0
  let w_158_0 := witness.w_158_0
  let w_160_0 := witness.w_160_0
  let w_162_0 := witness.w_162_0
  let w_164_0 := witness.w_164_0
  let w_166_0 := witness.w_166_0
  let w_168_0 := witness.w_168_0
  let w_170_0 := witness.w_170_0
  let w_172_0 := witness.w_172_0
  let w_174_0 := witness.w_174_0
  let w_176_0 := witness.w_176_0
  let w_178_0 := witness.w_178_0
  let w_180_0 := witness.w_180_0
  let w_182_0 := witness.w_182_0
  let w_184_0 := witness.w_184_0
  let w_186_0 := witness.w_186_0
  let w_188_0 := witness.w_188_0
  let w_190_0 := witness.w_190_0
  let w_192_0 := witness.w_192_0
  let w_194_0 := witness.w_194_0
  let w_196_0 := witness.w_196_0
  let w_198_0 := witness.w_198_0
  let w_200_0 := witness.w_200_0
  let w_202_0 := witness.w_202_0
  let w_204_0 := witness.w_204_0
  let w_206_0 := witness.w_206_0
  let w_208_0 := witness.w_208_0
  let w_210_0 := witness.w_210_0
  let w_211_0 : Int := if w_210_0 then 1 else 0
  let w_213_0 : Int := 1
  let w_215_0 : Int := (w_213_0 - w_214_0)
  let w_216_0 : Bool := decide (w_212_0 = w_215_0)
  let w_217_0 : Int := if w_216_0 then 1 else 0
  let w_218_0 : Int := (w_211_0 + w_217_0)
  let w_219_0 : Int := 2
  let w_220_0 : Bool := decide (w_218_0 = w_219_0)
  let w_222_0 := witness.w_222_0
  generatedRoot.constraints_0 params w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_67_0 w_69_0 w_71_0 w_73_0 w_75_0 w_77_0 w_79_0 w_81_0 w_83_0 w_85_0 w_87_0 w_89_0 w_91_0 w_93_0 w_95_0 w_97_0 w_99_0 w_101_0 w_103_0 w_105_0 w_107_0 w_109_0 w_111_0 w_113_0 w_115_0 w_117_0 w_119_0 w_121_0 w_123_0 w_125_0 w_127_0 w_129_0 w_131_0 w_133_0 w_135_0 w_137_0 w_139_0 w_141_0 w_143_0 w_145_0 w_147_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_159_0 w_161_0 w_163_0 w_165_0 w_167_0 w_169_0 w_171_0 w_173_0 w_175_0 w_177_0 w_179_0 w_181_0 w_183_0 w_185_0 w_187_0 w_189_0 w_191_0 w_193_0 w_195_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_207_0 w_209_0 w_221_0 w_2_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_68_0 w_70_0 w_72_0 w_74_0 w_76_0 w_78_0 w_80_0 w_82_0 w_84_0 w_86_0 w_88_0 w_90_0 w_92_0 w_94_0 w_96_0 w_98_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_110_0 w_112_0 w_114_0 w_116_0 w_118_0 w_120_0 w_122_0 w_124_0 w_126_0 w_128_0 w_130_0 w_132_0 w_134_0 w_136_0 w_138_0 w_140_0 w_142_0 w_144_0 w_146_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_158_0 w_160_0 w_162_0 w_164_0 w_166_0 w_168_0 w_170_0 w_172_0 w_174_0 w_176_0 w_178_0 w_180_0 w_182_0 w_184_0 w_186_0 w_188_0 w_190_0 w_192_0 w_194_0 w_196_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_208_0 w_210_0 w_220_0 w_222_0 outputs

def generatedRoot (params : Params) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 240 → Int) × (Fin 40 → Int) × (Fin 40 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × (Fin 800 → Int) × Int × Int × (Fin 8 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Requirement_0
