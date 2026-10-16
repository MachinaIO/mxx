import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Requirement_0

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_3.constraints_0 (w_2_0 : Fin 5 → Int) (w_5_0 : Fin 5 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_6_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  outputs = w_11_0

def sequential_generatedRoot_3 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 5 → Int) × (Fin 5 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 5 → Int := inputs.2.1
  let w_5_0 : Fin 5 → Int := inputs.2.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : Int),
    ∃ (w_6_0 : Int),
      let w_7_0 : Bool := decide (w_4_0 = w_6_0)
      let w_8_0 : Int := if w_7_0 then 1 else 0
      let w_9_0 : Int := (w_1_0 + w_8_0)
      let w_10_0 : Int := 2
      let w_11_0 : Bool := decide (w_9_0 = w_10_0)
      sequential_generatedRoot_3.constraints_0 w_2_0 w_5_0 w_3_0 w_4_0 w_6_0 w_11_0 outputs

abbrev sequential_generatedRoot_6.constraints_0 (w_2_0 : Fin 2 → Int) (w_9_0 : Fin 2 → Int) (w_21_0 : Fin 5 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_10_0 : Int) (w_20_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_26_0 : Int) (w_29_0 : Int) (w_30_0 : Int) (w_43_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_3_0 w_10_0 ∧
  w_22_0 ≠ 0 ∧
  0 ≤ w_25_0 ∧
  w_25_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_25_0 w_26_0 ∧
  0 ≤ w_20_0 ∧
  w_20_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_20_0 [w_26_0, w_29_0] w_30_0 ∧
  outputs = w_43_0

def sequential_generatedRoot_6 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 5 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 2 → Int := inputs.2.1
  let w_9_0 : Fin 2 → Int := inputs.2.2.1
  let w_21_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : Int),
    let w_5_0 : Int := (w_4_0 * w_4_0)
    let w_6_0 : Int := 1
    let w_7_0 : Bool := decide (w_5_0 = w_6_0)
    let w_8_0 : Int := if w_7_0 then 1 else 0
    ∃ (w_10_0 : Int),
      let w_11_0 : Int := 2
      let w_12_0 : Int := (w_10_0 * w_11_0)
      let w_13_0 : Int := 1
      let w_14_0 : Int := (w_13_0 - w_4_0)
      let w_15_0 : Int := 8
      let w_16_0 : Int := (w_14_0 * w_15_0)
      let w_17_0 : Int := (w_12_0 + w_16_0)
      let w_18_0 : Int := 1
      let w_19_0 : Bool := decide (w_3_0 = w_18_0)
      let w_20_0 : Int := if w_19_0 then 1 else 0
      let w_22_0 : Int := 1
      let w_23_0 : Int := (w_3_0 % w_22_0)
      let w_24_0 : Int := 5
      let w_25_0 : Int := (w_23_0 * w_24_0)
      ∃ (w_26_0 : Int),
        let w_27_0 : Int := 0
        let w_28_0 : Int := 0
        let w_29_0 : Int := (w_27_0 + w_28_0)
        ∃ (w_30_0 : Int),
          let w_31_0 : Int := 1
          let w_32_0 : Int := (w_30_0 * w_31_0)
          let w_33_0 : Int := 2
          let w_34_0 : Int := (w_32_0 * w_33_0)
          let w_35_0 : Bool := decide (w_17_0 = w_34_0)
          let w_36_0 : Int := if w_35_0 then 1 else 0
          let w_37_0 : Int := (w_8_0 + w_36_0)
          let w_38_0 : Int := 2
          let w_39_0 : Bool := decide (w_37_0 = w_38_0)
          let w_40_0 : Int := if w_39_0 then 1 else 0
          let w_41_0 : Int := (w_1_0 + w_40_0)
          let w_42_0 : Int := 2
          let w_43_0 : Bool := decide (w_41_0 = w_42_0)
          sequential_generatedRoot_6.constraints_0 w_2_0 w_9_0 w_21_0 w_3_0 w_4_0 w_10_0 w_20_0 w_22_0 w_25_0 w_26_0 w_29_0 w_30_0 w_43_0 outputs

abbrev sequential_parallel_generatedRoot_41_11.constraints_0 (i_1 : Nat) (w_8_0 : Fin 5 → Int) (w_4_0 : Int) (w_7_0 : Int) (w_26_0 : Int) (w_27_0 : Int) (w_28_0 : Int) (w_29_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((2 * (Int.ofNat i_1))) ∧
  0 ≤ ((1 + (2 * (Int.ofNat i_1)))) ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_26_0 w_27_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_4_0 [w_7_0, w_27_0] w_28_0 ∧
  outputs = w_29_0

def sequential_parallel_generatedRoot_41_11 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × Int × (Fin 5 → Int) × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_2_0 : Int := inputs.2.1
  let w_8_0 : Fin 5 → Int := inputs.2.2.1
  let w_9_0 : Int := inputs.2.2.2.1
  let w_13_0 : Int := inputs.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_1)
  let w_3_0 : Bool := decide (w_1_0 < w_2_0)
  let w_4_0 : Int := if w_3_0 then 1 else 0
  let w_5_0 : Int := 0
  let w_6_0 : Int := 0
  let w_7_0 : Int := (w_5_0 + w_6_0)
  let w_10_0 : Int := 5
  let w_11_0 : Int := (w_9_0 * w_10_0)
  let w_12_0 : Int := 0
  let w_14_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((2 * (Int.ofNat i_1)))))) % 2 = 1)
  let w_15_0 : Int := if w_14_0 then 1 else 0
  let w_16_0 : Int := 1
  let w_17_0 : Int := (w_15_0 * w_16_0)
  let w_18_0 : Int := (w_12_0 + w_17_0)
  let w_19_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_20_0 : Int := if w_19_0 then 1 else 0
  let w_21_0 : Int := 2
  let w_22_0 : Int := (w_20_0 * w_21_0)
  let w_23_0 : Int := (w_18_0 + w_22_0)
  let w_24_0 : Int := (w_11_0 + w_23_0)
  let w_25_0 : Int := 1
  let w_26_0 : Int := (w_24_0 + w_25_0)
  ∃ (w_27_0 : Int),
    ∃ (w_28_0 : Int),
      let w_29_0 : Int := (w_0_0 + w_28_0)
      sequential_parallel_generatedRoot_41_11.constraints_0 i_1 w_8_0 w_4_0 w_7_0 w_26_0 w_27_0 w_28_0 w_29_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_41.Witness where
  w_5_0 : Int
  w_10_0 : Int
  w_11_0 : Int

abbrev parallel_generatedRoot_41.constraints_0 (params : Params) (i_0 : Nat) (w_1_0 : Fin 5 → Int) (w_6_0 : Fin 5 → Int) (w_9_0 : Fin 5 → Int) (w_0_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (w_10_0 : Int) (w_11_0 : Int) (w_12_0 : Int) (w_13_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  w_7_0 ≠ 0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_4_0 w_10_0 ∧
  0 ≤ (4) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_parallel_generatedRoot_41_11 params i_0 i (current, (w_5_0, (w_6_0, (w_8_0, (w_10_0, ()))))) next) (Int.toNat (4)) w_0_0 w_11_0 ∧
  w_12_0 ≠ 0 ∧
  outputs = w_13_0

abbrev parallel_generatedRoot_41.body (params : Params) (i_0 : Nat) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × Unit) (outputs : Int) (witness : parallel_generatedRoot_41.Witness) : Prop :=
  let _params := params
  let w_1_0 : Fin 5 → Int := inputs.1
  let w_6_0 : Fin 5 → Int := inputs.2.1
  let w_9_0 : Fin 5 → Int := inputs.2.2.1
  let w_0_0 : Int := 0
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 : Int := 5
  let w_4_0 : Int := (w_2_0 % w_3_0)
  let w_5_0 := witness.w_5_0
  let w_7_0 : Int := 5
  let w_8_0 : Int := (w_2_0 / w_7_0)
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : Int := 16
  let w_13_0 : Int := (w_11_0 % w_12_0)
  parallel_generatedRoot_41.constraints_0 params i_0 w_1_0 w_6_0 w_9_0 w_0_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 outputs

def parallel_generatedRoot_41 (params : Params) (i_0 : Nat) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × Unit) (outputs : Int) : Prop :=
  ∃ witness : parallel_generatedRoot_41.Witness,
    parallel_generatedRoot_41.body params i_0 inputs outputs witness

abbrev sequential_generatedRoot_42.constraints_0 (w_2_0 : Fin 2 → Int) (w_9_0 : Fin 2 → Int) (w_21_0 : Fin 5 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_10_0 : Int) (w_20_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_26_0 : Int) (w_29_0 : Int) (w_30_0 : Int) (w_43_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_3_0 w_10_0 ∧
  w_22_0 ≠ 0 ∧
  0 ≤ w_25_0 ∧
  w_25_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_25_0 w_26_0 ∧
  0 ≤ w_20_0 ∧
  w_20_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_20_0 [w_26_0, w_29_0] w_30_0 ∧
  outputs = w_43_0

def sequential_generatedRoot_42 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 5 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 2 → Int := inputs.2.1
  let w_9_0 : Fin 2 → Int := inputs.2.2.1
  let w_21_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : Int),
    let w_5_0 : Int := (w_4_0 * w_4_0)
    let w_6_0 : Int := 1
    let w_7_0 : Bool := decide (w_5_0 = w_6_0)
    let w_8_0 : Int := if w_7_0 then 1 else 0
    ∃ (w_10_0 : Int),
      let w_11_0 : Int := 2
      let w_12_0 : Int := (w_10_0 * w_11_0)
      let w_13_0 : Int := 1
      let w_14_0 : Int := (w_13_0 - w_4_0)
      let w_15_0 : Int := 8
      let w_16_0 : Int := (w_14_0 * w_15_0)
      let w_17_0 : Int := (w_12_0 + w_16_0)
      let w_18_0 : Int := 1
      let w_19_0 : Bool := decide (w_3_0 = w_18_0)
      let w_20_0 : Int := if w_19_0 then 1 else 0
      let w_22_0 : Int := 1
      let w_23_0 : Int := (w_3_0 % w_22_0)
      let w_24_0 : Int := 5
      let w_25_0 : Int := (w_23_0 * w_24_0)
      ∃ (w_26_0 : Int),
        let w_27_0 : Int := 0
        let w_28_0 : Int := 0
        let w_29_0 : Int := (w_27_0 + w_28_0)
        ∃ (w_30_0 : Int),
          let w_31_0 : Int := 1
          let w_32_0 : Int := (w_30_0 * w_31_0)
          let w_33_0 : Int := 2
          let w_34_0 : Int := (w_32_0 * w_33_0)
          let w_35_0 : Bool := decide (w_17_0 = w_34_0)
          let w_36_0 : Int := if w_35_0 then 1 else 0
          let w_37_0 : Int := (w_8_0 + w_36_0)
          let w_38_0 : Int := 2
          let w_39_0 : Bool := decide (w_37_0 = w_38_0)
          let w_40_0 : Int := if w_39_0 then 1 else 0
          let w_41_0 : Int := (w_1_0 + w_40_0)
          let w_42_0 : Int := 2
          let w_43_0 : Bool := decide (w_41_0 = w_42_0)
          sequential_generatedRoot_42.constraints_0 w_2_0 w_9_0 w_21_0 w_3_0 w_4_0 w_10_0 w_20_0 w_22_0 w_25_0 w_26_0 w_29_0 w_30_0 w_43_0 outputs

abbrev sequential_sequential_generatedRoot_74_5.constraints_0 (w_1_0 : Fin 72000 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_74_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 72000 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 72000 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 2
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

abbrev sequential_generatedRoot_74.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 72000 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_74_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_74.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 72000 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_74.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 72000 → Int := inputs.2.1
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

def sequential_generatedRoot_74 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 72000 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_74.Witness,
    sequential_generatedRoot_74.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_76_5.constraints_0 (w_1_0 : Fin 72000 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_76_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 72000 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 72000 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 2
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

abbrev sequential_generatedRoot_76.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 72000 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_76_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_76.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 72000 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_76.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 72000 → Int := inputs.2.1
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

def sequential_generatedRoot_76 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 72000 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_76.Witness,
    sequential_generatedRoot_76.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_78_5.constraints_0 (w_1_0 : Fin 16 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_78_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 16 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 16 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 2
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

abbrev sequential_generatedRoot_78.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 16 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_78_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_78.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_78.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 16 → Int := inputs.2.1
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

def sequential_generatedRoot_78 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_78.Witness,
    sequential_generatedRoot_78.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_80_5.constraints_0 (w_1_0 : Fin 16 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_80_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 16 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 16 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 2
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

abbrev sequential_generatedRoot_80.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 16 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_80_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_80.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_80.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 16 → Int := inputs.2.1
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

def sequential_generatedRoot_80 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_80.Witness,
    sequential_generatedRoot_80.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_82_5.constraints_0 (w_1_0 : Fin 14400 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_82_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 14400 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14400 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 2
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

abbrev sequential_generatedRoot_82.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 14400 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_82_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_82.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 14400 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_82.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 14400 → Int := inputs.2.1
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

def sequential_generatedRoot_82 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 14400 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_82.Witness,
    sequential_generatedRoot_82.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_84_5.constraints_0 (w_1_0 : Fin 14400 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_84_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 14400 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14400 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 2
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

abbrev sequential_generatedRoot_84.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 14400 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_84_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_84.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 14400 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_84.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 14400 → Int := inputs.2.1
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

def sequential_generatedRoot_84 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 14400 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_84.Witness,
    sequential_generatedRoot_84.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_86_5.constraints_0 (w_1_0 : Fin 16 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_86_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 16 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 16 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 2
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

abbrev sequential_generatedRoot_86.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 16 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_86_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_86.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_86.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 16 → Int := inputs.2.1
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

def sequential_generatedRoot_86 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_86.Witness,
    sequential_generatedRoot_86.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_88_5.constraints_0 (w_1_0 : Fin 16 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_88_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 16 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 16 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 2
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

abbrev sequential_generatedRoot_88.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 16 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_88_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_88.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_88.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 16 → Int := inputs.2.1
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

def sequential_generatedRoot_88 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_88.Witness,
    sequential_generatedRoot_88.body params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Bool
  w_6_0 : Bool
  w_9_0 : Int
  w_15_0 : Int
  w_24_0 : Int
  w_40_0 : Fin 5 → Int
  w_40_1 : Fin 5 → Int
  w_41_0 : Fin 5 → Int
  w_42_0 : Bool
  w_45_0 : Int
  w_51_0 : Int
  w_59_0 : Int
  w_74_0 : Bool
  w_76_0 : Bool
  w_78_0 : Bool
  w_80_0 : Bool
  w_82_0 : Bool
  w_84_0 : Bool
  w_86_0 : Bool
  w_88_0 : Bool

abbrev generatedRoot.constraints_0 (params : Params) (w_1_0 : Fin 5 → Int) (w_2_0 : Fin 5 → Int) (w_4_0 : Fin 2 → Int) (w_5_0 : Fin 2 → Int) (w_8_0 : Fin 1 → Int) (w_14_0 : Fin 1 → Int) (w_23_0 : Fin 2 → Int) (w_38_0 : Fin 2 → Int) (w_39_0 : Fin 2 → Int) (w_44_0 : Fin 1 → Int) (w_50_0 : Fin 1 → Int) (w_73_0 : Fin 72000 → Int) (w_75_0 : Fin 72000 → Int) (w_77_0 : Fin 16 → Int) (w_79_0 : Fin 16 → Int) (w_81_0 : Fin 14400 → Int) (w_83_0 : Fin 14400 → Int) (w_85_0 : Fin 16 → Int) (w_87_0 : Fin 16 → Int) (w_0_0 : Bool) (w_3_0 : Bool) (w_6_0 : Bool) (w_9_0 : Int) (w_15_0 : Int) (w_24_0 : Int) (w_37_0 : Bool) (w_40_0 : Fin 5 → Int) (w_40_1 : Fin 5 → Int) (w_41_0 : Fin 5 → Int) (w_42_0 : Bool) (w_45_0 : Int) (w_51_0 : Int) (w_59_0 : Int) (w_72_0 : Bool) (w_74_0 : Bool) (w_76_0 : Bool) (w_78_0 : Bool) (w_80_0 : Bool) (w_82_0 : Bool) (w_84_0 : Bool) (w_86_0 : Bool) (w_88_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_3 params i (current, (w_1_0, (w_2_0, ()))) next) (Int.toNat (5)) w_0_0 w_3_0 ∧
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 params i (current, (w_4_0, (w_5_0, (w_2_0, ())))) next) (Int.toNat (2)) w_3_0 w_6_0 ∧
  MxxRuntime.familyGetStatic w_8_0 (0) w_9_0 ∧
  MxxRuntime.familyGetStatic w_14_0 (0) w_15_0 ∧
  MxxRuntime.familyGetStatic w_23_0 (0) w_24_0 ∧
  MxxRuntime.sha256BitColumns [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 w_40_0 w_40_1 ∧
  (5) = 5 ∧
  (∀ i : Fin 5, parallel_generatedRoot_41 params i (w_40_1, (w_2_0, (w_40_0, ()))) (w_41_0 i)) ∧
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_42 params i (current, (w_38_0, (w_39_0, (w_41_0, ())))) next) (Int.toNat (2)) w_37_0 w_42_0 ∧
  MxxRuntime.familyGetStatic w_44_0 (0) w_45_0 ∧
  MxxRuntime.familyGetStatic w_50_0 (0) w_51_0 ∧
  MxxRuntime.familyGetStatic w_23_0 (1) w_59_0 ∧
  0 ≤ (36000) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_74 params i (current, (w_73_0, ())) next) (Int.toNat (36000)) w_72_0 w_74_0 ∧
  0 ≤ (36000) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_76 params i (current, (w_75_0, ())) next) (Int.toNat (36000)) w_74_0 w_76_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_78 params i (current, (w_77_0, ())) next) (Int.toNat (8)) w_76_0 w_78_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_80 params i (current, (w_79_0, ())) next) (Int.toNat (8)) w_78_0 w_80_0 ∧
  0 ≤ (7200) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_82 params i (current, (w_81_0, ())) next) (Int.toNat (7200)) w_80_0 w_82_0 ∧
  0 ≤ (7200) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_84 params i (current, (w_83_0, ())) next) (Int.toNat (7200)) w_82_0 w_84_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_86 params i (current, (w_85_0, ())) next) (Int.toNat (8)) w_84_0 w_86_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_88 params i (current, (w_87_0, ())) next) (Int.toNat (8)) w_86_0 w_88_0 ∧
  outputs = w_88_0

abbrev generatedRoot.body (params : Params) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : Fin 5 → Int := inputs.1
  let w_2_0 : Fin 5 → Int := inputs.2.1
  let w_4_0 : Fin 2 → Int := inputs.2.2.1
  let w_5_0 : Fin 2 → Int := inputs.2.2.2.1
  let w_8_0 : Fin 1 → Int := inputs.2.2.2.2.1
  let w_14_0 : Fin 1 → Int := inputs.2.2.2.2.2.1
  let w_23_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.1
  let w_38_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.1
  let w_39_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_44_0 : Fin 1 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : Fin 1 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_73_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_77_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_85_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_87_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : Bool := true
  let w_3_0 := witness.w_3_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_9_0 := witness.w_9_0
  let w_10_0 : Int := (w_9_0 * w_9_0)
  let w_11_0 : Int := 1
  let w_12_0 : Bool := decide (w_10_0 = w_11_0)
  let w_13_0 : Int := if w_12_0 then 1 else 0
  let w_15_0 := witness.w_15_0
  let w_16_0 : Int := 2
  let w_17_0 : Int := (w_15_0 * w_16_0)
  let w_18_0 : Int := 1
  let w_19_0 : Int := (w_18_0 - w_9_0)
  let w_20_0 : Int := 8
  let w_21_0 : Int := (w_19_0 * w_20_0)
  let w_22_0 : Int := (w_17_0 + w_21_0)
  let w_24_0 := witness.w_24_0
  let w_25_0 : Int := 2
  let w_26_0 : Int := (w_24_0 * w_25_0)
  let w_27_0 : Int := 2
  let w_28_0 : Int := (w_26_0 * w_27_0)
  let w_29_0 : Bool := decide (w_22_0 = w_28_0)
  let w_30_0 : Int := if w_29_0 then 1 else 0
  let w_31_0 : Int := (w_13_0 + w_30_0)
  let w_32_0 : Int := 2
  let w_33_0 : Bool := decide (w_31_0 = w_32_0)
  let w_34_0 : Int := if w_33_0 then 1 else 0
  let w_35_0 : Int := (w_7_0 + w_34_0)
  let w_36_0 : Int := 2
  let w_37_0 : Bool := decide (w_35_0 = w_36_0)
  let w_40_0 := witness.w_40_0
  let w_40_1 := witness.w_40_1
  let w_41_0 := witness.w_41_0
  let w_42_0 := witness.w_42_0
  let w_43_0 : Int := if w_42_0 then 1 else 0
  let w_45_0 := witness.w_45_0
  let w_46_0 : Int := (w_45_0 * w_45_0)
  let w_47_0 : Int := 1
  let w_48_0 : Bool := decide (w_46_0 = w_47_0)
  let w_49_0 : Int := if w_48_0 then 1 else 0
  let w_51_0 := witness.w_51_0
  let w_52_0 : Int := 2
  let w_53_0 : Int := (w_51_0 * w_52_0)
  let w_54_0 : Int := 1
  let w_55_0 : Int := (w_54_0 - w_45_0)
  let w_56_0 : Int := 8
  let w_57_0 : Int := (w_55_0 * w_56_0)
  let w_58_0 : Int := (w_53_0 + w_57_0)
  let w_59_0 := witness.w_59_0
  let w_60_0 : Int := 2
  let w_61_0 : Int := (w_59_0 * w_60_0)
  let w_62_0 : Int := 2
  let w_63_0 : Int := (w_61_0 * w_62_0)
  let w_64_0 : Bool := decide (w_58_0 = w_63_0)
  let w_65_0 : Int := if w_64_0 then 1 else 0
  let w_66_0 : Int := (w_49_0 + w_65_0)
  let w_67_0 : Int := 2
  let w_68_0 : Bool := decide (w_66_0 = w_67_0)
  let w_69_0 : Int := if w_68_0 then 1 else 0
  let w_70_0 : Int := (w_43_0 + w_69_0)
  let w_71_0 : Int := 2
  let w_72_0 : Bool := decide (w_70_0 = w_71_0)
  let w_74_0 := witness.w_74_0
  let w_76_0 := witness.w_76_0
  let w_78_0 := witness.w_78_0
  let w_80_0 := witness.w_80_0
  let w_82_0 := witness.w_82_0
  let w_84_0 := witness.w_84_0
  let w_86_0 := witness.w_86_0
  let w_88_0 := witness.w_88_0
  generatedRoot.constraints_0 params w_1_0 w_2_0 w_4_0 w_5_0 w_8_0 w_14_0 w_23_0 w_38_0 w_39_0 w_44_0 w_50_0 w_73_0 w_75_0 w_77_0 w_79_0 w_81_0 w_83_0 w_85_0 w_87_0 w_0_0 w_3_0 w_6_0 w_9_0 w_15_0 w_24_0 w_37_0 w_40_0 w_40_1 w_41_0 w_42_0 w_45_0 w_51_0 w_59_0 w_72_0 w_74_0 w_76_0 w_78_0 w_80_0 w_82_0 w_84_0 w_86_0 w_88_0 outputs

def generatedRoot (params : Params) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end W_Requirement_0
