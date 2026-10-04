import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Requirement_0

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_6.constraints_0 (w_2_0 : Fin 17 → Int) (w_13_0 : Fin 17 → Int) (w_22_0 : Fin 17 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_14_0 : Int) (w_23_0 : Int) (w_30_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 17 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 17 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_3_0 w_14_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 17 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_3_0 w_23_0 ∧
  outputs = w_30_0

def sequential_generatedRoot_6 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 17 → Int) × (Fin 17 → Int) × (Fin 17 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 17 → Int := inputs.2.1
  let w_13_0 : Fin 17 → Int := inputs.2.2.1
  let w_22_0 : Fin 17 → Int := inputs.2.2.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : Int),
    let w_5_0 : Int := (w_4_0 * w_4_0)
    let w_6_0 : Int := 1
    let w_7_0 : Bool := decide (w_5_0 = w_6_0)
    let w_8_0 : Int := if w_7_0 then 1 else 0
    let w_9_0 : Int := (w_1_0 + w_8_0)
    let w_10_0 : Int := 2
    let w_11_0 : Bool := decide (w_9_0 = w_10_0)
    let w_12_0 : Int := if w_11_0 then 1 else 0
    ∃ (w_14_0 : Int),
      let w_15_0 : Int := 2
      let w_16_0 : Int := (w_14_0 * w_15_0)
      let w_17_0 : Int := 1
      let w_18_0 : Int := (w_17_0 - w_4_0)
      let w_19_0 : Int := 32
      let w_20_0 : Int := (w_18_0 * w_19_0)
      let w_21_0 : Int := (w_16_0 + w_20_0)
      ∃ (w_23_0 : Int),
        let w_24_0 : Int := 4
        let w_25_0 : Int := (w_23_0 * w_24_0)
        let w_26_0 : Bool := decide (w_21_0 = w_25_0)
        let w_27_0 : Int := if w_26_0 then 1 else 0
        let w_28_0 : Int := (w_12_0 + w_27_0)
        let w_29_0 : Int := 2
        let w_30_0 : Bool := decide (w_28_0 = w_29_0)
        sequential_generatedRoot_6.constraints_0 w_2_0 w_13_0 w_22_0 w_3_0 w_4_0 w_14_0 w_23_0 w_30_0 outputs

abbrev sequential_generatedRoot_10.constraints_0 (w_1_0 : Fin 14 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_10 (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_10.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_20.constraints_0 (w_1_0 : Fin 14 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_20 (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_20.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_30.constraints_0 (w_1_0 : Fin 14 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_30 (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_30.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_40.constraints_0 (w_1_0 : Fin 14 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_40 (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_40.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_50.constraints_0 (w_1_0 : Fin 14 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_50 (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_50.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_6_0 : Bool
  w_10_0 : Int
  w_20_0 : Int
  w_30_0 : Int
  w_40_0 : Int
  w_50_0 : Int

abbrev generatedRoot.constraints_0 (params : Params) (w_3_0 : Fin 17 → Int) (w_4_0 : Fin 17 → Int) (w_5_0 : Fin 17 → Int) (w_9_0 : Fin 14 → Int) (w_19_0 : Fin 14 → Int) (w_29_0 : Fin 14 → Int) (w_39_0 : Fin 14 → Int) (w_49_0 : Fin 14 → Int) (w_2_0 : Bool) (w_6_0 : Bool) (w_8_0 : Int) (w_10_0 : Int) (w_18_0 : Int) (w_20_0 : Int) (w_28_0 : Int) (w_30_0 : Int) (w_38_0 : Int) (w_40_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (w_56_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (17) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 params i (current, (w_3_0, (w_4_0, (w_5_0, ())))) next) (Int.toNat (17)) w_2_0 w_6_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_10 params i (current, (w_9_0, ())) next) (Int.toNat (14)) w_8_0 w_10_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_20 params i (current, (w_19_0, ())) next) (Int.toNat (14)) w_18_0 w_20_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_30 params i (current, (w_29_0, ())) next) (Int.toNat (14)) w_28_0 w_30_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_40 params i (current, (w_39_0, ())) next) (Int.toNat (14)) w_38_0 w_40_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_50 params i (current, (w_49_0, ())) next) (Int.toNat (14)) w_48_0 w_50_0 ∧
  outputs = w_56_0

abbrev generatedRoot.body (params : Params) (inputs : (Fin 17 → Int) × (Fin 17 → Int) × (Fin 17 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_3_0 : Fin 17 → Int := inputs.1
  let w_4_0 : Fin 17 → Int := inputs.2.1
  let w_5_0 : Fin 17 → Int := inputs.2.2.1
  let w_9_0 : Fin 14 → Int := inputs.2.2.2.1
  let w_19_0 : Fin 14 → Int := inputs.2.2.2.2.1
  let w_29_0 : Fin 14 → Int := inputs.2.2.2.2.2.1
  let w_39_0 : Fin 14 → Int := inputs.2.2.2.2.2.2.1
  let w_49_0 : Fin 14 → Int := inputs.2.2.2.2.2.2.2.1
  let w_0_0 : Int := 0
  let w_1_0 : Int := 0
  let w_2_0 : Bool := decide (w_0_0 = w_1_0)
  let w_6_0 := witness.w_6_0
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_8_0 : Int := 0
  let w_10_0 := witness.w_10_0
  let w_11_0 : Int := 1
  let w_12_0 : Bool := decide (w_10_0 = w_11_0)
  let w_13_0 : Int := if w_12_0 then 1 else 0
  let w_14_0 : Int := (w_7_0 + w_13_0)
  let w_15_0 : Int := 2
  let w_16_0 : Bool := decide (w_14_0 = w_15_0)
  let w_17_0 : Int := if w_16_0 then 1 else 0
  let w_18_0 : Int := 0
  let w_20_0 := witness.w_20_0
  let w_21_0 : Int := 1
  let w_22_0 : Bool := decide (w_20_0 = w_21_0)
  let w_23_0 : Int := if w_22_0 then 1 else 0
  let w_24_0 : Int := (w_17_0 + w_23_0)
  let w_25_0 : Int := 2
  let w_26_0 : Bool := decide (w_24_0 = w_25_0)
  let w_27_0 : Int := if w_26_0 then 1 else 0
  let w_28_0 : Int := 0
  let w_30_0 := witness.w_30_0
  let w_31_0 : Int := 1
  let w_32_0 : Bool := decide (w_30_0 = w_31_0)
  let w_33_0 : Int := if w_32_0 then 1 else 0
  let w_34_0 : Int := (w_27_0 + w_33_0)
  let w_35_0 : Int := 2
  let w_36_0 : Bool := decide (w_34_0 = w_35_0)
  let w_37_0 : Int := if w_36_0 then 1 else 0
  let w_38_0 : Int := 0
  let w_40_0 := witness.w_40_0
  let w_41_0 : Int := 1
  let w_42_0 : Bool := decide (w_40_0 = w_41_0)
  let w_43_0 : Int := if w_42_0 then 1 else 0
  let w_44_0 : Int := (w_37_0 + w_43_0)
  let w_45_0 : Int := 2
  let w_46_0 : Bool := decide (w_44_0 = w_45_0)
  let w_47_0 : Int := if w_46_0 then 1 else 0
  let w_48_0 : Int := 0
  let w_50_0 := witness.w_50_0
  let w_51_0 : Int := 1
  let w_52_0 : Bool := decide (w_50_0 = w_51_0)
  let w_53_0 : Int := if w_52_0 then 1 else 0
  let w_54_0 : Int := (w_47_0 + w_53_0)
  let w_55_0 : Int := 2
  let w_56_0 : Bool := decide (w_54_0 = w_55_0)
  generatedRoot.constraints_0 params w_3_0 w_4_0 w_5_0 w_9_0 w_19_0 w_29_0 w_39_0 w_49_0 w_2_0 w_6_0 w_8_0 w_10_0 w_18_0 w_20_0 w_28_0 w_30_0 w_38_0 w_40_0 w_48_0 w_50_0 w_56_0 outputs

def generatedRoot (params : Params) (inputs : (Fin 17 → Int) × (Fin 17 → Int) × (Fin 17 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Requirement_0
