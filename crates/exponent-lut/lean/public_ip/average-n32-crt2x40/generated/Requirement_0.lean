import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Requirement_0

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_6.constraints_0 (w_2_0 : Fin 5 → Int) (w_9_0 : Fin 5 → Int) (w_18_0 : Fin 5 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_10_0 : Int) (w_19_0 : Int) (w_30_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_3_0 w_10_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_3_0 w_19_0 ∧
  outputs = w_30_0

def sequential_generatedRoot_6 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 5 → Int := inputs.2.1
  let w_9_0 : Fin 5 → Int := inputs.2.2.1
  let w_18_0 : Fin 5 → Int := inputs.2.2.2.1
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
      let w_15_0 : Int := 32
      let w_16_0 : Int := (w_14_0 * w_15_0)
      let w_17_0 : Int := (w_12_0 + w_16_0)
      ∃ (w_19_0 : Int),
        let w_20_0 : Int := 4
        let w_21_0 : Int := (w_19_0 * w_20_0)
        let w_22_0 : Bool := decide (w_17_0 = w_21_0)
        let w_23_0 : Int := if w_22_0 then 1 else 0
        let w_24_0 : Int := (w_8_0 + w_23_0)
        let w_25_0 : Int := 2
        let w_26_0 : Bool := decide (w_24_0 = w_25_0)
        let w_27_0 : Int := if w_26_0 then 1 else 0
        let w_28_0 : Int := (w_1_0 + w_27_0)
        let w_29_0 : Int := 2
        let w_30_0 : Bool := decide (w_28_0 = w_29_0)
        sequential_generatedRoot_6.constraints_0 w_2_0 w_9_0 w_18_0 w_3_0 w_4_0 w_10_0 w_19_0 w_30_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_6_0 : Bool
  w_8_0 : Int
  w_30_0 : Int
  w_32_0 : Int
  w_34_0 : Int

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_3_0 : Fin 5 → Int) (w_4_0 : Fin 5 → Int) (w_5_0 : Fin 5 → Int) (w_2_0 : Bool) (w_6_0 : Bool) (w_8_0 : Int) (w_30_0 : Int) (w_32_0 : Int) (w_34_0 : Int) (w_36_0 : Int) (w_48_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (5) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 tape (path ++ [6, i]) params i (current, (w_3_0, (w_4_0, (w_5_0, ())))) next) (Int.toNat (5)) w_2_0 w_6_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (4) w_8_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_30_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (1) w_32_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (3) w_34_0 ∧
  w_36_0 ≠ 0 ∧
  outputs = w_48_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × Int × Int × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_3_0 : Fin 5 → Int := inputs.1
  let w_4_0 : Fin 5 → Int := inputs.2.1
  let w_5_0 : Fin 5 → Int := inputs.2.2.1
  let w_16_0 : Int := inputs.2.2.2.1
  let w_21_0 : Int := inputs.2.2.2.2.1
  let w_0_0 : Int := 0
  let w_1_0 : Int := 0
  let w_2_0 : Bool := decide (w_0_0 = w_1_0)
  let w_6_0 := witness.w_6_0
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Int := 0
  let w_10_0 : Bool := decide (w_8_0 = w_9_0)
  let w_11_0 : Int := if w_10_0 then 1 else 0
  let w_12_0 : Int := (w_7_0 + w_11_0)
  let w_13_0 : Int := 2
  let w_14_0 : Bool := decide (w_12_0 = w_13_0)
  let w_15_0 : Int := if w_14_0 then 1 else 0
  let w_17_0 : Int := (w_16_0 * w_16_0)
  let w_18_0 : Int := 1
  let w_19_0 : Bool := decide (w_17_0 = w_18_0)
  let w_20_0 : Int := if w_19_0 then 1 else 0
  let w_22_0 : Int := 2
  let w_23_0 : Int := (w_21_0 * w_22_0)
  let w_24_0 : Int := 1
  let w_25_0 : Int := (w_24_0 - w_16_0)
  let w_26_0 : Int := 32
  let w_27_0 : Int := (w_25_0 * w_26_0)
  let w_28_0 : Int := (w_23_0 + w_27_0)
  let w_29_0 : Int := 0
  let w_30_0 := witness.w_30_0
  let w_31_0 : Int := (w_29_0 + w_30_0)
  let w_32_0 := witness.w_32_0
  let w_33_0 : Int := (w_31_0 + w_32_0)
  let w_34_0 := witness.w_34_0
  let w_35_0 : Int := (w_33_0 + w_34_0)
  let w_36_0 : Int := 32
  let w_37_0 : Int := (w_35_0 % w_36_0)
  let w_38_0 : Int := 4
  let w_39_0 : Int := (w_37_0 * w_38_0)
  let w_40_0 : Bool := decide (w_28_0 = w_39_0)
  let w_41_0 : Int := if w_40_0 then 1 else 0
  let w_42_0 : Int := (w_20_0 + w_41_0)
  let w_43_0 : Int := 2
  let w_44_0 : Bool := decide (w_42_0 = w_43_0)
  let w_45_0 : Int := if w_44_0 then 1 else 0
  let w_46_0 : Int := (w_15_0 + w_45_0)
  let w_47_0 : Int := 2
  let w_48_0 : Bool := decide (w_46_0 = w_47_0)
  generatedRoot.constraints_0 tape path params w_3_0 w_4_0 w_5_0 w_2_0 w_6_0 w_8_0 w_30_0 w_32_0 w_34_0 w_36_0 w_48_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 5 → Int) × (Fin 5 → Int) × (Fin 5 → Int) × Int × Int × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Requirement_0
