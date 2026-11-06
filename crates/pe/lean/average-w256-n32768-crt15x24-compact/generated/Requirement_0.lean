import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Requirement_0

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_2.constraints_0 (w_2_0 : Fin 2 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  w_5_0 ≠ 0 ∧
  outputs = w_12_0

def sequential_generatedRoot_2 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 2 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : Int),
    let w_5_0 : Int := 8
    let w_6_0 : Int := (w_4_0 % w_5_0)
    let w_7_0 : Int := 0
    let w_8_0 : Bool := decide (w_6_0 = w_7_0)
    let w_9_0 : Int := if w_8_0 then 1 else 0
    let w_10_0 : Int := (w_1_0 + w_9_0)
    let w_11_0 : Int := 2
    let w_12_0 : Bool := decide (w_10_0 = w_11_0)
    sequential_generatedRoot_2.constraints_0 w_2_0 w_3_0 w_4_0 w_5_0 w_12_0 outputs

abbrev sequential_generatedRoot_6.constraints_0 (w_2_0 : Fin 512 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_10_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  outputs = w_10_0

def sequential_generatedRoot_6 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 512 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := (Int.ofNat i_0)
  ∃ (w_4_0 : Int),
    let w_5_0 : Int := 8
    let w_6_0 : Bool := decide (w_4_0 < w_5_0)
    let w_7_0 : Int := if w_6_0 then 1 else 0
    let w_8_0 : Int := (w_1_0 + w_7_0)
    let w_9_0 : Int := 2
    let w_10_0 : Bool := decide (w_8_0 = w_9_0)
    sequential_generatedRoot_6.constraints_0 w_2_0 w_3_0 w_4_0 w_10_0 outputs

abbrev sequential_generatedRoot_13.constraints_0 (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_12_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  outputs = w_12_0

def sequential_generatedRoot_13 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 512 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 2
    let w_8_0 : Bool := decide (w_6_0 < w_7_0)
    let w_9_0 : Int := if w_8_0 then 1 else 0
    let w_10_0 : Int := (w_1_0 + w_9_0)
    let w_11_0 : Int := 2
    let w_12_0 : Bool := decide (w_10_0 = w_11_0)
    sequential_generatedRoot_13.constraints_0 w_2_0 w_5_0 w_6_0 w_12_0 outputs

abbrev sequential_generatedRoot_21.constraints_0 (w_0_0 : Fin 512 → Int) (w_5_0 : Fin 2 → Int) (w_6_0 : Fin 2 → Int) (w_1_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (w_10_0 : Int) (w_11_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_1_0 w_7_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_7_0 w_8_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_10_0 w_11_0 ∧
  outputs = w_11_0

def sequential_generatedRoot_21 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 512 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_4_0 : Int := inputs.1
  let w_0_0 : Fin 512 → Int := inputs.2.1
  let w_5_0 : Fin 2 → Int := inputs.2.2.1
  let w_6_0 : Fin 2 → Int := inputs.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 : Int := 256
  let w_3_0 : Int := (w_1_0 * w_2_0)
  ∃ (w_7_0 : Int),
    ∃ (w_8_0 : Int),
      let w_9_0 : Int := (w_4_0 + w_8_0)
      let w_10_0 : Int := (w_3_0 + w_9_0)
      ∃ (w_11_0 : Int),
        sequential_generatedRoot_21.constraints_0 w_0_0 w_5_0 w_6_0 w_1_0 w_7_0 w_8_0 w_10_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Bool
  w_6_0 : Bool
  w_13_0 : Bool
  w_21_0 : Int

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_1_0 : Fin 2 → Int) (w_5_0 : Fin 512 → Int) (w_20_0 : Fin 2 → Int) (w_0_0 : Bool) (w_2_0 : Bool) (w_4_0 : Bool) (w_6_0 : Bool) (w_12_0 : Bool) (w_13_0 : Bool) (w_19_0 : Int) (w_21_0 : Int) (w_27_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_2 tape (path ++ [2, i]) params i (current, (w_1_0, ())) next) (Int.toNat (2)) w_0_0 w_2_0 ∧
  0 ≤ (256) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 tape (path ++ [6, i]) params i (current, (w_5_0, ())) next) (Int.toNat (256)) w_4_0 w_6_0 ∧
  0 ≤ (256) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_13 tape (path ++ [13, i]) params i (current, (w_5_0, ())) next) (Int.toNat (256)) w_12_0 w_13_0 ∧
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_21 tape (path ++ [21, i]) params i (current, (w_5_0, (w_1_0, (w_20_0, ())))) next) (Int.toNat (2)) w_19_0 w_21_0 ∧
  outputs = w_27_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 2 → Int) × (Fin 512 → Int) × (Fin 2 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : Fin 2 → Int := inputs.1
  let w_5_0 : Fin 512 → Int := inputs.2.1
  let w_20_0 : Fin 2 → Int := inputs.2.2.1
  let w_0_0 : Bool := true
  let w_2_0 := witness.w_2_0
  let w_3_0 : Int := if w_2_0 then 1 else 0
  let w_4_0 : Bool := true
  let w_6_0 := witness.w_6_0
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_8_0 : Int := (w_3_0 + w_7_0)
  let w_9_0 : Int := 2
  let w_10_0 : Bool := decide (w_8_0 = w_9_0)
  let w_11_0 : Int := if w_10_0 then 1 else 0
  let w_12_0 : Bool := true
  let w_13_0 := witness.w_13_0
  let w_14_0 : Int := if w_13_0 then 1 else 0
  let w_15_0 : Int := (w_11_0 + w_14_0)
  let w_16_0 : Int := 2
  let w_17_0 : Bool := decide (w_15_0 = w_16_0)
  let w_18_0 : Int := if w_17_0 then 1 else 0
  let w_19_0 : Int := 0
  let w_21_0 := witness.w_21_0
  let w_22_0 : Int := 0
  let w_23_0 : Bool := decide (w_21_0 = w_22_0)
  let w_24_0 : Int := if w_23_0 then 1 else 0
  let w_25_0 : Int := (w_18_0 + w_24_0)
  let w_26_0 : Int := 2
  let w_27_0 : Bool := decide (w_25_0 = w_26_0)
  generatedRoot.constraints_0 tape path params w_1_0 w_5_0 w_20_0 w_0_0 w_2_0 w_4_0 w_6_0 w_12_0 w_13_0 w_19_0 w_21_0 w_27_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 2 → Int) × (Fin 512 → Int) × (Fin 2 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Requirement_0
