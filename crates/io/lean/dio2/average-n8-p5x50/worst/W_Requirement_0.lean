import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Requirement_0

structure Params where
  unit : Unit

abbrev sequential_sequential_generatedRoot_298_5.constraints_0 (w_1_0 : Fin 72000 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_298_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 72000 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_298_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_298.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_298.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 72000 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_298_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_298.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 72000 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_298.Witness) : Prop :=
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
  sequential_generatedRoot_298.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_298 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 72000 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_298.Witness,
    sequential_generatedRoot_298.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_300_5.constraints_0 (w_1_0 : Fin 72000 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_300_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 72000 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_300_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_300.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_300.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 72000 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_300_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_300.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 72000 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_300.Witness) : Prop :=
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
  sequential_generatedRoot_300.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_300 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 72000 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_300.Witness,
    sequential_generatedRoot_300.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_302_5.constraints_0 (w_1_0 : Fin 16 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_302_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 16 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_302_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_302.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_302.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 16 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_302_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_302.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_302.Witness) : Prop :=
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
  sequential_generatedRoot_302.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_302 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_302.Witness,
    sequential_generatedRoot_302.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_304_5.constraints_0 (w_1_0 : Fin 16 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_304_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 16 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_304_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_304.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_304.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 16 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_304_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_304.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_304.Witness) : Prop :=
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
  sequential_generatedRoot_304.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_304 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_304.Witness,
    sequential_generatedRoot_304.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_306_5.constraints_0 (w_1_0 : Fin 14400 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_306_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 14400 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_306_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_306.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_306.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 14400 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_306_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_306.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 14400 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_306.Witness) : Prop :=
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
  sequential_generatedRoot_306.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_306 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 14400 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_306.Witness,
    sequential_generatedRoot_306.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_308_5.constraints_0 (w_1_0 : Fin 14400 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_308_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 14400 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_308_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_308.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_308.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 14400 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_308_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_308.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 14400 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_308.Witness) : Prop :=
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
  sequential_generatedRoot_308.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_308 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 14400 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_308.Witness,
    sequential_generatedRoot_308.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_310_5.constraints_0 (w_1_0 : Fin 16 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_310_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 16 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_310_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_310.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_310.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 16 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_310_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_310.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_310.Witness) : Prop :=
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
  sequential_generatedRoot_310.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_310 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_310.Witness,
    sequential_generatedRoot_310.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_312_5.constraints_0 (w_1_0 : Fin 16 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_312_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 16 → Int) × Int × Unit) (outputs : Int) : Prop :=
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
    sequential_sequential_generatedRoot_312_5.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_312.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_312.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 16 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_312_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (2)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_312.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_312.Witness) : Prop :=
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
  sequential_generatedRoot_312.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_312 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_312.Witness,
    sequential_generatedRoot_312.body params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Int
  w_9_0 : Int
  w_18_0 : Int
  w_33_0 : Int
  w_38_0 : Int
  w_62_0 : Int
  w_68_0 : Int
  w_76_0 : Int
  w_91_0 : Int
  w_96_0 : Int
  w_104_0 : Int
  w_119_0 : Int
  w_124_0 : Int
  w_132_0 : Int
  w_147_0 : Int
  w_152_0 : Int
  w_160_0 : Int
  w_176_0 : Int
  w_182_0 : Int
  w_191_0 : Int
  w_207_0 : Int
  w_213_0 : Int
  w_240_0 : Int
  w_245_0 : Int
  w_269_0 : Int
  w_275_0 : Int
  w_283_0 : Int
  w_298_0 : Bool
  w_300_0 : Bool
  w_302_0 : Bool
  w_304_0 : Bool
  w_306_0 : Bool
  w_308_0 : Bool
  w_310_0 : Bool
  w_312_0 : Bool

abbrev generatedRoot.constraints_1 (params : Params) (w_301_0 : Fin 16 → Int) (w_303_0 : Fin 16 → Int) (w_305_0 : Fin 14400 → Int) (w_307_0 : Fin 14400 → Int) (w_309_0 : Fin 16 → Int) (w_311_0 : Fin 16 → Int) (w_300_0 : Bool) (w_302_0 : Bool) (w_304_0 : Bool) (w_306_0 : Bool) (w_308_0 : Bool) (w_310_0 : Bool) (w_312_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_302 params i (current, (w_301_0, ())) next) (Int.toNat (8)) w_300_0 w_302_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_304 params i (current, (w_303_0, ())) next) (Int.toNat (8)) w_302_0 w_304_0 ∧
  0 ≤ (7200) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_306 params i (current, (w_305_0, ())) next) (Int.toNat (7200)) w_304_0 w_306_0 ∧
  0 ≤ (7200) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_308 params i (current, (w_307_0, ())) next) (Int.toNat (7200)) w_306_0 w_308_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_310 params i (current, (w_309_0, ())) next) (Int.toNat (8)) w_308_0 w_310_0 ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_312 params i (current, (w_311_0, ())) next) (Int.toNat (8)) w_310_0 w_312_0 ∧
  outputs = w_312_0

abbrev generatedRoot.constraints_0 (params : Params) (w_2_0 : Fin 2 → Int) (w_8_0 : Fin 2 → Int) (w_17_0 : Fin 5 → Int) (w_61_0 : Fin 4 → Int) (w_67_0 : Fin 4 → Int) (w_175_0 : Fin 1 → Int) (w_181_0 : Fin 1 → Int) (w_190_0 : Fin 2 → Int) (w_206_0 : Fin 2 → Int) (w_212_0 : Fin 2 → Int) (w_268_0 : Fin 1 → Int) (w_274_0 : Fin 1 → Int) (w_297_0 : Fin 72000 → Int) (w_299_0 : Fin 72000 → Int) (w_3_0 : Int) (w_9_0 : Int) (w_18_0 : Int) (w_33_0 : Int) (w_38_0 : Int) (w_62_0 : Int) (w_68_0 : Int) (w_76_0 : Int) (w_91_0 : Int) (w_96_0 : Int) (w_104_0 : Int) (w_119_0 : Int) (w_124_0 : Int) (w_132_0 : Int) (w_147_0 : Int) (w_152_0 : Int) (w_160_0 : Int) (w_176_0 : Int) (w_182_0 : Int) (w_191_0 : Int) (w_207_0 : Int) (w_213_0 : Int) (w_224_0 : Int) (w_240_0 : Int) (w_245_0 : Int) (w_269_0 : Int) (w_275_0 : Int) (w_283_0 : Int) (w_296_0 : Bool) (w_298_0 : Bool) (w_300_0 : Bool) : Prop :=
  MxxRuntime.familyGetStatic w_2_0 (0) w_3_0 ∧
  MxxRuntime.familyGetStatic w_8_0 (0) w_9_0 ∧
  MxxRuntime.familyGetStatic w_17_0 (0) w_18_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (1) w_33_0 ∧
  MxxRuntime.familyGetStatic w_8_0 (1) w_38_0 ∧
  MxxRuntime.familyGetStatic w_61_0 (0) w_62_0 ∧
  MxxRuntime.familyGetStatic w_67_0 (0) w_68_0 ∧
  MxxRuntime.familyGetStatic w_17_0 (1) w_76_0 ∧
  MxxRuntime.familyGetStatic w_61_0 (1) w_91_0 ∧
  MxxRuntime.familyGetStatic w_67_0 (1) w_96_0 ∧
  MxxRuntime.familyGetStatic w_17_0 (2) w_104_0 ∧
  MxxRuntime.familyGetStatic w_61_0 (2) w_119_0 ∧
  MxxRuntime.familyGetStatic w_67_0 (2) w_124_0 ∧
  MxxRuntime.familyGetStatic w_17_0 (3) w_132_0 ∧
  MxxRuntime.familyGetStatic w_61_0 (3) w_147_0 ∧
  MxxRuntime.familyGetStatic w_67_0 (3) w_152_0 ∧
  MxxRuntime.familyGetStatic w_17_0 (4) w_160_0 ∧
  MxxRuntime.familyGetStatic w_175_0 (0) w_176_0 ∧
  MxxRuntime.familyGetStatic w_181_0 (0) w_182_0 ∧
  MxxRuntime.familyGetStatic w_190_0 (0) w_191_0 ∧
  MxxRuntime.familyGetStatic w_206_0 (0) w_207_0 ∧
  MxxRuntime.familyGetStatic w_212_0 (0) w_213_0 ∧
  w_224_0 ≠ 0 ∧
  MxxRuntime.familyGetStatic w_206_0 (1) w_240_0 ∧
  MxxRuntime.familyGetStatic w_212_0 (1) w_245_0 ∧
  MxxRuntime.familyGetStatic w_268_0 (0) w_269_0 ∧
  MxxRuntime.familyGetStatic w_274_0 (0) w_275_0 ∧
  MxxRuntime.familyGetStatic w_190_0 (1) w_283_0 ∧
  0 ≤ (36000) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_298 params i (current, (w_297_0, ())) next) (Int.toNat (36000)) w_296_0 w_298_0 ∧
  0 ≤ (36000) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_300 params i (current, (w_299_0, ())) next) (Int.toNat (36000)) w_298_0 w_300_0

abbrev generatedRoot.body (params : Params) (inputs : (Fin 2 → Int) × (Fin 2 → Int) × (Fin 5 → Int) × (Fin 4 → Int) × (Fin 4 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_2_0 : Fin 2 → Int := inputs.1
  let w_8_0 : Fin 2 → Int := inputs.2.1
  let w_17_0 : Fin 5 → Int := inputs.2.2.1
  let w_61_0 : Fin 4 → Int := inputs.2.2.2.1
  let w_67_0 : Fin 4 → Int := inputs.2.2.2.2.1
  let w_175_0 : Fin 1 → Int := inputs.2.2.2.2.2.1
  let w_181_0 : Fin 1 → Int := inputs.2.2.2.2.2.2.1
  let w_190_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.1
  let w_206_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_212_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_268_0 : Fin 1 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_274_0 : Fin 1 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_297_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_299_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_301_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_303_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_305_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_307_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_309_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_311_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : Bool := true
  let w_1_0 : Int := if w_0_0 then 1 else 0
  let w_3_0 := witness.w_3_0
  let w_4_0 : Int := (w_3_0 * w_3_0)
  let w_5_0 : Int := 1
  let w_6_0 : Bool := decide (w_4_0 = w_5_0)
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_9_0 := witness.w_9_0
  let w_10_0 : Int := 2
  let w_11_0 : Int := (w_9_0 * w_10_0)
  let w_12_0 : Int := 1
  let w_13_0 : Int := (w_12_0 - w_3_0)
  let w_14_0 : Int := 8
  let w_15_0 : Int := (w_13_0 * w_14_0)
  let w_16_0 : Int := (w_11_0 + w_15_0)
  let w_18_0 := witness.w_18_0
  let w_19_0 : Int := 1
  let w_20_0 : Int := (w_18_0 * w_19_0)
  let w_21_0 : Int := 2
  let w_22_0 : Int := (w_20_0 * w_21_0)
  let w_23_0 : Bool := decide (w_16_0 = w_22_0)
  let w_24_0 : Int := if w_23_0 then 1 else 0
  let w_25_0 : Int := (w_7_0 + w_24_0)
  let w_26_0 : Int := 2
  let w_27_0 : Bool := decide (w_25_0 = w_26_0)
  let w_28_0 : Int := if w_27_0 then 1 else 0
  let w_29_0 : Int := (w_1_0 + w_28_0)
  let w_30_0 : Int := 2
  let w_31_0 : Bool := decide (w_29_0 = w_30_0)
  let w_32_0 : Int := if w_31_0 then 1 else 0
  let w_33_0 := witness.w_33_0
  let w_34_0 : Int := (w_33_0 * w_33_0)
  let w_35_0 : Int := 1
  let w_36_0 : Bool := decide (w_34_0 = w_35_0)
  let w_37_0 : Int := if w_36_0 then 1 else 0
  let w_38_0 := witness.w_38_0
  let w_39_0 : Int := 2
  let w_40_0 : Int := (w_38_0 * w_39_0)
  let w_41_0 : Int := 1
  let w_42_0 : Int := (w_41_0 - w_33_0)
  let w_43_0 : Int := 8
  let w_44_0 : Int := (w_42_0 * w_43_0)
  let w_45_0 : Int := (w_40_0 + w_44_0)
  let w_46_0 : Int := 0
  let w_47_0 : Int := 1
  let w_48_0 : Int := (w_46_0 * w_47_0)
  let w_49_0 : Int := 2
  let w_50_0 : Int := (w_48_0 * w_49_0)
  let w_51_0 : Bool := decide (w_45_0 = w_50_0)
  let w_52_0 : Int := if w_51_0 then 1 else 0
  let w_53_0 : Int := (w_37_0 + w_52_0)
  let w_54_0 : Int := 2
  let w_55_0 : Bool := decide (w_53_0 = w_54_0)
  let w_56_0 : Int := if w_55_0 then 1 else 0
  let w_57_0 : Int := (w_32_0 + w_56_0)
  let w_58_0 : Int := 2
  let w_59_0 : Bool := decide (w_57_0 = w_58_0)
  let w_60_0 : Int := if w_59_0 then 1 else 0
  let w_62_0 := witness.w_62_0
  let w_63_0 : Int := (w_62_0 * w_62_0)
  let w_64_0 : Int := 1
  let w_65_0 : Bool := decide (w_63_0 = w_64_0)
  let w_66_0 : Int := if w_65_0 then 1 else 0
  let w_68_0 := witness.w_68_0
  let w_69_0 : Int := 2
  let w_70_0 : Int := (w_68_0 * w_69_0)
  let w_71_0 : Int := 1
  let w_72_0 : Int := (w_71_0 - w_62_0)
  let w_73_0 : Int := 8
  let w_74_0 : Int := (w_72_0 * w_73_0)
  let w_75_0 : Int := (w_70_0 + w_74_0)
  let w_76_0 := witness.w_76_0
  let w_77_0 : Int := 1
  let w_78_0 : Int := (w_76_0 * w_77_0)
  let w_79_0 : Int := 2
  let w_80_0 : Int := (w_78_0 * w_79_0)
  let w_81_0 : Bool := decide (w_75_0 = w_80_0)
  let w_82_0 : Int := if w_81_0 then 1 else 0
  let w_83_0 : Int := (w_66_0 + w_82_0)
  let w_84_0 : Int := 2
  let w_85_0 : Bool := decide (w_83_0 = w_84_0)
  let w_86_0 : Int := if w_85_0 then 1 else 0
  let w_87_0 : Int := (w_60_0 + w_86_0)
  let w_88_0 : Int := 2
  let w_89_0 : Bool := decide (w_87_0 = w_88_0)
  let w_90_0 : Int := if w_89_0 then 1 else 0
  let w_91_0 := witness.w_91_0
  let w_92_0 : Int := (w_91_0 * w_91_0)
  let w_93_0 : Int := 1
  let w_94_0 : Bool := decide (w_92_0 = w_93_0)
  let w_95_0 : Int := if w_94_0 then 1 else 0
  let w_96_0 := witness.w_96_0
  let w_97_0 : Int := 2
  let w_98_0 : Int := (w_96_0 * w_97_0)
  let w_99_0 : Int := 1
  let w_100_0 : Int := (w_99_0 - w_91_0)
  let w_101_0 : Int := 8
  let w_102_0 : Int := (w_100_0 * w_101_0)
  let w_103_0 : Int := (w_98_0 + w_102_0)
  let w_104_0 := witness.w_104_0
  let w_105_0 : Int := 1
  let w_106_0 : Int := (w_104_0 * w_105_0)
  let w_107_0 : Int := 2
  let w_108_0 : Int := (w_106_0 * w_107_0)
  let w_109_0 : Bool := decide (w_103_0 = w_108_0)
  let w_110_0 : Int := if w_109_0 then 1 else 0
  let w_111_0 : Int := (w_95_0 + w_110_0)
  let w_112_0 : Int := 2
  let w_113_0 : Bool := decide (w_111_0 = w_112_0)
  let w_114_0 : Int := if w_113_0 then 1 else 0
  let w_115_0 : Int := (w_90_0 + w_114_0)
  let w_116_0 : Int := 2
  let w_117_0 : Bool := decide (w_115_0 = w_116_0)
  let w_118_0 : Int := if w_117_0 then 1 else 0
  let w_119_0 := witness.w_119_0
  let w_120_0 : Int := (w_119_0 * w_119_0)
  let w_121_0 : Int := 1
  let w_122_0 : Bool := decide (w_120_0 = w_121_0)
  let w_123_0 : Int := if w_122_0 then 1 else 0
  let w_124_0 := witness.w_124_0
  let w_125_0 : Int := 2
  let w_126_0 : Int := (w_124_0 * w_125_0)
  let w_127_0 : Int := 1
  let w_128_0 : Int := (w_127_0 - w_119_0)
  let w_129_0 : Int := 8
  let w_130_0 : Int := (w_128_0 * w_129_0)
  let w_131_0 : Int := (w_126_0 + w_130_0)
  let w_132_0 := witness.w_132_0
  let w_133_0 : Int := 1
  let w_134_0 : Int := (w_132_0 * w_133_0)
  let w_135_0 : Int := 2
  let w_136_0 : Int := (w_134_0 * w_135_0)
  let w_137_0 : Bool := decide (w_131_0 = w_136_0)
  let w_138_0 : Int := if w_137_0 then 1 else 0
  let w_139_0 : Int := (w_123_0 + w_138_0)
  let w_140_0 : Int := 2
  let w_141_0 : Bool := decide (w_139_0 = w_140_0)
  let w_142_0 : Int := if w_141_0 then 1 else 0
  let w_143_0 : Int := (w_118_0 + w_142_0)
  let w_144_0 : Int := 2
  let w_145_0 : Bool := decide (w_143_0 = w_144_0)
  let w_146_0 : Int := if w_145_0 then 1 else 0
  let w_147_0 := witness.w_147_0
  let w_148_0 : Int := (w_147_0 * w_147_0)
  let w_149_0 : Int := 1
  let w_150_0 : Bool := decide (w_148_0 = w_149_0)
  let w_151_0 : Int := if w_150_0 then 1 else 0
  let w_152_0 := witness.w_152_0
  let w_153_0 : Int := 2
  let w_154_0 : Int := (w_152_0 * w_153_0)
  let w_155_0 : Int := 1
  let w_156_0 : Int := (w_155_0 - w_147_0)
  let w_157_0 : Int := 8
  let w_158_0 : Int := (w_156_0 * w_157_0)
  let w_159_0 : Int := (w_154_0 + w_158_0)
  let w_160_0 := witness.w_160_0
  let w_161_0 : Int := 1
  let w_162_0 : Int := (w_160_0 * w_161_0)
  let w_163_0 : Int := 2
  let w_164_0 : Int := (w_162_0 * w_163_0)
  let w_165_0 : Bool := decide (w_159_0 = w_164_0)
  let w_166_0 : Int := if w_165_0 then 1 else 0
  let w_167_0 : Int := (w_151_0 + w_166_0)
  let w_168_0 : Int := 2
  let w_169_0 : Bool := decide (w_167_0 = w_168_0)
  let w_170_0 : Int := if w_169_0 then 1 else 0
  let w_171_0 : Int := (w_146_0 + w_170_0)
  let w_172_0 : Int := 2
  let w_173_0 : Bool := decide (w_171_0 = w_172_0)
  let w_174_0 : Int := if w_173_0 then 1 else 0
  let w_176_0 := witness.w_176_0
  let w_177_0 : Int := (w_176_0 * w_176_0)
  let w_178_0 : Int := 1
  let w_179_0 : Bool := decide (w_177_0 = w_178_0)
  let w_180_0 : Int := if w_179_0 then 1 else 0
  let w_182_0 := witness.w_182_0
  let w_183_0 : Int := 2
  let w_184_0 : Int := (w_182_0 * w_183_0)
  let w_185_0 : Int := 1
  let w_186_0 : Int := (w_185_0 - w_176_0)
  let w_187_0 : Int := 8
  let w_188_0 : Int := (w_186_0 * w_187_0)
  let w_189_0 : Int := (w_184_0 + w_188_0)
  let w_191_0 := witness.w_191_0
  let w_192_0 : Int := 2
  let w_193_0 : Int := (w_191_0 * w_192_0)
  let w_194_0 : Int := 2
  let w_195_0 : Int := (w_193_0 * w_194_0)
  let w_196_0 : Bool := decide (w_189_0 = w_195_0)
  let w_197_0 : Int := if w_196_0 then 1 else 0
  let w_198_0 : Int := (w_180_0 + w_197_0)
  let w_199_0 : Int := 2
  let w_200_0 : Bool := decide (w_198_0 = w_199_0)
  let w_201_0 : Int := if w_200_0 then 1 else 0
  let w_202_0 : Int := (w_174_0 + w_201_0)
  let w_203_0 : Int := 2
  let w_204_0 : Bool := decide (w_202_0 = w_203_0)
  let w_205_0 : Int := if w_204_0 then 1 else 0
  let w_207_0 := witness.w_207_0
  let w_208_0 : Int := (w_207_0 * w_207_0)
  let w_209_0 : Int := 1
  let w_210_0 : Bool := decide (w_208_0 = w_209_0)
  let w_211_0 : Int := if w_210_0 then 1 else 0
  let w_213_0 := witness.w_213_0
  let w_214_0 : Int := 2
  let w_215_0 : Int := (w_213_0 * w_214_0)
  let w_216_0 : Int := 1
  let w_217_0 : Int := (w_216_0 - w_207_0)
  let w_218_0 : Int := 8
  let w_219_0 : Int := (w_217_0 * w_218_0)
  let w_220_0 : Int := (w_215_0 + w_219_0)
  let w_221_0 : Int := 0
  let w_222_0 : Int := (w_221_0 + w_104_0)
  let w_223_0 : Int := (w_222_0 + w_160_0)
  let w_224_0 : Int := 16
  let w_225_0 : Int := (w_223_0 % w_224_0)
  let w_226_0 : Int := 1
  let w_227_0 : Int := (w_225_0 * w_226_0)
  let w_228_0 : Int := 2
  let w_229_0 : Int := (w_227_0 * w_228_0)
  let w_230_0 : Bool := decide (w_220_0 = w_229_0)
  let w_231_0 : Int := if w_230_0 then 1 else 0
  let w_232_0 : Int := (w_211_0 + w_231_0)
  let w_233_0 : Int := 2
  let w_234_0 : Bool := decide (w_232_0 = w_233_0)
  let w_235_0 : Int := if w_234_0 then 1 else 0
  let w_236_0 : Int := (w_205_0 + w_235_0)
  let w_237_0 : Int := 2
  let w_238_0 : Bool := decide (w_236_0 = w_237_0)
  let w_239_0 : Int := if w_238_0 then 1 else 0
  let w_240_0 := witness.w_240_0
  let w_241_0 : Int := (w_240_0 * w_240_0)
  let w_242_0 : Int := 1
  let w_243_0 : Bool := decide (w_241_0 = w_242_0)
  let w_244_0 : Int := if w_243_0 then 1 else 0
  let w_245_0 := witness.w_245_0
  let w_246_0 : Int := 2
  let w_247_0 : Int := (w_245_0 * w_246_0)
  let w_248_0 : Int := 1
  let w_249_0 : Int := (w_248_0 - w_240_0)
  let w_250_0 : Int := 8
  let w_251_0 : Int := (w_249_0 * w_250_0)
  let w_252_0 : Int := (w_247_0 + w_251_0)
  let w_253_0 : Int := 0
  let w_254_0 : Int := 1
  let w_255_0 : Int := (w_253_0 * w_254_0)
  let w_256_0 : Int := 2
  let w_257_0 : Int := (w_255_0 * w_256_0)
  let w_258_0 : Bool := decide (w_252_0 = w_257_0)
  let w_259_0 : Int := if w_258_0 then 1 else 0
  let w_260_0 : Int := (w_244_0 + w_259_0)
  let w_261_0 : Int := 2
  let w_262_0 : Bool := decide (w_260_0 = w_261_0)
  let w_263_0 : Int := if w_262_0 then 1 else 0
  let w_264_0 : Int := (w_239_0 + w_263_0)
  let w_265_0 : Int := 2
  let w_266_0 : Bool := decide (w_264_0 = w_265_0)
  let w_267_0 : Int := if w_266_0 then 1 else 0
  let w_269_0 := witness.w_269_0
  let w_270_0 : Int := (w_269_0 * w_269_0)
  let w_271_0 : Int := 1
  let w_272_0 : Bool := decide (w_270_0 = w_271_0)
  let w_273_0 : Int := if w_272_0 then 1 else 0
  let w_275_0 := witness.w_275_0
  let w_276_0 : Int := 2
  let w_277_0 : Int := (w_275_0 * w_276_0)
  let w_278_0 : Int := 1
  let w_279_0 : Int := (w_278_0 - w_269_0)
  let w_280_0 : Int := 8
  let w_281_0 : Int := (w_279_0 * w_280_0)
  let w_282_0 : Int := (w_277_0 + w_281_0)
  let w_283_0 := witness.w_283_0
  let w_284_0 : Int := 2
  let w_285_0 : Int := (w_283_0 * w_284_0)
  let w_286_0 : Int := 2
  let w_287_0 : Int := (w_285_0 * w_286_0)
  let w_288_0 : Bool := decide (w_282_0 = w_287_0)
  let w_289_0 : Int := if w_288_0 then 1 else 0
  let w_290_0 : Int := (w_273_0 + w_289_0)
  let w_291_0 : Int := 2
  let w_292_0 : Bool := decide (w_290_0 = w_291_0)
  let w_293_0 : Int := if w_292_0 then 1 else 0
  let w_294_0 : Int := (w_267_0 + w_293_0)
  let w_295_0 : Int := 2
  let w_296_0 : Bool := decide (w_294_0 = w_295_0)
  let w_298_0 := witness.w_298_0
  let w_300_0 := witness.w_300_0
  let w_302_0 := witness.w_302_0
  let w_304_0 := witness.w_304_0
  let w_306_0 := witness.w_306_0
  let w_308_0 := witness.w_308_0
  let w_310_0 := witness.w_310_0
  let w_312_0 := witness.w_312_0
  generatedRoot.constraints_0 params w_2_0 w_8_0 w_17_0 w_61_0 w_67_0 w_175_0 w_181_0 w_190_0 w_206_0 w_212_0 w_268_0 w_274_0 w_297_0 w_299_0 w_3_0 w_9_0 w_18_0 w_33_0 w_38_0 w_62_0 w_68_0 w_76_0 w_91_0 w_96_0 w_104_0 w_119_0 w_124_0 w_132_0 w_147_0 w_152_0 w_160_0 w_176_0 w_182_0 w_191_0 w_207_0 w_213_0 w_224_0 w_240_0 w_245_0 w_269_0 w_275_0 w_283_0 w_296_0 w_298_0 w_300_0 ∧
    generatedRoot.constraints_1 params w_301_0 w_303_0 w_305_0 w_307_0 w_309_0 w_311_0 w_300_0 w_302_0 w_304_0 w_306_0 w_308_0 w_310_0 w_312_0 outputs

def generatedRoot (params : Params) (inputs : (Fin 2 → Int) × (Fin 2 → Int) × (Fin 5 → Int) × (Fin 4 → Int) × (Fin 4 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 2 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end W_Requirement_0
