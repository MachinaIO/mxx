import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Requirement_0

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_6.constraints_0 (w_2_0 : Fin 1025 → Int) (w_13_0 : Fin 1025 → Int) (w_22_0 : Fin 1025 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_14_0 : Int) (w_23_0 : Int) (w_30_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 1025 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 1025 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_3_0 w_14_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 1025 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_3_0 w_23_0 ∧
  outputs = w_30_0

def sequential_generatedRoot_6 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 1025 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 1025 → Int := inputs.2.1
  let w_13_0 : Fin 1025 → Int := inputs.2.2.1
  let w_22_0 : Fin 1025 → Int := inputs.2.2.2.1
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
      let w_19_0 : Int := 65536
      let w_20_0 : Int := (w_18_0 * w_19_0)
      let w_21_0 : Int := (w_16_0 + w_20_0)
      ∃ (w_23_0 : Int),
        let w_24_0 : Int := 1024
        let w_25_0 : Int := (w_23_0 * w_24_0)
        let w_26_0 : Bool := decide (w_21_0 = w_25_0)
        let w_27_0 : Int := if w_26_0 then 1 else 0
        let w_28_0 : Int := (w_12_0 + w_27_0)
        let w_29_0 : Int := 2
        let w_30_0 : Bool := decide (w_28_0 = w_29_0)
        sequential_generatedRoot_6.constraints_0 w_2_0 w_13_0 w_22_0 w_3_0 w_4_0 w_14_0 w_23_0 w_30_0 outputs

abbrev sequential_generatedRoot_10.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_10 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_10.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_20.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_20 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_20.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_30.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_30 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_30.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_40.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_40 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_40.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_50.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_50 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_50.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_60.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_60 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_60.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_70.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_70 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_70.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_80.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_80 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_80.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_90.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_90 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_90.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_100.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_100 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_100.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_110.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_110 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_110.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_120.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_120 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_120.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_130.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_130 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_130.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_140.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_140 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_140.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_150.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_150 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_150.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_160.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_160 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_160.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_170.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_170 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_170.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_180.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_180 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_180.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_190.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_190 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_190.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_200.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_200 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_200.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_210.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_210 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_210.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_220.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_220 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_220.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_230.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_230 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_230.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_240.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_240 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_240.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_250.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_250 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_250.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_260.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_260 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_260.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_270.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_270 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_270.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_280.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_280 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_280.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_290.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_290 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_290.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_300.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_300 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_300.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_310.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_310 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_310.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_320.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_320 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_320.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_330.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_330 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_330.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

abbrev sequential_generatedRoot_340.constraints_0 (w_1_0 : Fin 113 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 113 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  outputs = w_4_0

def sequential_generatedRoot_340 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 113 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 113 → Int := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := (w_0_0 + w_3_0)
    sequential_generatedRoot_340.constraints_0 w_1_0 w_2_0 w_3_0 w_4_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_6_0 : Bool
  w_10_0 : Int
  w_20_0 : Int
  w_30_0 : Int
  w_40_0 : Int
  w_50_0 : Int
  w_60_0 : Int
  w_70_0 : Int
  w_80_0 : Int
  w_90_0 : Int
  w_100_0 : Int
  w_110_0 : Int
  w_120_0 : Int
  w_130_0 : Int
  w_140_0 : Int
  w_150_0 : Int
  w_160_0 : Int
  w_170_0 : Int
  w_180_0 : Int
  w_190_0 : Int
  w_200_0 : Int
  w_210_0 : Int
  w_220_0 : Int
  w_230_0 : Int
  w_240_0 : Int
  w_250_0 : Int
  w_260_0 : Int
  w_270_0 : Int
  w_280_0 : Int
  w_290_0 : Int
  w_300_0 : Int
  w_310_0 : Int
  w_320_0 : Int
  w_330_0 : Int
  w_340_0 : Int

abbrev generatedRoot.constraints_2 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_319_0 : Fin 113 → Int) (w_329_0 : Fin 113 → Int) (w_339_0 : Fin 113 → Int) (w_318_0 : Int) (w_320_0 : Int) (w_328_0 : Int) (w_330_0 : Int) (w_338_0 : Int) (w_340_0 : Int) (w_346_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_320 tape (path ++ [320, i]) params i (current, (w_319_0, ())) next) (Int.toNat (113)) w_318_0 w_320_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_330 tape (path ++ [330, i]) params i (current, (w_329_0, ())) next) (Int.toNat (113)) w_328_0 w_330_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_340 tape (path ++ [340, i]) params i (current, (w_339_0, ())) next) (Int.toNat (113)) w_338_0 w_340_0 ∧
  outputs = w_346_0

abbrev generatedRoot.constraints_1 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_159_0 : Fin 113 → Int) (w_169_0 : Fin 113 → Int) (w_179_0 : Fin 113 → Int) (w_189_0 : Fin 113 → Int) (w_199_0 : Fin 113 → Int) (w_209_0 : Fin 113 → Int) (w_219_0 : Fin 113 → Int) (w_229_0 : Fin 113 → Int) (w_239_0 : Fin 113 → Int) (w_249_0 : Fin 113 → Int) (w_259_0 : Fin 113 → Int) (w_269_0 : Fin 113 → Int) (w_279_0 : Fin 113 → Int) (w_289_0 : Fin 113 → Int) (w_299_0 : Fin 113 → Int) (w_309_0 : Fin 113 → Int) (w_319_0 : Fin 113 → Int) (w_329_0 : Fin 113 → Int) (w_339_0 : Fin 113 → Int) (w_158_0 : Int) (w_160_0 : Int) (w_168_0 : Int) (w_170_0 : Int) (w_178_0 : Int) (w_180_0 : Int) (w_188_0 : Int) (w_190_0 : Int) (w_198_0 : Int) (w_200_0 : Int) (w_208_0 : Int) (w_210_0 : Int) (w_218_0 : Int) (w_220_0 : Int) (w_228_0 : Int) (w_230_0 : Int) (w_238_0 : Int) (w_240_0 : Int) (w_248_0 : Int) (w_250_0 : Int) (w_258_0 : Int) (w_260_0 : Int) (w_268_0 : Int) (w_270_0 : Int) (w_278_0 : Int) (w_280_0 : Int) (w_288_0 : Int) (w_290_0 : Int) (w_298_0 : Int) (w_300_0 : Int) (w_308_0 : Int) (w_310_0 : Int) (w_318_0 : Int) (w_320_0 : Int) (w_328_0 : Int) (w_330_0 : Int) (w_338_0 : Int) (w_340_0 : Int) (w_346_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_160 tape (path ++ [160, i]) params i (current, (w_159_0, ())) next) (Int.toNat (113)) w_158_0 w_160_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_170 tape (path ++ [170, i]) params i (current, (w_169_0, ())) next) (Int.toNat (113)) w_168_0 w_170_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_180 tape (path ++ [180, i]) params i (current, (w_179_0, ())) next) (Int.toNat (113)) w_178_0 w_180_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_190 tape (path ++ [190, i]) params i (current, (w_189_0, ())) next) (Int.toNat (113)) w_188_0 w_190_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_200 tape (path ++ [200, i]) params i (current, (w_199_0, ())) next) (Int.toNat (113)) w_198_0 w_200_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_210 tape (path ++ [210, i]) params i (current, (w_209_0, ())) next) (Int.toNat (113)) w_208_0 w_210_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_220 tape (path ++ [220, i]) params i (current, (w_219_0, ())) next) (Int.toNat (113)) w_218_0 w_220_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_230 tape (path ++ [230, i]) params i (current, (w_229_0, ())) next) (Int.toNat (113)) w_228_0 w_230_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_240 tape (path ++ [240, i]) params i (current, (w_239_0, ())) next) (Int.toNat (113)) w_238_0 w_240_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_250 tape (path ++ [250, i]) params i (current, (w_249_0, ())) next) (Int.toNat (113)) w_248_0 w_250_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_260 tape (path ++ [260, i]) params i (current, (w_259_0, ())) next) (Int.toNat (113)) w_258_0 w_260_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_270 tape (path ++ [270, i]) params i (current, (w_269_0, ())) next) (Int.toNat (113)) w_268_0 w_270_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_280 tape (path ++ [280, i]) params i (current, (w_279_0, ())) next) (Int.toNat (113)) w_278_0 w_280_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_290 tape (path ++ [290, i]) params i (current, (w_289_0, ())) next) (Int.toNat (113)) w_288_0 w_290_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_300 tape (path ++ [300, i]) params i (current, (w_299_0, ())) next) (Int.toNat (113)) w_298_0 w_300_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_310 tape (path ++ [310, i]) params i (current, (w_309_0, ())) next) (Int.toNat (113)) w_308_0 w_310_0 ∧
  generatedRoot.constraints_2 tape path params w_319_0 w_329_0 w_339_0 w_318_0 w_320_0 w_328_0 w_330_0 w_338_0 w_340_0 w_346_0 outputs

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_3_0 : Fin 1025 → Int) (w_4_0 : Fin 1025 → Int) (w_5_0 : Fin 1025 → Int) (w_9_0 : Fin 113 → Int) (w_19_0 : Fin 113 → Int) (w_29_0 : Fin 113 → Int) (w_39_0 : Fin 113 → Int) (w_49_0 : Fin 113 → Int) (w_59_0 : Fin 113 → Int) (w_69_0 : Fin 113 → Int) (w_79_0 : Fin 113 → Int) (w_89_0 : Fin 113 → Int) (w_99_0 : Fin 113 → Int) (w_109_0 : Fin 113 → Int) (w_119_0 : Fin 113 → Int) (w_129_0 : Fin 113 → Int) (w_139_0 : Fin 113 → Int) (w_149_0 : Fin 113 → Int) (w_159_0 : Fin 113 → Int) (w_169_0 : Fin 113 → Int) (w_179_0 : Fin 113 → Int) (w_189_0 : Fin 113 → Int) (w_199_0 : Fin 113 → Int) (w_209_0 : Fin 113 → Int) (w_219_0 : Fin 113 → Int) (w_229_0 : Fin 113 → Int) (w_239_0 : Fin 113 → Int) (w_249_0 : Fin 113 → Int) (w_259_0 : Fin 113 → Int) (w_269_0 : Fin 113 → Int) (w_279_0 : Fin 113 → Int) (w_289_0 : Fin 113 → Int) (w_299_0 : Fin 113 → Int) (w_309_0 : Fin 113 → Int) (w_319_0 : Fin 113 → Int) (w_329_0 : Fin 113 → Int) (w_339_0 : Fin 113 → Int) (w_2_0 : Bool) (w_6_0 : Bool) (w_8_0 : Int) (w_10_0 : Int) (w_18_0 : Int) (w_20_0 : Int) (w_28_0 : Int) (w_30_0 : Int) (w_38_0 : Int) (w_40_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (w_58_0 : Int) (w_60_0 : Int) (w_68_0 : Int) (w_70_0 : Int) (w_78_0 : Int) (w_80_0 : Int) (w_88_0 : Int) (w_90_0 : Int) (w_98_0 : Int) (w_100_0 : Int) (w_108_0 : Int) (w_110_0 : Int) (w_118_0 : Int) (w_120_0 : Int) (w_128_0 : Int) (w_130_0 : Int) (w_138_0 : Int) (w_140_0 : Int) (w_148_0 : Int) (w_150_0 : Int) (w_158_0 : Int) (w_160_0 : Int) (w_168_0 : Int) (w_170_0 : Int) (w_178_0 : Int) (w_180_0 : Int) (w_188_0 : Int) (w_190_0 : Int) (w_198_0 : Int) (w_200_0 : Int) (w_208_0 : Int) (w_210_0 : Int) (w_218_0 : Int) (w_220_0 : Int) (w_228_0 : Int) (w_230_0 : Int) (w_238_0 : Int) (w_240_0 : Int) (w_248_0 : Int) (w_250_0 : Int) (w_258_0 : Int) (w_260_0 : Int) (w_268_0 : Int) (w_270_0 : Int) (w_278_0 : Int) (w_280_0 : Int) (w_288_0 : Int) (w_290_0 : Int) (w_298_0 : Int) (w_300_0 : Int) (w_308_0 : Int) (w_310_0 : Int) (w_318_0 : Int) (w_320_0 : Int) (w_328_0 : Int) (w_330_0 : Int) (w_338_0 : Int) (w_340_0 : Int) (w_346_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (1025) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 tape (path ++ [6, i]) params i (current, (w_3_0, (w_4_0, (w_5_0, ())))) next) (Int.toNat (1025)) w_2_0 w_6_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_10 tape (path ++ [10, i]) params i (current, (w_9_0, ())) next) (Int.toNat (113)) w_8_0 w_10_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_20 tape (path ++ [20, i]) params i (current, (w_19_0, ())) next) (Int.toNat (113)) w_18_0 w_20_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_30 tape (path ++ [30, i]) params i (current, (w_29_0, ())) next) (Int.toNat (113)) w_28_0 w_30_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_40 tape (path ++ [40, i]) params i (current, (w_39_0, ())) next) (Int.toNat (113)) w_38_0 w_40_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_50 tape (path ++ [50, i]) params i (current, (w_49_0, ())) next) (Int.toNat (113)) w_48_0 w_50_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_60 tape (path ++ [60, i]) params i (current, (w_59_0, ())) next) (Int.toNat (113)) w_58_0 w_60_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_70 tape (path ++ [70, i]) params i (current, (w_69_0, ())) next) (Int.toNat (113)) w_68_0 w_70_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_80 tape (path ++ [80, i]) params i (current, (w_79_0, ())) next) (Int.toNat (113)) w_78_0 w_80_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_90 tape (path ++ [90, i]) params i (current, (w_89_0, ())) next) (Int.toNat (113)) w_88_0 w_90_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_100 tape (path ++ [100, i]) params i (current, (w_99_0, ())) next) (Int.toNat (113)) w_98_0 w_100_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_110 tape (path ++ [110, i]) params i (current, (w_109_0, ())) next) (Int.toNat (113)) w_108_0 w_110_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_120 tape (path ++ [120, i]) params i (current, (w_119_0, ())) next) (Int.toNat (113)) w_118_0 w_120_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_130 tape (path ++ [130, i]) params i (current, (w_129_0, ())) next) (Int.toNat (113)) w_128_0 w_130_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_140 tape (path ++ [140, i]) params i (current, (w_139_0, ())) next) (Int.toNat (113)) w_138_0 w_140_0 ∧
  0 ≤ (113) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_150 tape (path ++ [150, i]) params i (current, (w_149_0, ())) next) (Int.toNat (113)) w_148_0 w_150_0 ∧
  generatedRoot.constraints_1 tape path params w_159_0 w_169_0 w_179_0 w_189_0 w_199_0 w_209_0 w_219_0 w_229_0 w_239_0 w_249_0 w_259_0 w_269_0 w_279_0 w_289_0 w_299_0 w_309_0 w_319_0 w_329_0 w_339_0 w_158_0 w_160_0 w_168_0 w_170_0 w_178_0 w_180_0 w_188_0 w_190_0 w_198_0 w_200_0 w_208_0 w_210_0 w_218_0 w_220_0 w_228_0 w_230_0 w_238_0 w_240_0 w_248_0 w_250_0 w_258_0 w_260_0 w_268_0 w_270_0 w_278_0 w_280_0 w_288_0 w_290_0 w_298_0 w_300_0 w_308_0 w_310_0 w_318_0 w_320_0 w_328_0 w_330_0 w_338_0 w_340_0 w_346_0 outputs

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × Unit) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_3_0 : Fin 1025 → Int := inputs.1
  let w_4_0 : Fin 1025 → Int := inputs.2.1
  let w_5_0 : Fin 1025 → Int := inputs.2.2.1
  let w_9_0 : Fin 113 → Int := inputs.2.2.2.1
  let w_19_0 : Fin 113 → Int := inputs.2.2.2.2.1
  let w_29_0 : Fin 113 → Int := inputs.2.2.2.2.2.1
  let w_39_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.1
  let w_49_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.1
  let w_59_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_69_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_89_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_99_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_109_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_119_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_129_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_139_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_149_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_159_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_169_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_179_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_189_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_199_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_209_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_219_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_229_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_239_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_249_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_259_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_269_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_279_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_289_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_299_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_309_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_319_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_329_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_339_0 : Fin 113 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  let w_57_0 : Int := if w_56_0 then 1 else 0
  let w_58_0 : Int := 0
  let w_60_0 := witness.w_60_0
  let w_61_0 : Int := 1
  let w_62_0 : Bool := decide (w_60_0 = w_61_0)
  let w_63_0 : Int := if w_62_0 then 1 else 0
  let w_64_0 : Int := (w_57_0 + w_63_0)
  let w_65_0 : Int := 2
  let w_66_0 : Bool := decide (w_64_0 = w_65_0)
  let w_67_0 : Int := if w_66_0 then 1 else 0
  let w_68_0 : Int := 0
  let w_70_0 := witness.w_70_0
  let w_71_0 : Int := 1
  let w_72_0 : Bool := decide (w_70_0 = w_71_0)
  let w_73_0 : Int := if w_72_0 then 1 else 0
  let w_74_0 : Int := (w_67_0 + w_73_0)
  let w_75_0 : Int := 2
  let w_76_0 : Bool := decide (w_74_0 = w_75_0)
  let w_77_0 : Int := if w_76_0 then 1 else 0
  let w_78_0 : Int := 0
  let w_80_0 := witness.w_80_0
  let w_81_0 : Int := 1
  let w_82_0 : Bool := decide (w_80_0 = w_81_0)
  let w_83_0 : Int := if w_82_0 then 1 else 0
  let w_84_0 : Int := (w_77_0 + w_83_0)
  let w_85_0 : Int := 2
  let w_86_0 : Bool := decide (w_84_0 = w_85_0)
  let w_87_0 : Int := if w_86_0 then 1 else 0
  let w_88_0 : Int := 0
  let w_90_0 := witness.w_90_0
  let w_91_0 : Int := 1
  let w_92_0 : Bool := decide (w_90_0 = w_91_0)
  let w_93_0 : Int := if w_92_0 then 1 else 0
  let w_94_0 : Int := (w_87_0 + w_93_0)
  let w_95_0 : Int := 2
  let w_96_0 : Bool := decide (w_94_0 = w_95_0)
  let w_97_0 : Int := if w_96_0 then 1 else 0
  let w_98_0 : Int := 0
  let w_100_0 := witness.w_100_0
  let w_101_0 : Int := 1
  let w_102_0 : Bool := decide (w_100_0 = w_101_0)
  let w_103_0 : Int := if w_102_0 then 1 else 0
  let w_104_0 : Int := (w_97_0 + w_103_0)
  let w_105_0 : Int := 2
  let w_106_0 : Bool := decide (w_104_0 = w_105_0)
  let w_107_0 : Int := if w_106_0 then 1 else 0
  let w_108_0 : Int := 0
  let w_110_0 := witness.w_110_0
  let w_111_0 : Int := 1
  let w_112_0 : Bool := decide (w_110_0 = w_111_0)
  let w_113_0 : Int := if w_112_0 then 1 else 0
  let w_114_0 : Int := (w_107_0 + w_113_0)
  let w_115_0 : Int := 2
  let w_116_0 : Bool := decide (w_114_0 = w_115_0)
  let w_117_0 : Int := if w_116_0 then 1 else 0
  let w_118_0 : Int := 0
  let w_120_0 := witness.w_120_0
  let w_121_0 : Int := 1
  let w_122_0 : Bool := decide (w_120_0 = w_121_0)
  let w_123_0 : Int := if w_122_0 then 1 else 0
  let w_124_0 : Int := (w_117_0 + w_123_0)
  let w_125_0 : Int := 2
  let w_126_0 : Bool := decide (w_124_0 = w_125_0)
  let w_127_0 : Int := if w_126_0 then 1 else 0
  let w_128_0 : Int := 0
  let w_130_0 := witness.w_130_0
  let w_131_0 : Int := 1
  let w_132_0 : Bool := decide (w_130_0 = w_131_0)
  let w_133_0 : Int := if w_132_0 then 1 else 0
  let w_134_0 : Int := (w_127_0 + w_133_0)
  let w_135_0 : Int := 2
  let w_136_0 : Bool := decide (w_134_0 = w_135_0)
  let w_137_0 : Int := if w_136_0 then 1 else 0
  let w_138_0 : Int := 0
  let w_140_0 := witness.w_140_0
  let w_141_0 : Int := 1
  let w_142_0 : Bool := decide (w_140_0 = w_141_0)
  let w_143_0 : Int := if w_142_0 then 1 else 0
  let w_144_0 : Int := (w_137_0 + w_143_0)
  let w_145_0 : Int := 2
  let w_146_0 : Bool := decide (w_144_0 = w_145_0)
  let w_147_0 : Int := if w_146_0 then 1 else 0
  let w_148_0 : Int := 0
  let w_150_0 := witness.w_150_0
  let w_151_0 : Int := 1
  let w_152_0 : Bool := decide (w_150_0 = w_151_0)
  let w_153_0 : Int := if w_152_0 then 1 else 0
  let w_154_0 : Int := (w_147_0 + w_153_0)
  let w_155_0 : Int := 2
  let w_156_0 : Bool := decide (w_154_0 = w_155_0)
  let w_157_0 : Int := if w_156_0 then 1 else 0
  let w_158_0 : Int := 0
  let w_160_0 := witness.w_160_0
  let w_161_0 : Int := 1
  let w_162_0 : Bool := decide (w_160_0 = w_161_0)
  let w_163_0 : Int := if w_162_0 then 1 else 0
  let w_164_0 : Int := (w_157_0 + w_163_0)
  let w_165_0 : Int := 2
  let w_166_0 : Bool := decide (w_164_0 = w_165_0)
  let w_167_0 : Int := if w_166_0 then 1 else 0
  let w_168_0 : Int := 0
  let w_170_0 := witness.w_170_0
  let w_171_0 : Int := 1
  let w_172_0 : Bool := decide (w_170_0 = w_171_0)
  let w_173_0 : Int := if w_172_0 then 1 else 0
  let w_174_0 : Int := (w_167_0 + w_173_0)
  let w_175_0 : Int := 2
  let w_176_0 : Bool := decide (w_174_0 = w_175_0)
  let w_177_0 : Int := if w_176_0 then 1 else 0
  let w_178_0 : Int := 0
  let w_180_0 := witness.w_180_0
  let w_181_0 : Int := 1
  let w_182_0 : Bool := decide (w_180_0 = w_181_0)
  let w_183_0 : Int := if w_182_0 then 1 else 0
  let w_184_0 : Int := (w_177_0 + w_183_0)
  let w_185_0 : Int := 2
  let w_186_0 : Bool := decide (w_184_0 = w_185_0)
  let w_187_0 : Int := if w_186_0 then 1 else 0
  let w_188_0 : Int := 0
  let w_190_0 := witness.w_190_0
  let w_191_0 : Int := 1
  let w_192_0 : Bool := decide (w_190_0 = w_191_0)
  let w_193_0 : Int := if w_192_0 then 1 else 0
  let w_194_0 : Int := (w_187_0 + w_193_0)
  let w_195_0 : Int := 2
  let w_196_0 : Bool := decide (w_194_0 = w_195_0)
  let w_197_0 : Int := if w_196_0 then 1 else 0
  let w_198_0 : Int := 0
  let w_200_0 := witness.w_200_0
  let w_201_0 : Int := 1
  let w_202_0 : Bool := decide (w_200_0 = w_201_0)
  let w_203_0 : Int := if w_202_0 then 1 else 0
  let w_204_0 : Int := (w_197_0 + w_203_0)
  let w_205_0 : Int := 2
  let w_206_0 : Bool := decide (w_204_0 = w_205_0)
  let w_207_0 : Int := if w_206_0 then 1 else 0
  let w_208_0 : Int := 0
  let w_210_0 := witness.w_210_0
  let w_211_0 : Int := 1
  let w_212_0 : Bool := decide (w_210_0 = w_211_0)
  let w_213_0 : Int := if w_212_0 then 1 else 0
  let w_214_0 : Int := (w_207_0 + w_213_0)
  let w_215_0 : Int := 2
  let w_216_0 : Bool := decide (w_214_0 = w_215_0)
  let w_217_0 : Int := if w_216_0 then 1 else 0
  let w_218_0 : Int := 0
  let w_220_0 := witness.w_220_0
  let w_221_0 : Int := 1
  let w_222_0 : Bool := decide (w_220_0 = w_221_0)
  let w_223_0 : Int := if w_222_0 then 1 else 0
  let w_224_0 : Int := (w_217_0 + w_223_0)
  let w_225_0 : Int := 2
  let w_226_0 : Bool := decide (w_224_0 = w_225_0)
  let w_227_0 : Int := if w_226_0 then 1 else 0
  let w_228_0 : Int := 0
  let w_230_0 := witness.w_230_0
  let w_231_0 : Int := 1
  let w_232_0 : Bool := decide (w_230_0 = w_231_0)
  let w_233_0 : Int := if w_232_0 then 1 else 0
  let w_234_0 : Int := (w_227_0 + w_233_0)
  let w_235_0 : Int := 2
  let w_236_0 : Bool := decide (w_234_0 = w_235_0)
  let w_237_0 : Int := if w_236_0 then 1 else 0
  let w_238_0 : Int := 0
  let w_240_0 := witness.w_240_0
  let w_241_0 : Int := 1
  let w_242_0 : Bool := decide (w_240_0 = w_241_0)
  let w_243_0 : Int := if w_242_0 then 1 else 0
  let w_244_0 : Int := (w_237_0 + w_243_0)
  let w_245_0 : Int := 2
  let w_246_0 : Bool := decide (w_244_0 = w_245_0)
  let w_247_0 : Int := if w_246_0 then 1 else 0
  let w_248_0 : Int := 0
  let w_250_0 := witness.w_250_0
  let w_251_0 : Int := 1
  let w_252_0 : Bool := decide (w_250_0 = w_251_0)
  let w_253_0 : Int := if w_252_0 then 1 else 0
  let w_254_0 : Int := (w_247_0 + w_253_0)
  let w_255_0 : Int := 2
  let w_256_0 : Bool := decide (w_254_0 = w_255_0)
  let w_257_0 : Int := if w_256_0 then 1 else 0
  let w_258_0 : Int := 0
  let w_260_0 := witness.w_260_0
  let w_261_0 : Int := 1
  let w_262_0 : Bool := decide (w_260_0 = w_261_0)
  let w_263_0 : Int := if w_262_0 then 1 else 0
  let w_264_0 : Int := (w_257_0 + w_263_0)
  let w_265_0 : Int := 2
  let w_266_0 : Bool := decide (w_264_0 = w_265_0)
  let w_267_0 : Int := if w_266_0 then 1 else 0
  let w_268_0 : Int := 0
  let w_270_0 := witness.w_270_0
  let w_271_0 : Int := 1
  let w_272_0 : Bool := decide (w_270_0 = w_271_0)
  let w_273_0 : Int := if w_272_0 then 1 else 0
  let w_274_0 : Int := (w_267_0 + w_273_0)
  let w_275_0 : Int := 2
  let w_276_0 : Bool := decide (w_274_0 = w_275_0)
  let w_277_0 : Int := if w_276_0 then 1 else 0
  let w_278_0 : Int := 0
  let w_280_0 := witness.w_280_0
  let w_281_0 : Int := 1
  let w_282_0 : Bool := decide (w_280_0 = w_281_0)
  let w_283_0 : Int := if w_282_0 then 1 else 0
  let w_284_0 : Int := (w_277_0 + w_283_0)
  let w_285_0 : Int := 2
  let w_286_0 : Bool := decide (w_284_0 = w_285_0)
  let w_287_0 : Int := if w_286_0 then 1 else 0
  let w_288_0 : Int := 0
  let w_290_0 := witness.w_290_0
  let w_291_0 : Int := 1
  let w_292_0 : Bool := decide (w_290_0 = w_291_0)
  let w_293_0 : Int := if w_292_0 then 1 else 0
  let w_294_0 : Int := (w_287_0 + w_293_0)
  let w_295_0 : Int := 2
  let w_296_0 : Bool := decide (w_294_0 = w_295_0)
  let w_297_0 : Int := if w_296_0 then 1 else 0
  let w_298_0 : Int := 0
  let w_300_0 := witness.w_300_0
  let w_301_0 : Int := 1
  let w_302_0 : Bool := decide (w_300_0 = w_301_0)
  let w_303_0 : Int := if w_302_0 then 1 else 0
  let w_304_0 : Int := (w_297_0 + w_303_0)
  let w_305_0 : Int := 2
  let w_306_0 : Bool := decide (w_304_0 = w_305_0)
  let w_307_0 : Int := if w_306_0 then 1 else 0
  let w_308_0 : Int := 0
  let w_310_0 := witness.w_310_0
  let w_311_0 : Int := 1
  let w_312_0 : Bool := decide (w_310_0 = w_311_0)
  let w_313_0 : Int := if w_312_0 then 1 else 0
  let w_314_0 : Int := (w_307_0 + w_313_0)
  let w_315_0 : Int := 2
  let w_316_0 : Bool := decide (w_314_0 = w_315_0)
  let w_317_0 : Int := if w_316_0 then 1 else 0
  let w_318_0 : Int := 0
  let w_320_0 := witness.w_320_0
  let w_321_0 : Int := 1
  let w_322_0 : Bool := decide (w_320_0 = w_321_0)
  let w_323_0 : Int := if w_322_0 then 1 else 0
  let w_324_0 : Int := (w_317_0 + w_323_0)
  let w_325_0 : Int := 2
  let w_326_0 : Bool := decide (w_324_0 = w_325_0)
  let w_327_0 : Int := if w_326_0 then 1 else 0
  let w_328_0 : Int := 0
  let w_330_0 := witness.w_330_0
  let w_331_0 : Int := 1
  let w_332_0 : Bool := decide (w_330_0 = w_331_0)
  let w_333_0 : Int := if w_332_0 then 1 else 0
  let w_334_0 : Int := (w_327_0 + w_333_0)
  let w_335_0 : Int := 2
  let w_336_0 : Bool := decide (w_334_0 = w_335_0)
  let w_337_0 : Int := if w_336_0 then 1 else 0
  let w_338_0 : Int := 0
  let w_340_0 := witness.w_340_0
  let w_341_0 : Int := 1
  let w_342_0 : Bool := decide (w_340_0 = w_341_0)
  let w_343_0 : Int := if w_342_0 then 1 else 0
  let w_344_0 : Int := (w_337_0 + w_343_0)
  let w_345_0 : Int := 2
  let w_346_0 : Bool := decide (w_344_0 = w_345_0)
  generatedRoot.constraints_0 tape path params w_3_0 w_4_0 w_5_0 w_9_0 w_19_0 w_29_0 w_39_0 w_49_0 w_59_0 w_69_0 w_79_0 w_89_0 w_99_0 w_109_0 w_119_0 w_129_0 w_139_0 w_149_0 w_159_0 w_169_0 w_179_0 w_189_0 w_199_0 w_209_0 w_219_0 w_229_0 w_239_0 w_249_0 w_259_0 w_269_0 w_279_0 w_289_0 w_299_0 w_309_0 w_319_0 w_329_0 w_339_0 w_2_0 w_6_0 w_8_0 w_10_0 w_18_0 w_20_0 w_28_0 w_30_0 w_38_0 w_40_0 w_48_0 w_50_0 w_58_0 w_60_0 w_68_0 w_70_0 w_78_0 w_80_0 w_88_0 w_90_0 w_98_0 w_100_0 w_108_0 w_110_0 w_118_0 w_120_0 w_128_0 w_130_0 w_138_0 w_140_0 w_148_0 w_150_0 w_158_0 w_160_0 w_168_0 w_170_0 w_178_0 w_180_0 w_188_0 w_190_0 w_198_0 w_200_0 w_208_0 w_210_0 w_218_0 w_220_0 w_228_0 w_230_0 w_238_0 w_240_0 w_248_0 w_250_0 w_258_0 w_260_0 w_268_0 w_270_0 w_278_0 w_280_0 w_288_0 w_290_0 w_298_0 w_300_0 w_308_0 w_310_0 w_318_0 w_320_0 w_328_0 w_330_0 w_338_0 w_340_0 w_346_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × (Fin 113 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Requirement_0
