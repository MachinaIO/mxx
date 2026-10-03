import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Ideal

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_4.constraints_0 (w_1_0 : Fin 14 → Int) (w_4_0 : Fin 17 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_28_0 : Int) (w_31_0 : Int) (w_34_0 : Int) (w_37_0 : Int) (w_40_0 : Int) (w_43_0 : Int) (w_46_0 : Int) (w_47_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  14 = 14 ∧
  MxxRuntime.select w_2_0 [w_7_0, w_10_0, w_13_0, w_16_0, w_19_0, w_22_0, w_25_0, w_28_0, w_31_0, w_34_0, w_37_0, w_40_0, w_43_0, w_46_0] w_47_0 ∧
  0 ≤ w_47_0 ∧
  w_47_0 < 17 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_47_0 w_48_0 ∧
  outputs = w_50_0

def sequential_generatedRoot_4 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × (Fin 17 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_4_0 : Fin 17 → Int := inputs.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_5_0 : Int := 4
    let w_6_0 : Int := 0
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_8_0 : Int := 5
    let w_9_0 : Int := 0
    let w_10_0 : Int := (w_8_0 + w_9_0)
    let w_11_0 : Int := 10
    let w_12_0 : Int := 0
    let w_13_0 : Int := (w_11_0 + w_12_0)
    let w_14_0 : Int := 11
    let w_15_0 : Int := 0
    let w_16_0 : Int := (w_14_0 + w_15_0)
    let w_17_0 : Int := 12
    let w_18_0 : Int := 0
    let w_19_0 : Int := (w_17_0 + w_18_0)
    let w_20_0 : Int := 14
    let w_21_0 : Int := 0
    let w_22_0 : Int := (w_20_0 + w_21_0)
    let w_23_0 : Int := 15
    let w_24_0 : Int := 0
    let w_25_0 : Int := (w_23_0 + w_24_0)
    let w_26_0 : Int := 16
    let w_27_0 : Int := 0
    let w_28_0 : Int := (w_26_0 + w_27_0)
    let w_29_0 : Int := 16
    let w_30_0 : Int := 0
    let w_31_0 : Int := (w_29_0 + w_30_0)
    let w_32_0 : Int := 16
    let w_33_0 : Int := 0
    let w_34_0 : Int := (w_32_0 + w_33_0)
    let w_35_0 : Int := 16
    let w_36_0 : Int := 0
    let w_37_0 : Int := (w_35_0 + w_36_0)
    let w_38_0 : Int := 16
    let w_39_0 : Int := 0
    let w_40_0 : Int := (w_38_0 + w_39_0)
    let w_41_0 : Int := 16
    let w_42_0 : Int := 0
    let w_43_0 : Int := (w_41_0 + w_42_0)
    let w_44_0 : Int := 16
    let w_45_0 : Int := 0
    let w_46_0 : Int := (w_44_0 + w_45_0)
    ∃ (w_47_0 : Int),
      ∃ (w_48_0 : Int),
        let w_49_0 : Int := (w_3_0 * w_48_0)
        let w_50_0 : Int := (w_0_0 + w_49_0)
        sequential_generatedRoot_4.constraints_0 w_1_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_13_0 w_16_0 w_19_0 w_22_0 w_25_0 w_28_0 w_31_0 w_34_0 w_37_0 w_40_0 w_43_0 w_46_0 w_47_0 w_48_0 w_50_0 outputs

abbrev sequential_generatedRoot_8.constraints_0 (w_1_0 : Fin 14 → Int) (w_4_0 : Fin 17 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_28_0 : Int) (w_31_0 : Int) (w_34_0 : Int) (w_37_0 : Int) (w_40_0 : Int) (w_43_0 : Int) (w_46_0 : Int) (w_47_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  14 = 14 ∧
  MxxRuntime.select w_2_0 [w_7_0, w_10_0, w_13_0, w_16_0, w_19_0, w_22_0, w_25_0, w_28_0, w_31_0, w_34_0, w_37_0, w_40_0, w_43_0, w_46_0] w_47_0 ∧
  0 ≤ w_47_0 ∧
  w_47_0 < 17 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_47_0 w_48_0 ∧
  outputs = w_50_0

def sequential_generatedRoot_8 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × (Fin 17 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_4_0 : Fin 17 → Int := inputs.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_5_0 : Int := 0
    let w_6_0 : Int := 0
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_8_0 : Int := 1
    let w_9_0 : Int := 0
    let w_10_0 : Int := (w_8_0 + w_9_0)
    let w_11_0 : Int := 2
    let w_12_0 : Int := 0
    let w_13_0 : Int := (w_11_0 + w_12_0)
    let w_14_0 : Int := 4
    let w_15_0 : Int := 0
    let w_16_0 : Int := (w_14_0 + w_15_0)
    let w_17_0 : Int := 6
    let w_18_0 : Int := 0
    let w_19_0 : Int := (w_17_0 + w_18_0)
    let w_20_0 : Int := 7
    let w_21_0 : Int := 0
    let w_22_0 : Int := (w_20_0 + w_21_0)
    let w_23_0 : Int := 8
    let w_24_0 : Int := 0
    let w_25_0 : Int := (w_23_0 + w_24_0)
    let w_26_0 : Int := 9
    let w_27_0 : Int := 0
    let w_28_0 : Int := (w_26_0 + w_27_0)
    let w_29_0 : Int := 12
    let w_30_0 : Int := 0
    let w_31_0 : Int := (w_29_0 + w_30_0)
    let w_32_0 : Int := 13
    let w_33_0 : Int := 0
    let w_34_0 : Int := (w_32_0 + w_33_0)
    let w_35_0 : Int := 14
    let w_36_0 : Int := 0
    let w_37_0 : Int := (w_35_0 + w_36_0)
    let w_38_0 : Int := 15
    let w_39_0 : Int := 0
    let w_40_0 : Int := (w_38_0 + w_39_0)
    let w_41_0 : Int := 16
    let w_42_0 : Int := 0
    let w_43_0 : Int := (w_41_0 + w_42_0)
    let w_44_0 : Int := 16
    let w_45_0 : Int := 0
    let w_46_0 : Int := (w_44_0 + w_45_0)
    ∃ (w_47_0 : Int),
      ∃ (w_48_0 : Int),
        let w_49_0 : Int := (w_3_0 * w_48_0)
        let w_50_0 : Int := (w_0_0 + w_49_0)
        sequential_generatedRoot_8.constraints_0 w_1_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_13_0 w_16_0 w_19_0 w_22_0 w_25_0 w_28_0 w_31_0 w_34_0 w_37_0 w_40_0 w_43_0 w_46_0 w_47_0 w_48_0 w_50_0 outputs

abbrev sequential_generatedRoot_12.constraints_0 (w_1_0 : Fin 14 → Int) (w_4_0 : Fin 17 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_28_0 : Int) (w_31_0 : Int) (w_34_0 : Int) (w_37_0 : Int) (w_40_0 : Int) (w_43_0 : Int) (w_46_0 : Int) (w_47_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  14 = 14 ∧
  MxxRuntime.select w_2_0 [w_7_0, w_10_0, w_13_0, w_16_0, w_19_0, w_22_0, w_25_0, w_28_0, w_31_0, w_34_0, w_37_0, w_40_0, w_43_0, w_46_0] w_47_0 ∧
  0 ≤ w_47_0 ∧
  w_47_0 < 17 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_47_0 w_48_0 ∧
  outputs = w_50_0

def sequential_generatedRoot_12 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × (Fin 17 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_4_0 : Fin 17 → Int := inputs.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_5_0 : Int := 0
    let w_6_0 : Int := 0
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_8_0 : Int := 1
    let w_9_0 : Int := 0
    let w_10_0 : Int := (w_8_0 + w_9_0)
    let w_11_0 : Int := 2
    let w_12_0 : Int := 0
    let w_13_0 : Int := (w_11_0 + w_12_0)
    let w_14_0 : Int := 3
    let w_15_0 : Int := 0
    let w_16_0 : Int := (w_14_0 + w_15_0)
    let w_17_0 : Int := 5
    let w_18_0 : Int := 0
    let w_19_0 : Int := (w_17_0 + w_18_0)
    let w_20_0 : Int := 8
    let w_21_0 : Int := 0
    let w_22_0 : Int := (w_20_0 + w_21_0)
    let w_23_0 : Int := 9
    let w_24_0 : Int := 0
    let w_25_0 : Int := (w_23_0 + w_24_0)
    let w_26_0 : Int := 10
    let w_27_0 : Int := 0
    let w_28_0 : Int := (w_26_0 + w_27_0)
    let w_29_0 : Int := 11
    let w_30_0 : Int := 0
    let w_31_0 : Int := (w_29_0 + w_30_0)
    let w_32_0 : Int := 12
    let w_33_0 : Int := 0
    let w_34_0 : Int := (w_32_0 + w_33_0)
    let w_35_0 : Int := 15
    let w_36_0 : Int := 0
    let w_37_0 : Int := (w_35_0 + w_36_0)
    let w_38_0 : Int := 16
    let w_39_0 : Int := 0
    let w_40_0 : Int := (w_38_0 + w_39_0)
    let w_41_0 : Int := 16
    let w_42_0 : Int := 0
    let w_43_0 : Int := (w_41_0 + w_42_0)
    let w_44_0 : Int := 16
    let w_45_0 : Int := 0
    let w_46_0 : Int := (w_44_0 + w_45_0)
    ∃ (w_47_0 : Int),
      ∃ (w_48_0 : Int),
        let w_49_0 : Int := (w_3_0 * w_48_0)
        let w_50_0 : Int := (w_0_0 + w_49_0)
        sequential_generatedRoot_12.constraints_0 w_1_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_13_0 w_16_0 w_19_0 w_22_0 w_25_0 w_28_0 w_31_0 w_34_0 w_37_0 w_40_0 w_43_0 w_46_0 w_47_0 w_48_0 w_50_0 outputs

abbrev sequential_generatedRoot_16.constraints_0 (w_1_0 : Fin 14 → Int) (w_4_0 : Fin 17 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_28_0 : Int) (w_31_0 : Int) (w_34_0 : Int) (w_37_0 : Int) (w_40_0 : Int) (w_43_0 : Int) (w_46_0 : Int) (w_47_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  14 = 14 ∧
  MxxRuntime.select w_2_0 [w_7_0, w_10_0, w_13_0, w_16_0, w_19_0, w_22_0, w_25_0, w_28_0, w_31_0, w_34_0, w_37_0, w_40_0, w_43_0, w_46_0] w_47_0 ∧
  0 ≤ w_47_0 ∧
  w_47_0 < 17 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_47_0 w_48_0 ∧
  outputs = w_50_0

def sequential_generatedRoot_16 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × (Fin 17 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_4_0 : Fin 17 → Int := inputs.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_5_0 : Int := 0
    let w_6_0 : Int := 0
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_8_0 : Int := 1
    let w_9_0 : Int := 0
    let w_10_0 : Int := (w_8_0 + w_9_0)
    let w_11_0 : Int := 2
    let w_12_0 : Int := 0
    let w_13_0 : Int := (w_11_0 + w_12_0)
    let w_14_0 : Int := 3
    let w_15_0 : Int := 0
    let w_16_0 : Int := (w_14_0 + w_15_0)
    let w_17_0 : Int := 4
    let w_18_0 : Int := 0
    let w_19_0 : Int := (w_17_0 + w_18_0)
    let w_20_0 : Int := 5
    let w_21_0 : Int := 0
    let w_22_0 : Int := (w_20_0 + w_21_0)
    let w_23_0 : Int := 6
    let w_24_0 : Int := 0
    let w_25_0 : Int := (w_23_0 + w_24_0)
    let w_26_0 : Int := 7
    let w_27_0 : Int := 0
    let w_28_0 : Int := (w_26_0 + w_27_0)
    let w_29_0 : Int := 8
    let w_30_0 : Int := 0
    let w_31_0 : Int := (w_29_0 + w_30_0)
    let w_32_0 : Int := 9
    let w_33_0 : Int := 0
    let w_34_0 : Int := (w_32_0 + w_33_0)
    let w_35_0 : Int := 10
    let w_36_0 : Int := 0
    let w_37_0 : Int := (w_35_0 + w_36_0)
    let w_38_0 : Int := 11
    let w_39_0 : Int := 0
    let w_40_0 : Int := (w_38_0 + w_39_0)
    let w_41_0 : Int := 13
    let w_42_0 : Int := 0
    let w_43_0 : Int := (w_41_0 + w_42_0)
    let w_44_0 : Int := 16
    let w_45_0 : Int := 0
    let w_46_0 : Int := (w_44_0 + w_45_0)
    ∃ (w_47_0 : Int),
      ∃ (w_48_0 : Int),
        let w_49_0 : Int := (w_3_0 * w_48_0)
        let w_50_0 : Int := (w_0_0 + w_49_0)
        sequential_generatedRoot_16.constraints_0 w_1_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_13_0 w_16_0 w_19_0 w_22_0 w_25_0 w_28_0 w_31_0 w_34_0 w_37_0 w_40_0 w_43_0 w_46_0 w_47_0 w_48_0 w_50_0 outputs

abbrev sequential_generatedRoot_20.constraints_0 (w_1_0 : Fin 14 → Int) (w_4_0 : Fin 17 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_28_0 : Int) (w_31_0 : Int) (w_34_0 : Int) (w_37_0 : Int) (w_40_0 : Int) (w_43_0 : Int) (w_46_0 : Int) (w_47_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 14 ∧
  14 = 14 ∧
  MxxRuntime.select w_2_0 [w_7_0, w_10_0, w_13_0, w_16_0, w_19_0, w_22_0, w_25_0, w_28_0, w_31_0, w_34_0, w_37_0, w_40_0, w_43_0, w_46_0] w_47_0 ∧
  0 ≤ w_47_0 ∧
  w_47_0 < 17 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_47_0 w_48_0 ∧
  outputs = w_50_0

def sequential_generatedRoot_20 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 14 → Int) × (Fin 17 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 14 → Int := inputs.2.1
  let w_4_0 : Fin 17 → Int := inputs.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_5_0 : Int := 3
    let w_6_0 : Int := 0
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_8_0 : Int := 6
    let w_9_0 : Int := 0
    let w_10_0 : Int := (w_8_0 + w_9_0)
    let w_11_0 : Int := 7
    let w_12_0 : Int := 0
    let w_13_0 : Int := (w_11_0 + w_12_0)
    let w_14_0 : Int := 13
    let w_15_0 : Int := 0
    let w_16_0 : Int := (w_14_0 + w_15_0)
    let w_17_0 : Int := 14
    let w_18_0 : Int := 0
    let w_19_0 : Int := (w_17_0 + w_18_0)
    let w_20_0 : Int := 16
    let w_21_0 : Int := 0
    let w_22_0 : Int := (w_20_0 + w_21_0)
    let w_23_0 : Int := 16
    let w_24_0 : Int := 0
    let w_25_0 : Int := (w_23_0 + w_24_0)
    let w_26_0 : Int := 16
    let w_27_0 : Int := 0
    let w_28_0 : Int := (w_26_0 + w_27_0)
    let w_29_0 : Int := 16
    let w_30_0 : Int := 0
    let w_31_0 : Int := (w_29_0 + w_30_0)
    let w_32_0 : Int := 16
    let w_33_0 : Int := 0
    let w_34_0 : Int := (w_32_0 + w_33_0)
    let w_35_0 : Int := 16
    let w_36_0 : Int := 0
    let w_37_0 : Int := (w_35_0 + w_36_0)
    let w_38_0 : Int := 16
    let w_39_0 : Int := 0
    let w_40_0 : Int := (w_38_0 + w_39_0)
    let w_41_0 : Int := 16
    let w_42_0 : Int := 0
    let w_43_0 : Int := (w_41_0 + w_42_0)
    let w_44_0 : Int := 16
    let w_45_0 : Int := 0
    let w_46_0 : Int := (w_44_0 + w_45_0)
    ∃ (w_47_0 : Int),
      ∃ (w_48_0 : Int),
        let w_49_0 : Int := (w_3_0 * w_48_0)
        let w_50_0 : Int := (w_0_0 + w_49_0)
        sequential_generatedRoot_20.constraints_0 w_1_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_13_0 w_16_0 w_19_0 w_22_0 w_25_0 w_28_0 w_31_0 w_34_0 w_37_0 w_40_0 w_43_0 w_46_0 w_47_0 w_48_0 w_50_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_4_0 : Int
  w_8_0 : Int
  w_12_0 : Int
  w_16_0 : Int
  w_20_0 : Int

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_2_0 : Fin 14 → Int) (w_3_0 : Fin 17 → Int) (w_7_0 : Fin 14 → Int) (w_11_0 : Fin 14 → Int) (w_15_0 : Fin 14 → Int) (w_19_0 : Fin 14 → Int) (w_1_0 : Int) (w_4_0 : Int) (w_6_0 : Int) (w_8_0 : Int) (w_10_0 : Int) (w_12_0 : Int) (w_14_0 : Int) (w_16_0 : Int) (w_18_0 : Int) (w_20_0 : Int) (w_22_0 : Int) (w_26_0 : Int) (w_30_0 : Int) (w_31_0 : Int) (outputs : Int) : Prop :=
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_4 tape (path ++ [4, i]) params i (current, (w_2_0, (w_3_0, ()))) next) (Int.toNat (14)) w_1_0 w_4_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_8 tape (path ++ [8, i]) params i (current, (w_7_0, (w_3_0, ()))) next) (Int.toNat (14)) w_6_0 w_8_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_12 tape (path ++ [12, i]) params i (current, (w_11_0, (w_3_0, ()))) next) (Int.toNat (14)) w_10_0 w_12_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_16 tape (path ++ [16, i]) params i (current, (w_15_0, (w_3_0, ()))) next) (Int.toNat (14)) w_14_0 w_16_0 ∧
  0 ≤ (14) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_20 tape (path ++ [20, i]) params i (current, (w_19_0, (w_3_0, ()))) next) (Int.toNat (14)) w_18_0 w_20_0 ∧
  w_22_0 ≠ 0 ∧
  w_26_0 ≠ 0 ∧
  w_30_0 ≠ 0 ∧
  outputs = w_31_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 14 → Int) × (Fin 17 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × Unit) (outputs : Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_2_0 : Fin 14 → Int := inputs.1
  let w_3_0 : Fin 17 → Int := inputs.2.1
  let w_7_0 : Fin 14 → Int := inputs.2.2.1
  let w_11_0 : Fin 14 → Int := inputs.2.2.2.1
  let w_15_0 : Fin 14 → Int := inputs.2.2.2.2.1
  let w_19_0 : Fin 14 → Int := inputs.2.2.2.2.2.1
  let w_0_0 : Int := 0
  let w_1_0 : Int := 0
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := (w_0_0 + w_4_0)
  let w_6_0 : Int := 0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Int := (w_5_0 + w_8_0)
  let w_10_0 : Int := 0
  let w_12_0 := witness.w_12_0
  let w_13_0 : Int := (w_9_0 + w_12_0)
  let w_14_0 : Int := 0
  let w_16_0 := witness.w_16_0
  let w_17_0 : Int := (w_13_0 + w_16_0)
  let w_18_0 : Int := 0
  let w_20_0 := witness.w_20_0
  let w_21_0 : Int := (w_17_0 + w_20_0)
  let w_22_0 : Int := 16
  let w_23_0 : Int := (w_21_0 % w_22_0)
  let w_24_0 : Int := 2
  let w_25_0 : Int := (w_23_0 * w_24_0)
  let w_26_0 : Int := 16
  let w_27_0 : Int := (w_25_0 / w_26_0)
  let w_28_0 : Int := 1
  let w_29_0 : Int := (w_27_0 + w_28_0)
  let w_30_0 : Int := 2
  let w_31_0 : Int := (w_29_0 % w_30_0)
  generatedRoot.constraints_0 tape path params w_2_0 w_3_0 w_7_0 w_11_0 w_15_0 w_19_0 w_1_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_26_0 w_30_0 w_31_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 14 → Int) × (Fin 17 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × Unit) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Ideal
