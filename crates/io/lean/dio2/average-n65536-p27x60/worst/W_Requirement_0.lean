import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Requirement_0

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_3.constraints_0 (w_2_0 : Fin 8389632 → Int) (w_5_0 : Fin 8389632 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_6_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  outputs = w_11_0

def sequential_generatedRoot_3 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8389632 → Int) × (Fin 8389632 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 8389632 → Int := inputs.2.1
  let w_5_0 : Fin 8389632 → Int := inputs.2.2.1
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

abbrev sequential_generatedRoot_6.constraints_0 (w_2_0 : Fin 1025 → Int) (w_9_0 : Fin 1025 → Int) (w_21_0 : Fin 8389632 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_10_0 : Int) (w_20_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_26_0 : Int) (w_29_0 : Int) (w_30_0 : Int) (w_43_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 1025 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 1025 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_3_0 w_10_0 ∧
  w_22_0 ≠ 0 ∧
  0 ≤ w_25_0 ∧
  w_25_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_25_0 w_26_0 ∧
  0 ≤ w_20_0 ∧
  w_20_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_20_0 [w_26_0, w_29_0] w_30_0 ∧
  outputs = w_43_0

def sequential_generatedRoot_6 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 8389632 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 1025 → Int := inputs.2.1
  let w_9_0 : Fin 1025 → Int := inputs.2.2.1
  let w_21_0 : Fin 8389632 → Int := inputs.2.2.2.1
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
      let w_15_0 : Int := 65536
      let w_16_0 : Int := (w_14_0 * w_15_0)
      let w_17_0 : Int := (w_12_0 + w_16_0)
      let w_18_0 : Int := 1024
      let w_19_0 : Bool := decide (w_3_0 = w_18_0)
      let w_20_0 : Int := if w_19_0 then 1 else 0
      let w_22_0 : Int := 1024
      let w_23_0 : Int := (w_3_0 % w_22_0)
      let w_24_0 : Int := 8193
      let w_25_0 : Int := (w_23_0 * w_24_0)
      ∃ (w_26_0 : Int),
        let w_27_0 : Int := 0
        let w_28_0 : Int := 0
        let w_29_0 : Int := (w_27_0 + w_28_0)
        ∃ (w_30_0 : Int),
          let w_31_0 : Int := 512
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

abbrev sequential_parallel_generatedRoot_41_11.constraints_0 (i_1 : Nat) (w_8_0 : Fin 8389632 → Int) (w_4_0 : Int) (w_7_0 : Int) (w_81_0 : Int) (w_82_0 : Int) (w_83_0 : Int) (w_84_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((13 * (Int.ofNat i_1))) ∧
  0 ≤ ((1 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((2 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((3 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((4 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((5 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((6 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((7 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((8 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((9 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((10 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((11 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ ((12 + (13 * (Int.ofNat i_1)))) ∧
  0 ≤ w_81_0 ∧
  w_81_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_81_0 w_82_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_4_0 [w_7_0, w_82_0] w_83_0 ∧
  outputs = w_84_0

def sequential_parallel_generatedRoot_41_11 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × Int × (Fin 8389632 → Int) × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_2_0 : Int := inputs.2.1
  let w_8_0 : Fin 8389632 → Int := inputs.2.2.1
  let w_9_0 : Int := inputs.2.2.2.1
  let w_13_0 : Int := inputs.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_1)
  let w_3_0 : Bool := decide (w_1_0 < w_2_0)
  let w_4_0 : Int := if w_3_0 then 1 else 0
  let w_5_0 : Int := 0
  let w_6_0 : Int := 0
  let w_7_0 : Int := (w_5_0 + w_6_0)
  let w_10_0 : Int := 8193
  let w_11_0 : Int := (w_9_0 * w_10_0)
  let w_12_0 : Int := 0
  let w_14_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((13 * (Int.ofNat i_1)))))) % 2 = 1)
  let w_15_0 : Int := if w_14_0 then 1 else 0
  let w_16_0 : Int := 1
  let w_17_0 : Int := (w_15_0 * w_16_0)
  let w_18_0 : Int := (w_12_0 + w_17_0)
  let w_19_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((1 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_20_0 : Int := if w_19_0 then 1 else 0
  let w_21_0 : Int := 2
  let w_22_0 : Int := (w_20_0 * w_21_0)
  let w_23_0 : Int := (w_18_0 + w_22_0)
  let w_24_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((2 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_25_0 : Int := if w_24_0 then 1 else 0
  let w_26_0 : Int := 4
  let w_27_0 : Int := (w_25_0 * w_26_0)
  let w_28_0 : Int := (w_23_0 + w_27_0)
  let w_29_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((3 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_30_0 : Int := if w_29_0 then 1 else 0
  let w_31_0 : Int := 8
  let w_32_0 : Int := (w_30_0 * w_31_0)
  let w_33_0 : Int := (w_28_0 + w_32_0)
  let w_34_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((4 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_35_0 : Int := if w_34_0 then 1 else 0
  let w_36_0 : Int := 16
  let w_37_0 : Int := (w_35_0 * w_36_0)
  let w_38_0 : Int := (w_33_0 + w_37_0)
  let w_39_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((5 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_40_0 : Int := if w_39_0 then 1 else 0
  let w_41_0 : Int := 32
  let w_42_0 : Int := (w_40_0 * w_41_0)
  let w_43_0 : Int := (w_38_0 + w_42_0)
  let w_44_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((6 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_45_0 : Int := if w_44_0 then 1 else 0
  let w_46_0 : Int := 64
  let w_47_0 : Int := (w_45_0 * w_46_0)
  let w_48_0 : Int := (w_43_0 + w_47_0)
  let w_49_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((7 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_50_0 : Int := if w_49_0 then 1 else 0
  let w_51_0 : Int := 128
  let w_52_0 : Int := (w_50_0 * w_51_0)
  let w_53_0 : Int := (w_48_0 + w_52_0)
  let w_54_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((8 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_55_0 : Int := if w_54_0 then 1 else 0
  let w_56_0 : Int := 256
  let w_57_0 : Int := (w_55_0 * w_56_0)
  let w_58_0 : Int := (w_53_0 + w_57_0)
  let w_59_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((9 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_60_0 : Int := if w_59_0 then 1 else 0
  let w_61_0 : Int := 512
  let w_62_0 : Int := (w_60_0 * w_61_0)
  let w_63_0 : Int := (w_58_0 + w_62_0)
  let w_64_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((10 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_65_0 : Int := if w_64_0 then 1 else 0
  let w_66_0 : Int := 1024
  let w_67_0 : Int := (w_65_0 * w_66_0)
  let w_68_0 : Int := (w_63_0 + w_67_0)
  let w_69_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((11 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_70_0 : Int := if w_69_0 then 1 else 0
  let w_71_0 : Int := 2048
  let w_72_0 : Int := (w_70_0 * w_71_0)
  let w_73_0 : Int := (w_68_0 + w_72_0)
  let w_74_0 : Bool := decide ((w_13_0 / (2 ^ (Int.toNat ((12 + (13 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_75_0 : Int := if w_74_0 then 1 else 0
  let w_76_0 : Int := 4096
  let w_77_0 : Int := (w_75_0 * w_76_0)
  let w_78_0 : Int := (w_73_0 + w_77_0)
  let w_79_0 : Int := (w_11_0 + w_78_0)
  let w_80_0 : Int := 1
  let w_81_0 : Int := (w_79_0 + w_80_0)
  ∃ (w_82_0 : Int),
    ∃ (w_83_0 : Int),
      let w_84_0 : Int := (w_0_0 + w_83_0)
      sequential_parallel_generatedRoot_41_11.constraints_0 i_1 w_8_0 w_4_0 w_7_0 w_81_0 w_82_0 w_83_0 w_84_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_41.Witness where
  w_5_0 : Int
  w_10_0 : Int
  w_11_0 : Int

abbrev parallel_generatedRoot_41.constraints_0 (params : Params) (i_0 : Nat) (w_1_0 : Fin 8193 → Int) (w_6_0 : Fin 8389632 → Int) (w_9_0 : Fin 8193 → Int) (w_0_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (w_10_0 : Int) (w_11_0 : Int) (w_12_0 : Int) (w_13_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 8193 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  w_7_0 ≠ 0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 8193 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_4_0 w_10_0 ∧
  0 ≤ (4253) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_parallel_generatedRoot_41_11 params i_0 i (current, (w_5_0, (w_6_0, (w_8_0, (w_10_0, ()))))) next) (Int.toNat (4253)) w_0_0 w_11_0 ∧
  w_12_0 ≠ 0 ∧
  outputs = w_13_0

abbrev parallel_generatedRoot_41.body (params : Params) (i_0 : Nat) (inputs : (Fin 8193 → Int) × (Fin 8389632 → Int) × (Fin 8193 → Int) × Unit) (outputs : Int) (witness : parallel_generatedRoot_41.Witness) : Prop :=
  let _params := params
  let w_1_0 : Fin 8193 → Int := inputs.1
  let w_6_0 : Fin 8389632 → Int := inputs.2.1
  let w_9_0 : Fin 8193 → Int := inputs.2.2.1
  let w_0_0 : Int := 0
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 : Int := 8193
  let w_4_0 : Int := (w_2_0 % w_3_0)
  let w_5_0 := witness.w_5_0
  let w_7_0 : Int := 8193
  let w_8_0 : Int := (w_2_0 / w_7_0)
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : Int := 256
  let w_13_0 : Int := (w_11_0 % w_12_0)
  parallel_generatedRoot_41.constraints_0 params i_0 w_1_0 w_6_0 w_9_0 w_0_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 outputs

def parallel_generatedRoot_41 (params : Params) (i_0 : Nat) (inputs : (Fin 8193 → Int) × (Fin 8389632 → Int) × (Fin 8193 → Int) × Unit) (outputs : Int) : Prop :=
  ∃ witness : parallel_generatedRoot_41.Witness,
    parallel_generatedRoot_41.body params i_0 inputs outputs witness

abbrev sequential_generatedRoot_42.constraints_0 (w_2_0 : Fin 1025 → Int) (w_9_0 : Fin 1025 → Int) (w_21_0 : Fin 8389632 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_10_0 : Int) (w_20_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_26_0 : Int) (w_29_0 : Int) (w_30_0 : Int) (w_43_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 1025 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 1025 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_3_0 w_10_0 ∧
  w_22_0 ≠ 0 ∧
  0 ≤ w_25_0 ∧
  w_25_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_25_0 w_26_0 ∧
  0 ≤ w_20_0 ∧
  w_20_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_20_0 [w_26_0, w_29_0] w_30_0 ∧
  outputs = w_43_0

def sequential_generatedRoot_42 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 8389632 → Int) × Unit) (outputs : Bool) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_2_0 : Fin 1025 → Int := inputs.2.1
  let w_9_0 : Fin 1025 → Int := inputs.2.2.1
  let w_21_0 : Fin 8389632 → Int := inputs.2.2.2.1
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
      let w_15_0 : Int := 65536
      let w_16_0 : Int := (w_14_0 * w_15_0)
      let w_17_0 : Int := (w_12_0 + w_16_0)
      let w_18_0 : Int := 1024
      let w_19_0 : Bool := decide (w_3_0 = w_18_0)
      let w_20_0 : Int := if w_19_0 then 1 else 0
      let w_22_0 : Int := 1024
      let w_23_0 : Int := (w_3_0 % w_22_0)
      let w_24_0 : Int := 8193
      let w_25_0 : Int := (w_23_0 * w_24_0)
      ∃ (w_26_0 : Int),
        let w_27_0 : Int := 0
        let w_28_0 : Int := 0
        let w_29_0 : Int := (w_27_0 + w_28_0)
        ∃ (w_30_0 : Int),
          let w_31_0 : Int := 512
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

abbrev sequential_sequential_generatedRoot_74_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_74_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_74.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_74_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_74.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_74.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_74 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_74.Witness,
    sequential_generatedRoot_74.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_76_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_76_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_76.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_76_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_76.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_76.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_76 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_76.Witness,
    sequential_generatedRoot_76.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_78_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_78_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_78.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_78_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_78.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_78.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_78 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_78.Witness,
    sequential_generatedRoot_78.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_80_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_80_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_80.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_80_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_80.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_80.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_80 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_80.Witness,
    sequential_generatedRoot_80.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_82_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_82_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_82.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_82_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_82.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_82.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_82 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_82.Witness,
    sequential_generatedRoot_82.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_84_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_84_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_84.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_84_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_84.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_84.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_84 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_84.Witness,
    sequential_generatedRoot_84.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_86_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_86_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_86.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_86_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_86.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_86.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_86 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_86.Witness,
    sequential_generatedRoot_86.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_88_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_88_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_88.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_88_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_88.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_88.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_88 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_88.Witness,
    sequential_generatedRoot_88.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_90_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_90_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_90.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_90_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_90.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_90.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_90 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_90.Witness,
    sequential_generatedRoot_90.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_92_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_92_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_92.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_92_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_92.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_92.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_92 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_92.Witness,
    sequential_generatedRoot_92.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_94_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_94_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_94.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_94_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_94.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_94.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_94 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_94.Witness,
    sequential_generatedRoot_94.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_96_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_96_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_96.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_96_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_96.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_96.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_96 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_96.Witness,
    sequential_generatedRoot_96.body params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_98_5.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def sequential_sequential_generatedRoot_98_5 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
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

abbrev sequential_generatedRoot_98.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_98_5 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_98.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_98.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_98 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_98.Witness,
    sequential_generatedRoot_98.body params i_0 inputs outputs witness

abbrev scope_155.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_155 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_155.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_100.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_100.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_155 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_100.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_100.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_100 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_100.Witness,
    sequential_generatedRoot_100.body params i_0 inputs outputs witness

abbrev scope_156.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_156 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_156.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_102.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_102.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_156 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_102.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_102.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_102 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_102.Witness,
    sequential_generatedRoot_102.body params i_0 inputs outputs witness

abbrev scope_157.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_157 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_157.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_104.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_104.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_157 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_104.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_104.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_104 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_104.Witness,
    sequential_generatedRoot_104.body params i_0 inputs outputs witness

abbrev scope_158.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_158 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_158.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_106.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_106.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_158 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_106.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_106.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_106 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_106.Witness,
    sequential_generatedRoot_106.body params i_0 inputs outputs witness

abbrev scope_159.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_159 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_159.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_108.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_108.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_159 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_108.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_108.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_108 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_108.Witness,
    sequential_generatedRoot_108.body params i_0 inputs outputs witness

abbrev scope_160.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_160 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_160.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_110.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_110.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_160 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_110.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_110.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_110 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_110.Witness,
    sequential_generatedRoot_110.body params i_0 inputs outputs witness

abbrev scope_161.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_161 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_161.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_112.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_112.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_161 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_112.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_112.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_112 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_112.Witness,
    sequential_generatedRoot_112.body params i_0 inputs outputs witness

abbrev scope_162.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_162 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_162.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_114.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_114.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_162 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_114.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_114.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_114 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_114.Witness,
    sequential_generatedRoot_114.body params i_0 inputs outputs witness

abbrev scope_163.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_163 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_163.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_116.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_116.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_163 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_116.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_116.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_116 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_116.Witness,
    sequential_generatedRoot_116.body params i_0 inputs outputs witness

abbrev scope_164.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_164 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_164.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_118.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_118.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_164 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_118.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_118.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_118 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_118.Witness,
    sequential_generatedRoot_118.body params i_0 inputs outputs witness

abbrev scope_165.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_165 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_165.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_120.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_120.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_165 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_120.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_120.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_120 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_120.Witness,
    sequential_generatedRoot_120.body params i_0 inputs outputs witness

abbrev scope_166.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_166 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_166.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_122.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_122.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_166 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_122.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_122.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_122 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_122.Witness,
    sequential_generatedRoot_122.body params i_0 inputs outputs witness

abbrev scope_167.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_167 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_167.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_124.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_124.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_167 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_124.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_124.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_124 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_124.Witness,
    sequential_generatedRoot_124.body params i_0 inputs outputs witness

abbrev scope_168.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_168 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_168.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_126.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_126.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_168 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_126.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_126.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_126 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_126.Witness,
    sequential_generatedRoot_126.body params i_0 inputs outputs witness

abbrev scope_169.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_169 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_169.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_128.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_128.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_169 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_128.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_128.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_128 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_128.Witness,
    sequential_generatedRoot_128.body params i_0 inputs outputs witness

abbrev scope_170.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_170 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_170.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_130.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_130.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_170 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_130.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_130.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_130 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_130.Witness,
    sequential_generatedRoot_130.body params i_0 inputs outputs witness

abbrev scope_171.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_171 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_171.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_132.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_132.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_171 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_132.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_132.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_132 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_132.Witness,
    sequential_generatedRoot_132.body params i_0 inputs outputs witness

abbrev scope_172.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_172 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_172.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_134.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_134.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_172 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_134.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_134.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_134 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_134.Witness,
    sequential_generatedRoot_134.body params i_0 inputs outputs witness

abbrev scope_173.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_173 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_173.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_136.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_136.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_173 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_136.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_136.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_136 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_136.Witness,
    sequential_generatedRoot_136.body params i_0 inputs outputs witness

abbrev scope_174.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_174 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_174.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_138.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_138.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_174 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_138.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_138.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_138 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_138.Witness,
    sequential_generatedRoot_138.body params i_0 inputs outputs witness

abbrev scope_175.constraints_0 (w_1_0 : Fin 318898285953889075200 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 318898285953889075200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_175 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 318898285953889075200 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 318898285953889075200 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_175.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_140.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_140.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 318898285953889075200 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_175 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_140.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_140.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 318898285953889075200 → Int := inputs.2.1
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

def sequential_generatedRoot_140 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 318898285953889075200 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_140.Witness,
    sequential_generatedRoot_140.body params i_0 inputs outputs witness

abbrev scope_176.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_176 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_176.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_142.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_142.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_176 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_142.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_142.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_142 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_142.Witness,
    sequential_generatedRoot_142.body params i_0 inputs outputs witness

abbrev scope_177.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_177 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_177.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_144.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_144.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_177 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_144.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_144.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_144 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_144.Witness,
    sequential_generatedRoot_144.body params i_0 inputs outputs witness

abbrev scope_178.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_178 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_178.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_146.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_146.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_178 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_146.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_146.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_146 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_146.Witness,
    sequential_generatedRoot_146.body params i_0 inputs outputs witness

abbrev scope_179.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_179 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_179.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_148.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_148.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_179 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_148.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_148.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_148 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_148.Witness,
    sequential_generatedRoot_148.body params i_0 inputs outputs witness

abbrev scope_180.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_180 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_180.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_150.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_150.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_180 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_150.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_150.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_150 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_150.Witness,
    sequential_generatedRoot_150.body params i_0 inputs outputs witness

abbrev scope_181.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_181 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_181.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_152.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_152.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_181 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_152.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_152.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_152 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_152.Witness,
    sequential_generatedRoot_152.body params i_0 inputs outputs witness

abbrev scope_182.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_182 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_182.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_154.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_154.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_182 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_154.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_154.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_154 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_154.Witness,
    sequential_generatedRoot_154.body params i_0 inputs outputs witness

abbrev scope_183.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_183 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_183.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_156.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_156.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_183 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_156.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_156.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_156 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_156.Witness,
    sequential_generatedRoot_156.body params i_0 inputs outputs witness

abbrev scope_184.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_184 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_184.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_158.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_158.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_184 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_158.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_158.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_158 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_158.Witness,
    sequential_generatedRoot_158.body params i_0 inputs outputs witness

abbrev scope_185.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_185 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_185.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_160.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_160.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_185 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_160.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_160.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_160 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_160.Witness,
    sequential_generatedRoot_160.body params i_0 inputs outputs witness

abbrev scope_186.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_186 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_186.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_162.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_162.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_186 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_162.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_162.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_162 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_162.Witness,
    sequential_generatedRoot_162.body params i_0 inputs outputs witness

abbrev scope_187.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_187 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_187.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_164.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_164.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_187 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_164.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_164.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_164 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_164.Witness,
    sequential_generatedRoot_164.body params i_0 inputs outputs witness

abbrev scope_188.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_188 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_188.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_166.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_166.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_188 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_166.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_166.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_166 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_166.Witness,
    sequential_generatedRoot_166.body params i_0 inputs outputs witness

abbrev scope_189.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_189 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_189.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_168.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_168.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_189 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_168.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_168.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_168 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_168.Witness,
    sequential_generatedRoot_168.body params i_0 inputs outputs witness

abbrev scope_190.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_190 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_190.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_170.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_170.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_190 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_170.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_170.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_170 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_170.Witness,
    sequential_generatedRoot_170.body params i_0 inputs outputs witness

abbrev scope_191.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_191 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_191.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_172.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_172.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_191 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_172.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_172.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_172 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_172.Witness,
    sequential_generatedRoot_172.body params i_0 inputs outputs witness

abbrev scope_192.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_192 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_192.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_174.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_174.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_192 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_174.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_174.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_174 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_174.Witness,
    sequential_generatedRoot_174.body params i_0 inputs outputs witness

abbrev scope_193.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_193 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_193.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_176.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_176.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_193 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_176.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_176.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_176 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_176.Witness,
    sequential_generatedRoot_176.body params i_0 inputs outputs witness

abbrev scope_194.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_194 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_194.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_178.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_178.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_194 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_178.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_178.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_178 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_178.Witness,
    sequential_generatedRoot_178.body params i_0 inputs outputs witness

abbrev scope_195.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_195 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_195.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_180.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_180.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_195 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_180.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_180.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_180 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_180.Witness,
    sequential_generatedRoot_180.body params i_0 inputs outputs witness

abbrev scope_196.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_196 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_196.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_182.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_182.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_196 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_182.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_182.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_182 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_182.Witness,
    sequential_generatedRoot_182.body params i_0 inputs outputs witness

abbrev scope_197.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_197 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_197.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_184.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_184.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_197 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_184.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_184.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_184 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_184.Witness,
    sequential_generatedRoot_184.body params i_0 inputs outputs witness

abbrev scope_198.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_198 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_198.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_186.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_186.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_198 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_186.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_186.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_186 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_186.Witness,
    sequential_generatedRoot_186.body params i_0 inputs outputs witness

abbrev scope_199.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_199 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_199.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_188.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_188.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_199 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_188.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_188.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_188 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_188.Witness,
    sequential_generatedRoot_188.body params i_0 inputs outputs witness

abbrev scope_200.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_200 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_200.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_190.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_190.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_200 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_190.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_190.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_190 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_190.Witness,
    sequential_generatedRoot_190.body params i_0 inputs outputs witness

abbrev scope_201.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_201 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_201.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_192.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_192.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_201 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_192.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_192.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_192 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_192.Witness,
    sequential_generatedRoot_192.body params i_0 inputs outputs witness

abbrev scope_202.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_202 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_202.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_194.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_194.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_202 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_194.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_194.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_194 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_194.Witness,
    sequential_generatedRoot_194.body params i_0 inputs outputs witness

abbrev scope_203.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_203 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_203.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_196.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_196.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_203 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_196.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_196.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_196 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_196.Witness,
    sequential_generatedRoot_196.body params i_0 inputs outputs witness

abbrev scope_204.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_204 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_204.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_198.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_198.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_204 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_198.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_198.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_198 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_198.Witness,
    sequential_generatedRoot_198.body params i_0 inputs outputs witness

abbrev scope_205.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_205 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_205.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_200.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_200.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_205 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_200.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_200.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_200 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_200.Witness,
    sequential_generatedRoot_200.body params i_0 inputs outputs witness

abbrev scope_206.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_206 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_206.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_202.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_202.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_206 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_202.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_202.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_202 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_202.Witness,
    sequential_generatedRoot_202.body params i_0 inputs outputs witness

abbrev scope_207.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_207 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_207.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_204.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_204.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_207 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_204.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_204.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_204 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_204.Witness,
    sequential_generatedRoot_204.body params i_0 inputs outputs witness

abbrev scope_208.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_208 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_208.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_206.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_206.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_208 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_206.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_206.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_206 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_206.Witness,
    sequential_generatedRoot_206.body params i_0 inputs outputs witness

abbrev scope_209.constraints_0 (w_1_0 : Fin 8519680 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 8519680 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_209 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 8519680 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 8519680 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_209.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_208.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_208.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 8519680 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_209 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_208.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_208.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 8519680 → Int := inputs.2.1
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

def sequential_generatedRoot_208 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 8519680 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_208.Witness,
    sequential_generatedRoot_208.body params i_0 inputs outputs witness

abbrev scope_210.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_210 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_210.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_210.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_210.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_210 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_210.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_210.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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

def sequential_generatedRoot_210 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_210.Witness,
    sequential_generatedRoot_210.body params i_0 inputs outputs witness

abbrev scope_211.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_211 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_211.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_212.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_212.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_211 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_212.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_212.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_212.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_212 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_212.Witness,
    sequential_generatedRoot_212.body params i_0 inputs outputs witness

abbrev scope_212.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_212 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_212.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_214.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_214.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_212 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_214.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_214.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_214.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_214 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_214.Witness,
    sequential_generatedRoot_214.body params i_0 inputs outputs witness

abbrev scope_213.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_213 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_213.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_216.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_216.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_213 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_216.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_216.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_216.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_216 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_216.Witness,
    sequential_generatedRoot_216.body params i_0 inputs outputs witness

abbrev scope_214.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_214 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_214.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_218.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_218.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_214 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_218.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_218.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_218.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_218 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_218.Witness,
    sequential_generatedRoot_218.body params i_0 inputs outputs witness

abbrev scope_215.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_215 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_215.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_220.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_220.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_215 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_220.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_220.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_220.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_220 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_220.Witness,
    sequential_generatedRoot_220.body params i_0 inputs outputs witness

abbrev scope_216.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_216 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_216.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_222.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_222.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_216 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_222.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_222.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_222.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_222 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_222.Witness,
    sequential_generatedRoot_222.body params i_0 inputs outputs witness

abbrev scope_217.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_217 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_217.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_224.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_224.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_217 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_224.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_224.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_224.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_224 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_224.Witness,
    sequential_generatedRoot_224.body params i_0 inputs outputs witness

abbrev scope_218.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_218 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_218.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_226.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_226.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_218 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_226.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_226.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_226.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_226 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_226.Witness,
    sequential_generatedRoot_226.body params i_0 inputs outputs witness

abbrev scope_219.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_219 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_219.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_228.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_228.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_219 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_228.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_228.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_228.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_228 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_228.Witness,
    sequential_generatedRoot_228.body params i_0 inputs outputs witness

abbrev scope_220.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_220 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_220.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_230.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_230.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_220 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_230.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_230.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_230.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_230 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_230.Witness,
    sequential_generatedRoot_230.body params i_0 inputs outputs witness

abbrev scope_221.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_221 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_221.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_232.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_232.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_221 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_232.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_232.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_232.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_232 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_232.Witness,
    sequential_generatedRoot_232.body params i_0 inputs outputs witness

abbrev scope_222.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_222 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_222.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_234.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_234.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_222 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_234.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_234.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_234.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_234 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_234.Witness,
    sequential_generatedRoot_234.body params i_0 inputs outputs witness

abbrev scope_223.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_223 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_223.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_236.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_236.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_223 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_236.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_236.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_236.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_236 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_236.Witness,
    sequential_generatedRoot_236.body params i_0 inputs outputs witness

abbrev scope_224.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_224 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_224.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_238.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_238.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_224 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_238.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_238.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_238.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_238 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_238.Witness,
    sequential_generatedRoot_238.body params i_0 inputs outputs witness

abbrev scope_225.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_225 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_225.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_240.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_240.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_225 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_240.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_240.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_240.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_240 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_240.Witness,
    sequential_generatedRoot_240.body params i_0 inputs outputs witness

abbrev scope_226.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_226 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_226.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_242.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_242.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_226 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_242.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_242.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_242.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_242 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_242.Witness,
    sequential_generatedRoot_242.body params i_0 inputs outputs witness

abbrev scope_227.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_227 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_227.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_244.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_244.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_227 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_244.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_244.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_244.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_244 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_244.Witness,
    sequential_generatedRoot_244.body params i_0 inputs outputs witness

abbrev scope_228.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_228 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_228.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_246.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_246.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_228 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_246.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_246.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_246.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_246 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_246.Witness,
    sequential_generatedRoot_246.body params i_0 inputs outputs witness

abbrev scope_229.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_229 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_229.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_248.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_248.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_229 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_248.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_248.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_248.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_248 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_248.Witness,
    sequential_generatedRoot_248.body params i_0 inputs outputs witness

abbrev scope_230.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_230 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_230.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_250.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_250.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_230 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_250.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_250.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_250.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_250 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_250.Witness,
    sequential_generatedRoot_250.body params i_0 inputs outputs witness

abbrev scope_231.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_231 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_231.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_252.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_252.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_231 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_252.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_252.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_252.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_252 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_252.Witness,
    sequential_generatedRoot_252.body params i_0 inputs outputs witness

abbrev scope_232.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_232 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_232.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_254.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_254.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_232 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_254.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_254.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_254.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_254 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_254.Witness,
    sequential_generatedRoot_254.body params i_0 inputs outputs witness

abbrev scope_233.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_233 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_233.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_256.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_256.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_233 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_256.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_256.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_256.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_256 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_256.Witness,
    sequential_generatedRoot_256.body params i_0 inputs outputs witness

abbrev scope_234.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_234 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_234.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_258.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_258.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_234 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_258.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_258.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_258.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_258 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_258.Witness,
    sequential_generatedRoot_258.body params i_0 inputs outputs witness

abbrev scope_235.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_235 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_235.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_260.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_260.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_235 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_260.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_260.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_260.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_260 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_260.Witness,
    sequential_generatedRoot_260.body params i_0 inputs outputs witness

abbrev scope_236.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_236 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_236.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_262.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_262.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_236 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_262.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_262.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_262.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_262 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_262.Witness,
    sequential_generatedRoot_262.body params i_0 inputs outputs witness

abbrev scope_237.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_237 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_237.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_264.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_264.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_237 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_264.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_264.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_264.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_264 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_264.Witness,
    sequential_generatedRoot_264.body params i_0 inputs outputs witness

abbrev scope_238.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_238 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_238.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_266.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_266.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_238 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_266.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_266.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_266.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_266 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_266.Witness,
    sequential_generatedRoot_266.body params i_0 inputs outputs witness

abbrev scope_239.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_239 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_239.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_268.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_268.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_239 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_268.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_268.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_268.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_268 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_268.Witness,
    sequential_generatedRoot_268.body params i_0 inputs outputs witness

abbrev scope_240.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_240 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_240.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_270.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_270.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_240 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_270.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_270.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_270.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_270 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_270.Witness,
    sequential_generatedRoot_270.body params i_0 inputs outputs witness

abbrev scope_241.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_241 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_241.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_272.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_272.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_241 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_272.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_272.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_272.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_272 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_272.Witness,
    sequential_generatedRoot_272.body params i_0 inputs outputs witness

abbrev scope_242.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_242 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_242.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_274.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_274.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_242 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_274.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_274.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_274.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_274 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_274.Witness,
    sequential_generatedRoot_274.body params i_0 inputs outputs witness

abbrev scope_243.constraints_0 (w_1_0 : Fin 38598194862489600 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 38598194862489600 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_243 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 38598194862489600 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 38598194862489600 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_243.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_276.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_276.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 38598194862489600 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_243 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_276.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_276.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 38598194862489600 → Int := inputs.2.1
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
  sequential_generatedRoot_276.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_276 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 38598194862489600 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_276.Witness,
    sequential_generatedRoot_276.body params i_0 inputs outputs witness

abbrev scope_244.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_244 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_244.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_278.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_278.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_244 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_278.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_278.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_278.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_278 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_278.Witness,
    sequential_generatedRoot_278.body params i_0 inputs outputs witness

abbrev scope_245.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_245 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_245.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_280.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_280.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_245 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_280.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_280.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_280.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_280 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_280.Witness,
    sequential_generatedRoot_280.body params i_0 inputs outputs witness

abbrev scope_246.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_246 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_246.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_282.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_282.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_246 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_282.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_282.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_282.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_282 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_282.Witness,
    sequential_generatedRoot_282.body params i_0 inputs outputs witness

abbrev scope_247.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_247 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_247.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_284.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_284.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_247 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_284.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_284.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_284.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_284 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_284.Witness,
    sequential_generatedRoot_284.body params i_0 inputs outputs witness

abbrev scope_248.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_248 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_248.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_286.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_286.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_248 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_286.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_286.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_286.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_286 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_286.Witness,
    sequential_generatedRoot_286.body params i_0 inputs outputs witness

abbrev scope_249.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_249 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_249.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_288.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_288.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_249 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_288.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_288.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_288.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_288 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_288.Witness,
    sequential_generatedRoot_288.body params i_0 inputs outputs witness

abbrev scope_250.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_250 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_250.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_290.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_290.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_250 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_290.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_290.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_290.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_290 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_290.Witness,
    sequential_generatedRoot_290.body params i_0 inputs outputs witness

abbrev scope_251.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_251 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_251.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_292.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_292.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_251 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_292.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_292.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_292.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_292 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_292.Witness,
    sequential_generatedRoot_292.body params i_0 inputs outputs witness

abbrev scope_252.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_252 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_252.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_294.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_294.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_252 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_294.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_294.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_294.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_294 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_294.Witness,
    sequential_generatedRoot_294.body params i_0 inputs outputs witness

abbrev scope_253.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_253 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_253.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_296.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_296.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_253 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_296.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_296.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_296.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_296 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_296.Witness,
    sequential_generatedRoot_296.body params i_0 inputs outputs witness

abbrev scope_254.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_254 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_254.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_298.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_298.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_254 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_298.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_298.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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

def sequential_generatedRoot_298 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_298.Witness,
    sequential_generatedRoot_298.body params i_0 inputs outputs witness

abbrev scope_255.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_255 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_255.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_300.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_300.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_255 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_300.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_300.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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

def sequential_generatedRoot_300 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_300.Witness,
    sequential_generatedRoot_300.body params i_0 inputs outputs witness

abbrev scope_256.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_256 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_256.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_302.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_302.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_256 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_302.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_302.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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

def sequential_generatedRoot_302 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_302.Witness,
    sequential_generatedRoot_302.body params i_0 inputs outputs witness

abbrev scope_257.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_257 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_257.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_304.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_304.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_257 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_304.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_304.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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

def sequential_generatedRoot_304 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_304.Witness,
    sequential_generatedRoot_304.body params i_0 inputs outputs witness

abbrev scope_258.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_258 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_258.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_306.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_306.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_258 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_306.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_306.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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

def sequential_generatedRoot_306 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_306.Witness,
    sequential_generatedRoot_306.body params i_0 inputs outputs witness

abbrev scope_259.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_259 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_259.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_308.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_308.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_259 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_308.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_308.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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

def sequential_generatedRoot_308 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_308.Witness,
    sequential_generatedRoot_308.body params i_0 inputs outputs witness

abbrev scope_260.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_260 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_260.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_310.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_310.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_260 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_310.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_310.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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

def sequential_generatedRoot_310 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_310.Witness,
    sequential_generatedRoot_310.body params i_0 inputs outputs witness

abbrev scope_261.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_261 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_261.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_312.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_312.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_261 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_312.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_312.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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

def sequential_generatedRoot_312 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_312.Witness,
    sequential_generatedRoot_312.body params i_0 inputs outputs witness

abbrev scope_262.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_262 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_262.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_314.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_314.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_262 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_314.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_314.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_314.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_314 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_314.Witness,
    sequential_generatedRoot_314.body params i_0 inputs outputs witness

abbrev scope_263.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_263 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_263.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_316.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_316.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_263 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_316.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_316.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_316.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_316 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_316.Witness,
    sequential_generatedRoot_316.body params i_0 inputs outputs witness

abbrev scope_264.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_264 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_264.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_318.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_318.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_264 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_318.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_318.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_318.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_318 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_318.Witness,
    sequential_generatedRoot_318.body params i_0 inputs outputs witness

abbrev scope_265.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_265 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_265.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_320.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_320.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_265 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_320.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_320.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_320.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_320 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_320.Witness,
    sequential_generatedRoot_320.body params i_0 inputs outputs witness

abbrev scope_266.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_266 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_266.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_322.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_322.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_266 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_322.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_322.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_322.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_322 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_322.Witness,
    sequential_generatedRoot_322.body params i_0 inputs outputs witness

abbrev scope_267.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_267 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_267.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_324.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_324.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_267 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_324.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_324.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_324.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_324 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_324.Witness,
    sequential_generatedRoot_324.body params i_0 inputs outputs witness

abbrev scope_268.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_268 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_268.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_326.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_326.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_268 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_326.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_326.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_326.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_326 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_326.Witness,
    sequential_generatedRoot_326.body params i_0 inputs outputs witness

abbrev scope_269.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_269 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_269.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_328.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_328.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_269 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_328.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_328.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_328.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_328 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_328.Witness,
    sequential_generatedRoot_328.body params i_0 inputs outputs witness

abbrev scope_270.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_270 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_270.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_330.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_330.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_270 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_330.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_330.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_330.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_330 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_330.Witness,
    sequential_generatedRoot_330.body params i_0 inputs outputs witness

abbrev scope_271.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_271 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_271.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_332.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_332.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_271 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_332.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_332.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_332.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_332 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_332.Witness,
    sequential_generatedRoot_332.body params i_0 inputs outputs witness

abbrev scope_272.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_272 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_272.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_334.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_334.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_272 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_334.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_334.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_334.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_334 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_334.Witness,
    sequential_generatedRoot_334.body params i_0 inputs outputs witness

abbrev scope_273.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_273 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_273.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_336.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_336.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_273 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_336.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_336.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_336.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_336 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_336.Witness,
    sequential_generatedRoot_336.body params i_0 inputs outputs witness

abbrev scope_274.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_274 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_274.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_338.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_338.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_274 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_338.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_338.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_338.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_338 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_338.Witness,
    sequential_generatedRoot_338.body params i_0 inputs outputs witness

abbrev scope_275.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_275 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_275.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_340.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_340.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_275 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_340.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_340.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_340.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_340 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_340.Witness,
    sequential_generatedRoot_340.body params i_0 inputs outputs witness

abbrev scope_276.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_276 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_276.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_342.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_342.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_276 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_342.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_342.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_342.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_342 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_342.Witness,
    sequential_generatedRoot_342.body params i_0 inputs outputs witness

abbrev scope_277.constraints_0 (w_1_0 : Fin 2607022080 → Int) (w_6_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_6_0 ∧
  w_6_0 < 2607022080 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_6_0 w_7_0 ∧
  outputs = w_8_0

def scope_277 (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2607022080 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2607022080 → Int := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_3_0 : Int := 130
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : Int := (Int.ofNat i_1)
  let w_6_0 : Int := (w_4_0 + w_5_0)
  ∃ (w_7_0 : Int),
    let w_8_0 : Int := (w_0_0 + w_7_0)
    scope_277.constraints_0 w_1_0 w_6_0 w_7_0 w_8_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_344.Witness where
  w_5_0 : Int

abbrev sequential_generatedRoot_344.constraints_0 (params : Params) (i_0 : Nat) (w_3_0 : Fin 2607022080 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_11_0 : Bool) (outputs : Bool) : Prop :=
  0 ≤ (130) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => scope_277 params i_0 i (current, (w_3_0, (w_4_0, ()))) next) (Int.toNat (130)) w_2_0 w_5_0 ∧
  outputs = w_11_0

abbrev sequential_generatedRoot_344.body (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) (witness : sequential_generatedRoot_344.Witness) : Prop :=
  let _params := params
  let w_0_0 : Bool := inputs.1
  let w_3_0 : Fin 2607022080 → Int := inputs.2.1
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
  sequential_generatedRoot_344.constraints_0 params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs

def sequential_generatedRoot_344 (params : Params) (i_0 : Nat) (inputs : Bool × (Fin 2607022080 → Int) × Unit) (outputs : Bool) : Prop :=
  ∃ witness : sequential_generatedRoot_344.Witness,
    sequential_generatedRoot_344.body params i_0 inputs outputs witness

abbrev generatedRoot.InputType : Type :=
  (Fin 8389632 → Int) × (Fin 8389632 → Int) × (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 2 → Int) × (Fin 1025 → Int) × (Fin 1025 → Int) × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 318898285953889075200 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 8519680 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 38598194862489600 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × (Fin 2607022080 → Int) × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Bool
  w_6_0 : Bool
  w_9_0 : Int
  w_15_0 : Int
  w_24_0 : Int
  w_40_0 : Fin 8193 → Int
  w_40_1 : Fin 8193 → Int
  w_41_0 : Fin 8389632 → Int
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
  w_212_0 : Bool
  w_214_0 : Bool
  w_216_0 : Bool
  w_218_0 : Bool
  w_220_0 : Bool
  w_222_0 : Bool
  w_224_0 : Bool
  w_226_0 : Bool
  w_228_0 : Bool
  w_230_0 : Bool
  w_232_0 : Bool
  w_234_0 : Bool
  w_236_0 : Bool
  w_238_0 : Bool
  w_240_0 : Bool
  w_242_0 : Bool
  w_244_0 : Bool
  w_246_0 : Bool
  w_248_0 : Bool
  w_250_0 : Bool
  w_252_0 : Bool
  w_254_0 : Bool
  w_256_0 : Bool
  w_258_0 : Bool
  w_260_0 : Bool
  w_262_0 : Bool
  w_264_0 : Bool
  w_266_0 : Bool
  w_268_0 : Bool
  w_270_0 : Bool
  w_272_0 : Bool
  w_274_0 : Bool
  w_276_0 : Bool
  w_278_0 : Bool
  w_280_0 : Bool
  w_282_0 : Bool
  w_284_0 : Bool
  w_286_0 : Bool
  w_288_0 : Bool
  w_290_0 : Bool
  w_292_0 : Bool
  w_294_0 : Bool
  w_296_0 : Bool
  w_298_0 : Bool
  w_300_0 : Bool
  w_302_0 : Bool
  w_304_0 : Bool
  w_306_0 : Bool
  w_308_0 : Bool
  w_310_0 : Bool
  w_312_0 : Bool
  w_314_0 : Bool
  w_316_0 : Bool
  w_318_0 : Bool
  w_320_0 : Bool
  w_322_0 : Bool
  w_324_0 : Bool
  w_326_0 : Bool
  w_328_0 : Bool
  w_330_0 : Bool
  w_332_0 : Bool
  w_334_0 : Bool
  w_336_0 : Bool
  w_338_0 : Bool
  w_340_0 : Bool
  w_342_0 : Bool
  w_344_0 : Bool

abbrev generatedRoot.constraints_8 (params : Params) (w_313_0 : Fin 2607022080 → Int) (w_315_0 : Fin 2607022080 → Int) (w_317_0 : Fin 2607022080 → Int) (w_319_0 : Fin 2607022080 → Int) (w_321_0 : Fin 2607022080 → Int) (w_323_0 : Fin 2607022080 → Int) (w_325_0 : Fin 2607022080 → Int) (w_327_0 : Fin 2607022080 → Int) (w_329_0 : Fin 2607022080 → Int) (w_331_0 : Fin 2607022080 → Int) (w_333_0 : Fin 2607022080 → Int) (w_335_0 : Fin 2607022080 → Int) (w_337_0 : Fin 2607022080 → Int) (w_339_0 : Fin 2607022080 → Int) (w_341_0 : Fin 2607022080 → Int) (w_343_0 : Fin 2607022080 → Int) (w_312_0 : Bool) (w_314_0 : Bool) (w_316_0 : Bool) (w_318_0 : Bool) (w_320_0 : Bool) (w_322_0 : Bool) (w_324_0 : Bool) (w_326_0 : Bool) (w_328_0 : Bool) (w_330_0 : Bool) (w_332_0 : Bool) (w_334_0 : Bool) (w_336_0 : Bool) (w_338_0 : Bool) (w_340_0 : Bool) (w_342_0 : Bool) (w_344_0 : Bool) (outputs : Bool) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_314 params i (current, (w_313_0, ())) next) (Int.toNat (20054016)) w_312_0 w_314_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_316 params i (current, (w_315_0, ())) next) (Int.toNat (20054016)) w_314_0 w_316_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_318 params i (current, (w_317_0, ())) next) (Int.toNat (20054016)) w_316_0 w_318_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_320 params i (current, (w_319_0, ())) next) (Int.toNat (20054016)) w_318_0 w_320_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_322 params i (current, (w_321_0, ())) next) (Int.toNat (20054016)) w_320_0 w_322_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_324 params i (current, (w_323_0, ())) next) (Int.toNat (20054016)) w_322_0 w_324_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_326 params i (current, (w_325_0, ())) next) (Int.toNat (20054016)) w_324_0 w_326_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_328 params i (current, (w_327_0, ())) next) (Int.toNat (20054016)) w_326_0 w_328_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_330 params i (current, (w_329_0, ())) next) (Int.toNat (20054016)) w_328_0 w_330_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_332 params i (current, (w_331_0, ())) next) (Int.toNat (20054016)) w_330_0 w_332_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_334 params i (current, (w_333_0, ())) next) (Int.toNat (20054016)) w_332_0 w_334_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_336 params i (current, (w_335_0, ())) next) (Int.toNat (20054016)) w_334_0 w_336_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_338 params i (current, (w_337_0, ())) next) (Int.toNat (20054016)) w_336_0 w_338_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_340 params i (current, (w_339_0, ())) next) (Int.toNat (20054016)) w_338_0 w_340_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_342 params i (current, (w_341_0, ())) next) (Int.toNat (20054016)) w_340_0 w_342_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_344 params i (current, (w_343_0, ())) next) (Int.toNat (20054016)) w_342_0 w_344_0 ∧
  outputs = w_344_0

abbrev generatedRoot.constraints_7 (params : Params) (w_281_0 : Fin 2607022080 → Int) (w_283_0 : Fin 2607022080 → Int) (w_285_0 : Fin 2607022080 → Int) (w_287_0 : Fin 2607022080 → Int) (w_289_0 : Fin 2607022080 → Int) (w_291_0 : Fin 2607022080 → Int) (w_293_0 : Fin 2607022080 → Int) (w_295_0 : Fin 2607022080 → Int) (w_297_0 : Fin 2607022080 → Int) (w_299_0 : Fin 2607022080 → Int) (w_301_0 : Fin 2607022080 → Int) (w_303_0 : Fin 2607022080 → Int) (w_305_0 : Fin 2607022080 → Int) (w_307_0 : Fin 2607022080 → Int) (w_309_0 : Fin 2607022080 → Int) (w_311_0 : Fin 2607022080 → Int) (w_280_0 : Bool) (w_282_0 : Bool) (w_284_0 : Bool) (w_286_0 : Bool) (w_288_0 : Bool) (w_290_0 : Bool) (w_292_0 : Bool) (w_294_0 : Bool) (w_296_0 : Bool) (w_298_0 : Bool) (w_300_0 : Bool) (w_302_0 : Bool) (w_304_0 : Bool) (w_306_0 : Bool) (w_308_0 : Bool) (w_310_0 : Bool) (w_312_0 : Bool) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_282 params i (current, (w_281_0, ())) next) (Int.toNat (20054016)) w_280_0 w_282_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_284 params i (current, (w_283_0, ())) next) (Int.toNat (20054016)) w_282_0 w_284_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_286 params i (current, (w_285_0, ())) next) (Int.toNat (20054016)) w_284_0 w_286_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_288 params i (current, (w_287_0, ())) next) (Int.toNat (20054016)) w_286_0 w_288_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_290 params i (current, (w_289_0, ())) next) (Int.toNat (20054016)) w_288_0 w_290_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_292 params i (current, (w_291_0, ())) next) (Int.toNat (20054016)) w_290_0 w_292_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_294 params i (current, (w_293_0, ())) next) (Int.toNat (20054016)) w_292_0 w_294_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_296 params i (current, (w_295_0, ())) next) (Int.toNat (20054016)) w_294_0 w_296_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_298 params i (current, (w_297_0, ())) next) (Int.toNat (20054016)) w_296_0 w_298_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_300 params i (current, (w_299_0, ())) next) (Int.toNat (20054016)) w_298_0 w_300_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_302 params i (current, (w_301_0, ())) next) (Int.toNat (20054016)) w_300_0 w_302_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_304 params i (current, (w_303_0, ())) next) (Int.toNat (20054016)) w_302_0 w_304_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_306 params i (current, (w_305_0, ())) next) (Int.toNat (20054016)) w_304_0 w_306_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_308 params i (current, (w_307_0, ())) next) (Int.toNat (20054016)) w_306_0 w_308_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_310 params i (current, (w_309_0, ())) next) (Int.toNat (20054016)) w_308_0 w_310_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_312 params i (current, (w_311_0, ())) next) (Int.toNat (20054016)) w_310_0 w_312_0 ∧
  0 ≤ (20054016)

abbrev generatedRoot.constraints_6 (params : Params) (w_249_0 : Fin 38598194862489600 → Int) (w_251_0 : Fin 38598194862489600 → Int) (w_253_0 : Fin 38598194862489600 → Int) (w_255_0 : Fin 38598194862489600 → Int) (w_257_0 : Fin 38598194862489600 → Int) (w_259_0 : Fin 38598194862489600 → Int) (w_261_0 : Fin 38598194862489600 → Int) (w_263_0 : Fin 38598194862489600 → Int) (w_265_0 : Fin 38598194862489600 → Int) (w_267_0 : Fin 38598194862489600 → Int) (w_269_0 : Fin 38598194862489600 → Int) (w_271_0 : Fin 38598194862489600 → Int) (w_273_0 : Fin 38598194862489600 → Int) (w_275_0 : Fin 38598194862489600 → Int) (w_277_0 : Fin 2607022080 → Int) (w_279_0 : Fin 2607022080 → Int) (w_248_0 : Bool) (w_250_0 : Bool) (w_252_0 : Bool) (w_254_0 : Bool) (w_256_0 : Bool) (w_258_0 : Bool) (w_260_0 : Bool) (w_262_0 : Bool) (w_264_0 : Bool) (w_266_0 : Bool) (w_268_0 : Bool) (w_270_0 : Bool) (w_272_0 : Bool) (w_274_0 : Bool) (w_276_0 : Bool) (w_278_0 : Bool) (w_280_0 : Bool) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_250 params i (current, (w_249_0, ())) next) (Int.toNat (296909191249920)) w_248_0 w_250_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_252 params i (current, (w_251_0, ())) next) (Int.toNat (296909191249920)) w_250_0 w_252_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_254 params i (current, (w_253_0, ())) next) (Int.toNat (296909191249920)) w_252_0 w_254_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_256 params i (current, (w_255_0, ())) next) (Int.toNat (296909191249920)) w_254_0 w_256_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_258 params i (current, (w_257_0, ())) next) (Int.toNat (296909191249920)) w_256_0 w_258_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_260 params i (current, (w_259_0, ())) next) (Int.toNat (296909191249920)) w_258_0 w_260_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_262 params i (current, (w_261_0, ())) next) (Int.toNat (296909191249920)) w_260_0 w_262_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_264 params i (current, (w_263_0, ())) next) (Int.toNat (296909191249920)) w_262_0 w_264_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_266 params i (current, (w_265_0, ())) next) (Int.toNat (296909191249920)) w_264_0 w_266_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_268 params i (current, (w_267_0, ())) next) (Int.toNat (296909191249920)) w_266_0 w_268_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_270 params i (current, (w_269_0, ())) next) (Int.toNat (296909191249920)) w_268_0 w_270_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_272 params i (current, (w_271_0, ())) next) (Int.toNat (296909191249920)) w_270_0 w_272_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_274 params i (current, (w_273_0, ())) next) (Int.toNat (296909191249920)) w_272_0 w_274_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_276 params i (current, (w_275_0, ())) next) (Int.toNat (296909191249920)) w_274_0 w_276_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_278 params i (current, (w_277_0, ())) next) (Int.toNat (20054016)) w_276_0 w_278_0 ∧
  0 ≤ (20054016) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_280 params i (current, (w_279_0, ())) next) (Int.toNat (20054016)) w_278_0 w_280_0 ∧
  0 ≤ (20054016)

abbrev generatedRoot.constraints_5 (params : Params) (w_217_0 : Fin 38598194862489600 → Int) (w_219_0 : Fin 38598194862489600 → Int) (w_221_0 : Fin 38598194862489600 → Int) (w_223_0 : Fin 38598194862489600 → Int) (w_225_0 : Fin 38598194862489600 → Int) (w_227_0 : Fin 38598194862489600 → Int) (w_229_0 : Fin 38598194862489600 → Int) (w_231_0 : Fin 38598194862489600 → Int) (w_233_0 : Fin 38598194862489600 → Int) (w_235_0 : Fin 38598194862489600 → Int) (w_237_0 : Fin 38598194862489600 → Int) (w_239_0 : Fin 38598194862489600 → Int) (w_241_0 : Fin 38598194862489600 → Int) (w_243_0 : Fin 38598194862489600 → Int) (w_245_0 : Fin 38598194862489600 → Int) (w_247_0 : Fin 38598194862489600 → Int) (w_216_0 : Bool) (w_218_0 : Bool) (w_220_0 : Bool) (w_222_0 : Bool) (w_224_0 : Bool) (w_226_0 : Bool) (w_228_0 : Bool) (w_230_0 : Bool) (w_232_0 : Bool) (w_234_0 : Bool) (w_236_0 : Bool) (w_238_0 : Bool) (w_240_0 : Bool) (w_242_0 : Bool) (w_244_0 : Bool) (w_246_0 : Bool) (w_248_0 : Bool) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_218 params i (current, (w_217_0, ())) next) (Int.toNat (296909191249920)) w_216_0 w_218_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_220 params i (current, (w_219_0, ())) next) (Int.toNat (296909191249920)) w_218_0 w_220_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_222 params i (current, (w_221_0, ())) next) (Int.toNat (296909191249920)) w_220_0 w_222_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_224 params i (current, (w_223_0, ())) next) (Int.toNat (296909191249920)) w_222_0 w_224_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_226 params i (current, (w_225_0, ())) next) (Int.toNat (296909191249920)) w_224_0 w_226_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_228 params i (current, (w_227_0, ())) next) (Int.toNat (296909191249920)) w_226_0 w_228_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_230 params i (current, (w_229_0, ())) next) (Int.toNat (296909191249920)) w_228_0 w_230_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_232 params i (current, (w_231_0, ())) next) (Int.toNat (296909191249920)) w_230_0 w_232_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_234 params i (current, (w_233_0, ())) next) (Int.toNat (296909191249920)) w_232_0 w_234_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_236 params i (current, (w_235_0, ())) next) (Int.toNat (296909191249920)) w_234_0 w_236_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_238 params i (current, (w_237_0, ())) next) (Int.toNat (296909191249920)) w_236_0 w_238_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_240 params i (current, (w_239_0, ())) next) (Int.toNat (296909191249920)) w_238_0 w_240_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_242 params i (current, (w_241_0, ())) next) (Int.toNat (296909191249920)) w_240_0 w_242_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_244 params i (current, (w_243_0, ())) next) (Int.toNat (296909191249920)) w_242_0 w_244_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_246 params i (current, (w_245_0, ())) next) (Int.toNat (296909191249920)) w_244_0 w_246_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_248 params i (current, (w_247_0, ())) next) (Int.toNat (296909191249920)) w_246_0 w_248_0 ∧
  0 ≤ (296909191249920)

abbrev generatedRoot.constraints_4 (params : Params) (w_185_0 : Fin 8519680 → Int) (w_187_0 : Fin 8519680 → Int) (w_189_0 : Fin 8519680 → Int) (w_191_0 : Fin 8519680 → Int) (w_193_0 : Fin 8519680 → Int) (w_195_0 : Fin 8519680 → Int) (w_197_0 : Fin 8519680 → Int) (w_199_0 : Fin 8519680 → Int) (w_201_0 : Fin 8519680 → Int) (w_203_0 : Fin 8519680 → Int) (w_205_0 : Fin 8519680 → Int) (w_207_0 : Fin 8519680 → Int) (w_209_0 : Fin 38598194862489600 → Int) (w_211_0 : Fin 38598194862489600 → Int) (w_213_0 : Fin 38598194862489600 → Int) (w_215_0 : Fin 38598194862489600 → Int) (w_184_0 : Bool) (w_186_0 : Bool) (w_188_0 : Bool) (w_190_0 : Bool) (w_192_0 : Bool) (w_194_0 : Bool) (w_196_0 : Bool) (w_198_0 : Bool) (w_200_0 : Bool) (w_202_0 : Bool) (w_204_0 : Bool) (w_206_0 : Bool) (w_208_0 : Bool) (w_210_0 : Bool) (w_212_0 : Bool) (w_214_0 : Bool) (w_216_0 : Bool) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_186 params i (current, (w_185_0, ())) next) (Int.toNat (65536)) w_184_0 w_186_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_188 params i (current, (w_187_0, ())) next) (Int.toNat (65536)) w_186_0 w_188_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_190 params i (current, (w_189_0, ())) next) (Int.toNat (65536)) w_188_0 w_190_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_192 params i (current, (w_191_0, ())) next) (Int.toNat (65536)) w_190_0 w_192_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_194 params i (current, (w_193_0, ())) next) (Int.toNat (65536)) w_192_0 w_194_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_196 params i (current, (w_195_0, ())) next) (Int.toNat (65536)) w_194_0 w_196_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_198 params i (current, (w_197_0, ())) next) (Int.toNat (65536)) w_196_0 w_198_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_200 params i (current, (w_199_0, ())) next) (Int.toNat (65536)) w_198_0 w_200_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_202 params i (current, (w_201_0, ())) next) (Int.toNat (65536)) w_200_0 w_202_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_204 params i (current, (w_203_0, ())) next) (Int.toNat (65536)) w_202_0 w_204_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_206 params i (current, (w_205_0, ())) next) (Int.toNat (65536)) w_204_0 w_206_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_208 params i (current, (w_207_0, ())) next) (Int.toNat (65536)) w_206_0 w_208_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_210 params i (current, (w_209_0, ())) next) (Int.toNat (296909191249920)) w_208_0 w_210_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_212 params i (current, (w_211_0, ())) next) (Int.toNat (296909191249920)) w_210_0 w_212_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_214 params i (current, (w_213_0, ())) next) (Int.toNat (296909191249920)) w_212_0 w_214_0 ∧
  0 ≤ (296909191249920) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_216 params i (current, (w_215_0, ())) next) (Int.toNat (296909191249920)) w_214_0 w_216_0 ∧
  0 ≤ (296909191249920)

abbrev generatedRoot.constraints_3 (params : Params) (w_153_0 : Fin 8519680 → Int) (w_155_0 : Fin 8519680 → Int) (w_157_0 : Fin 8519680 → Int) (w_159_0 : Fin 8519680 → Int) (w_161_0 : Fin 8519680 → Int) (w_163_0 : Fin 8519680 → Int) (w_165_0 : Fin 8519680 → Int) (w_167_0 : Fin 8519680 → Int) (w_169_0 : Fin 8519680 → Int) (w_171_0 : Fin 8519680 → Int) (w_173_0 : Fin 8519680 → Int) (w_175_0 : Fin 8519680 → Int) (w_177_0 : Fin 8519680 → Int) (w_179_0 : Fin 8519680 → Int) (w_181_0 : Fin 8519680 → Int) (w_183_0 : Fin 8519680 → Int) (w_152_0 : Bool) (w_154_0 : Bool) (w_156_0 : Bool) (w_158_0 : Bool) (w_160_0 : Bool) (w_162_0 : Bool) (w_164_0 : Bool) (w_166_0 : Bool) (w_168_0 : Bool) (w_170_0 : Bool) (w_172_0 : Bool) (w_174_0 : Bool) (w_176_0 : Bool) (w_178_0 : Bool) (w_180_0 : Bool) (w_182_0 : Bool) (w_184_0 : Bool) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_154 params i (current, (w_153_0, ())) next) (Int.toNat (65536)) w_152_0 w_154_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_156 params i (current, (w_155_0, ())) next) (Int.toNat (65536)) w_154_0 w_156_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_158 params i (current, (w_157_0, ())) next) (Int.toNat (65536)) w_156_0 w_158_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_160 params i (current, (w_159_0, ())) next) (Int.toNat (65536)) w_158_0 w_160_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_162 params i (current, (w_161_0, ())) next) (Int.toNat (65536)) w_160_0 w_162_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_164 params i (current, (w_163_0, ())) next) (Int.toNat (65536)) w_162_0 w_164_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_166 params i (current, (w_165_0, ())) next) (Int.toNat (65536)) w_164_0 w_166_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_168 params i (current, (w_167_0, ())) next) (Int.toNat (65536)) w_166_0 w_168_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_170 params i (current, (w_169_0, ())) next) (Int.toNat (65536)) w_168_0 w_170_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_172 params i (current, (w_171_0, ())) next) (Int.toNat (65536)) w_170_0 w_172_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_174 params i (current, (w_173_0, ())) next) (Int.toNat (65536)) w_172_0 w_174_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_176 params i (current, (w_175_0, ())) next) (Int.toNat (65536)) w_174_0 w_176_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_178 params i (current, (w_177_0, ())) next) (Int.toNat (65536)) w_176_0 w_178_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_180 params i (current, (w_179_0, ())) next) (Int.toNat (65536)) w_178_0 w_180_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_182 params i (current, (w_181_0, ())) next) (Int.toNat (65536)) w_180_0 w_182_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_184 params i (current, (w_183_0, ())) next) (Int.toNat (65536)) w_182_0 w_184_0 ∧
  0 ≤ (65536)

abbrev generatedRoot.constraints_2 (params : Params) (w_121_0 : Fin 318898285953889075200 → Int) (w_123_0 : Fin 318898285953889075200 → Int) (w_125_0 : Fin 318898285953889075200 → Int) (w_127_0 : Fin 318898285953889075200 → Int) (w_129_0 : Fin 318898285953889075200 → Int) (w_131_0 : Fin 318898285953889075200 → Int) (w_133_0 : Fin 318898285953889075200 → Int) (w_135_0 : Fin 318898285953889075200 → Int) (w_137_0 : Fin 318898285953889075200 → Int) (w_139_0 : Fin 318898285953889075200 → Int) (w_141_0 : Fin 8519680 → Int) (w_143_0 : Fin 8519680 → Int) (w_145_0 : Fin 8519680 → Int) (w_147_0 : Fin 8519680 → Int) (w_149_0 : Fin 8519680 → Int) (w_151_0 : Fin 8519680 → Int) (w_120_0 : Bool) (w_122_0 : Bool) (w_124_0 : Bool) (w_126_0 : Bool) (w_128_0 : Bool) (w_130_0 : Bool) (w_132_0 : Bool) (w_134_0 : Bool) (w_136_0 : Bool) (w_138_0 : Bool) (w_140_0 : Bool) (w_142_0 : Bool) (w_144_0 : Bool) (w_146_0 : Bool) (w_148_0 : Bool) (w_150_0 : Bool) (w_152_0 : Bool) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_122 params i (current, (w_121_0, ())) next) (Int.toNat (2453063738106839040)) w_120_0 w_122_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_124 params i (current, (w_123_0, ())) next) (Int.toNat (2453063738106839040)) w_122_0 w_124_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_126 params i (current, (w_125_0, ())) next) (Int.toNat (2453063738106839040)) w_124_0 w_126_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_128 params i (current, (w_127_0, ())) next) (Int.toNat (2453063738106839040)) w_126_0 w_128_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_130 params i (current, (w_129_0, ())) next) (Int.toNat (2453063738106839040)) w_128_0 w_130_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_132 params i (current, (w_131_0, ())) next) (Int.toNat (2453063738106839040)) w_130_0 w_132_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_134 params i (current, (w_133_0, ())) next) (Int.toNat (2453063738106839040)) w_132_0 w_134_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_136 params i (current, (w_135_0, ())) next) (Int.toNat (2453063738106839040)) w_134_0 w_136_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_138 params i (current, (w_137_0, ())) next) (Int.toNat (2453063738106839040)) w_136_0 w_138_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_140 params i (current, (w_139_0, ())) next) (Int.toNat (2453063738106839040)) w_138_0 w_140_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_142 params i (current, (w_141_0, ())) next) (Int.toNat (65536)) w_140_0 w_142_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_144 params i (current, (w_143_0, ())) next) (Int.toNat (65536)) w_142_0 w_144_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_146 params i (current, (w_145_0, ())) next) (Int.toNat (65536)) w_144_0 w_146_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_148 params i (current, (w_147_0, ())) next) (Int.toNat (65536)) w_146_0 w_148_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_150 params i (current, (w_149_0, ())) next) (Int.toNat (65536)) w_148_0 w_150_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_152 params i (current, (w_151_0, ())) next) (Int.toNat (65536)) w_150_0 w_152_0 ∧
  0 ≤ (65536)

abbrev generatedRoot.constraints_1 (params : Params) (w_89_0 : Fin 318898285953889075200 → Int) (w_91_0 : Fin 318898285953889075200 → Int) (w_93_0 : Fin 318898285953889075200 → Int) (w_95_0 : Fin 318898285953889075200 → Int) (w_97_0 : Fin 318898285953889075200 → Int) (w_99_0 : Fin 318898285953889075200 → Int) (w_101_0 : Fin 318898285953889075200 → Int) (w_103_0 : Fin 318898285953889075200 → Int) (w_105_0 : Fin 318898285953889075200 → Int) (w_107_0 : Fin 318898285953889075200 → Int) (w_109_0 : Fin 318898285953889075200 → Int) (w_111_0 : Fin 318898285953889075200 → Int) (w_113_0 : Fin 318898285953889075200 → Int) (w_115_0 : Fin 318898285953889075200 → Int) (w_117_0 : Fin 318898285953889075200 → Int) (w_119_0 : Fin 318898285953889075200 → Int) (w_88_0 : Bool) (w_90_0 : Bool) (w_92_0 : Bool) (w_94_0 : Bool) (w_96_0 : Bool) (w_98_0 : Bool) (w_100_0 : Bool) (w_102_0 : Bool) (w_104_0 : Bool) (w_106_0 : Bool) (w_108_0 : Bool) (w_110_0 : Bool) (w_112_0 : Bool) (w_114_0 : Bool) (w_116_0 : Bool) (w_118_0 : Bool) (w_120_0 : Bool) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_90 params i (current, (w_89_0, ())) next) (Int.toNat (2453063738106839040)) w_88_0 w_90_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_92 params i (current, (w_91_0, ())) next) (Int.toNat (2453063738106839040)) w_90_0 w_92_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_94 params i (current, (w_93_0, ())) next) (Int.toNat (2453063738106839040)) w_92_0 w_94_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_96 params i (current, (w_95_0, ())) next) (Int.toNat (2453063738106839040)) w_94_0 w_96_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_98 params i (current, (w_97_0, ())) next) (Int.toNat (2453063738106839040)) w_96_0 w_98_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_100 params i (current, (w_99_0, ())) next) (Int.toNat (2453063738106839040)) w_98_0 w_100_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_102 params i (current, (w_101_0, ())) next) (Int.toNat (2453063738106839040)) w_100_0 w_102_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_104 params i (current, (w_103_0, ())) next) (Int.toNat (2453063738106839040)) w_102_0 w_104_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_106 params i (current, (w_105_0, ())) next) (Int.toNat (2453063738106839040)) w_104_0 w_106_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_108 params i (current, (w_107_0, ())) next) (Int.toNat (2453063738106839040)) w_106_0 w_108_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_110 params i (current, (w_109_0, ())) next) (Int.toNat (2453063738106839040)) w_108_0 w_110_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_112 params i (current, (w_111_0, ())) next) (Int.toNat (2453063738106839040)) w_110_0 w_112_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_114 params i (current, (w_113_0, ())) next) (Int.toNat (2453063738106839040)) w_112_0 w_114_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_116 params i (current, (w_115_0, ())) next) (Int.toNat (2453063738106839040)) w_114_0 w_116_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_118 params i (current, (w_117_0, ())) next) (Int.toNat (2453063738106839040)) w_116_0 w_118_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_120 params i (current, (w_119_0, ())) next) (Int.toNat (2453063738106839040)) w_118_0 w_120_0 ∧
  0 ≤ (2453063738106839040)

abbrev generatedRoot.constraints_0 (params : Params) (w_1_0 : Fin 8389632 → Int) (w_2_0 : Fin 8389632 → Int) (w_4_0 : Fin 1025 → Int) (w_5_0 : Fin 1025 → Int) (w_8_0 : Fin 1 → Int) (w_14_0 : Fin 1 → Int) (w_23_0 : Fin 2 → Int) (w_38_0 : Fin 1025 → Int) (w_39_0 : Fin 1025 → Int) (w_44_0 : Fin 1 → Int) (w_50_0 : Fin 1 → Int) (w_73_0 : Fin 318898285953889075200 → Int) (w_75_0 : Fin 318898285953889075200 → Int) (w_77_0 : Fin 318898285953889075200 → Int) (w_79_0 : Fin 318898285953889075200 → Int) (w_81_0 : Fin 318898285953889075200 → Int) (w_83_0 : Fin 318898285953889075200 → Int) (w_85_0 : Fin 318898285953889075200 → Int) (w_87_0 : Fin 318898285953889075200 → Int) (w_0_0 : Bool) (w_3_0 : Bool) (w_6_0 : Bool) (w_9_0 : Int) (w_15_0 : Int) (w_24_0 : Int) (w_37_0 : Bool) (w_40_0 : Fin 8193 → Int) (w_40_1 : Fin 8193 → Int) (w_41_0 : Fin 8389632 → Int) (w_42_0 : Bool) (w_45_0 : Int) (w_51_0 : Int) (w_59_0 : Int) (w_72_0 : Bool) (w_74_0 : Bool) (w_76_0 : Bool) (w_78_0 : Bool) (w_80_0 : Bool) (w_82_0 : Bool) (w_84_0 : Bool) (w_86_0 : Bool) (w_88_0 : Bool) : Prop :=
  0 ≤ (8389632) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_3 params i (current, (w_1_0, (w_2_0, ()))) next) (Int.toNat (8389632)) w_0_0 w_3_0 ∧
  0 ≤ (1025) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_6 params i (current, (w_4_0, (w_5_0, (w_2_0, ())))) next) (Int.toNat (1025)) w_3_0 w_6_0 ∧
  MxxRuntime.familyGetStatic w_8_0 (0) w_9_0 ∧
  MxxRuntime.familyGetStatic w_14_0 (0) w_15_0 ∧
  MxxRuntime.familyGetStatic w_23_0 (0) w_24_0 ∧
  MxxRuntime.sha256BitColumns [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 8192 8193 13 w_40_0 w_40_1 ∧
  (8389632) = 8389632 ∧
  (∀ i : Fin 8389632, parallel_generatedRoot_41 params i (w_40_1, (w_2_0, (w_40_0, ()))) (w_41_0 i)) ∧
  0 ≤ (1025) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_42 params i (current, (w_38_0, (w_39_0, (w_41_0, ())))) next) (Int.toNat (1025)) w_37_0 w_42_0 ∧
  MxxRuntime.familyGetStatic w_44_0 (0) w_45_0 ∧
  MxxRuntime.familyGetStatic w_50_0 (0) w_51_0 ∧
  MxxRuntime.familyGetStatic w_23_0 (1) w_59_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_74 params i (current, (w_73_0, ())) next) (Int.toNat (2453063738106839040)) w_72_0 w_74_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_76 params i (current, (w_75_0, ())) next) (Int.toNat (2453063738106839040)) w_74_0 w_76_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_78 params i (current, (w_77_0, ())) next) (Int.toNat (2453063738106839040)) w_76_0 w_78_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_80 params i (current, (w_79_0, ())) next) (Int.toNat (2453063738106839040)) w_78_0 w_80_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_82 params i (current, (w_81_0, ())) next) (Int.toNat (2453063738106839040)) w_80_0 w_82_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_84 params i (current, (w_83_0, ())) next) (Int.toNat (2453063738106839040)) w_82_0 w_84_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_86 params i (current, (w_85_0, ())) next) (Int.toNat (2453063738106839040)) w_84_0 w_86_0 ∧
  0 ≤ (2453063738106839040) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Bool) => sequential_generatedRoot_88 params i (current, (w_87_0, ())) next) (Int.toNat (2453063738106839040)) w_86_0 w_88_0 ∧
  0 ≤ (2453063738106839040)

abbrev generatedRoot.body (params : Params) (inputs : (generatedRoot.InputType)) (outputs : Bool) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : Fin 8389632 → Int := inputs.1
  let w_2_0 : Fin 8389632 → Int := inputs.2.1
  let w_4_0 : Fin 1025 → Int := inputs.2.2.1
  let w_5_0 : Fin 1025 → Int := inputs.2.2.2.1
  let w_8_0 : Fin 1 → Int := inputs.2.2.2.2.1
  let w_14_0 : Fin 1 → Int := inputs.2.2.2.2.2.1
  let w_23_0 : Fin 2 → Int := inputs.2.2.2.2.2.2.1
  let w_38_0 : Fin 1025 → Int := inputs.2.2.2.2.2.2.2.1
  let w_39_0 : Fin 1025 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_44_0 : Fin 1 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : Fin 1 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_73_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_77_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_85_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_87_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_89_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_91_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_93_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_95_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_97_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_99_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_101_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_103_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_105_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_107_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_109_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_111_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_113_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_115_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_117_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_119_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_121_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_123_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_125_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_127_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_129_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_131_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_133_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_135_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_137_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_139_0 : Fin 318898285953889075200 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_141_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_143_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_145_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_147_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_149_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_151_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_153_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_155_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_157_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_159_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_161_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_163_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_165_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_167_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_169_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_171_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_173_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_175_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_177_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_179_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_181_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_183_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_185_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_187_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_189_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_191_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_193_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_195_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_197_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_199_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_201_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_203_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_205_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_207_0 : Fin 8519680 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_209_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_211_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_213_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_215_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_217_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_219_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_221_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_223_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_225_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_227_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_229_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_231_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_233_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_235_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_237_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_239_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_241_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_243_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_245_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_247_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_249_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_251_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_253_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_255_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_257_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_259_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_261_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_263_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_265_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_267_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_269_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_271_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_273_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_275_0 : Fin 38598194862489600 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_277_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_279_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_281_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_283_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_285_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_287_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_289_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_291_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_293_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_295_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_297_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_299_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_301_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_303_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_305_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_307_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_309_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_311_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_313_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_315_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_317_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_319_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_321_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_323_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_325_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_327_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_329_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_331_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_333_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_335_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_337_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_339_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_341_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_343_0 : Fin 2607022080 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
  let w_20_0 : Int := 65536
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
  let w_56_0 : Int := 65536
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
  let w_212_0 := witness.w_212_0
  let w_214_0 := witness.w_214_0
  let w_216_0 := witness.w_216_0
  let w_218_0 := witness.w_218_0
  let w_220_0 := witness.w_220_0
  let w_222_0 := witness.w_222_0
  let w_224_0 := witness.w_224_0
  let w_226_0 := witness.w_226_0
  let w_228_0 := witness.w_228_0
  let w_230_0 := witness.w_230_0
  let w_232_0 := witness.w_232_0
  let w_234_0 := witness.w_234_0
  let w_236_0 := witness.w_236_0
  let w_238_0 := witness.w_238_0
  let w_240_0 := witness.w_240_0
  let w_242_0 := witness.w_242_0
  let w_244_0 := witness.w_244_0
  let w_246_0 := witness.w_246_0
  let w_248_0 := witness.w_248_0
  let w_250_0 := witness.w_250_0
  let w_252_0 := witness.w_252_0
  let w_254_0 := witness.w_254_0
  let w_256_0 := witness.w_256_0
  let w_258_0 := witness.w_258_0
  let w_260_0 := witness.w_260_0
  let w_262_0 := witness.w_262_0
  let w_264_0 := witness.w_264_0
  let w_266_0 := witness.w_266_0
  let w_268_0 := witness.w_268_0
  let w_270_0 := witness.w_270_0
  let w_272_0 := witness.w_272_0
  let w_274_0 := witness.w_274_0
  let w_276_0 := witness.w_276_0
  let w_278_0 := witness.w_278_0
  let w_280_0 := witness.w_280_0
  let w_282_0 := witness.w_282_0
  let w_284_0 := witness.w_284_0
  let w_286_0 := witness.w_286_0
  let w_288_0 := witness.w_288_0
  let w_290_0 := witness.w_290_0
  let w_292_0 := witness.w_292_0
  let w_294_0 := witness.w_294_0
  let w_296_0 := witness.w_296_0
  let w_298_0 := witness.w_298_0
  let w_300_0 := witness.w_300_0
  let w_302_0 := witness.w_302_0
  let w_304_0 := witness.w_304_0
  let w_306_0 := witness.w_306_0
  let w_308_0 := witness.w_308_0
  let w_310_0 := witness.w_310_0
  let w_312_0 := witness.w_312_0
  let w_314_0 := witness.w_314_0
  let w_316_0 := witness.w_316_0
  let w_318_0 := witness.w_318_0
  let w_320_0 := witness.w_320_0
  let w_322_0 := witness.w_322_0
  let w_324_0 := witness.w_324_0
  let w_326_0 := witness.w_326_0
  let w_328_0 := witness.w_328_0
  let w_330_0 := witness.w_330_0
  let w_332_0 := witness.w_332_0
  let w_334_0 := witness.w_334_0
  let w_336_0 := witness.w_336_0
  let w_338_0 := witness.w_338_0
  let w_340_0 := witness.w_340_0
  let w_342_0 := witness.w_342_0
  let w_344_0 := witness.w_344_0
  generatedRoot.constraints_0 params w_1_0 w_2_0 w_4_0 w_5_0 w_8_0 w_14_0 w_23_0 w_38_0 w_39_0 w_44_0 w_50_0 w_73_0 w_75_0 w_77_0 w_79_0 w_81_0 w_83_0 w_85_0 w_87_0 w_0_0 w_3_0 w_6_0 w_9_0 w_15_0 w_24_0 w_37_0 w_40_0 w_40_1 w_41_0 w_42_0 w_45_0 w_51_0 w_59_0 w_72_0 w_74_0 w_76_0 w_78_0 w_80_0 w_82_0 w_84_0 w_86_0 w_88_0 ∧
    generatedRoot.constraints_1 params w_89_0 w_91_0 w_93_0 w_95_0 w_97_0 w_99_0 w_101_0 w_103_0 w_105_0 w_107_0 w_109_0 w_111_0 w_113_0 w_115_0 w_117_0 w_119_0 w_88_0 w_90_0 w_92_0 w_94_0 w_96_0 w_98_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_110_0 w_112_0 w_114_0 w_116_0 w_118_0 w_120_0 ∧
    generatedRoot.constraints_2 params w_121_0 w_123_0 w_125_0 w_127_0 w_129_0 w_131_0 w_133_0 w_135_0 w_137_0 w_139_0 w_141_0 w_143_0 w_145_0 w_147_0 w_149_0 w_151_0 w_120_0 w_122_0 w_124_0 w_126_0 w_128_0 w_130_0 w_132_0 w_134_0 w_136_0 w_138_0 w_140_0 w_142_0 w_144_0 w_146_0 w_148_0 w_150_0 w_152_0 ∧
    generatedRoot.constraints_3 params w_153_0 w_155_0 w_157_0 w_159_0 w_161_0 w_163_0 w_165_0 w_167_0 w_169_0 w_171_0 w_173_0 w_175_0 w_177_0 w_179_0 w_181_0 w_183_0 w_152_0 w_154_0 w_156_0 w_158_0 w_160_0 w_162_0 w_164_0 w_166_0 w_168_0 w_170_0 w_172_0 w_174_0 w_176_0 w_178_0 w_180_0 w_182_0 w_184_0 ∧
    generatedRoot.constraints_4 params w_185_0 w_187_0 w_189_0 w_191_0 w_193_0 w_195_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_207_0 w_209_0 w_211_0 w_213_0 w_215_0 w_184_0 w_186_0 w_188_0 w_190_0 w_192_0 w_194_0 w_196_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_208_0 w_210_0 w_212_0 w_214_0 w_216_0 ∧
    generatedRoot.constraints_5 params w_217_0 w_219_0 w_221_0 w_223_0 w_225_0 w_227_0 w_229_0 w_231_0 w_233_0 w_235_0 w_237_0 w_239_0 w_241_0 w_243_0 w_245_0 w_247_0 w_216_0 w_218_0 w_220_0 w_222_0 w_224_0 w_226_0 w_228_0 w_230_0 w_232_0 w_234_0 w_236_0 w_238_0 w_240_0 w_242_0 w_244_0 w_246_0 w_248_0 ∧
    generatedRoot.constraints_6 params w_249_0 w_251_0 w_253_0 w_255_0 w_257_0 w_259_0 w_261_0 w_263_0 w_265_0 w_267_0 w_269_0 w_271_0 w_273_0 w_275_0 w_277_0 w_279_0 w_248_0 w_250_0 w_252_0 w_254_0 w_256_0 w_258_0 w_260_0 w_262_0 w_264_0 w_266_0 w_268_0 w_270_0 w_272_0 w_274_0 w_276_0 w_278_0 w_280_0 ∧
    generatedRoot.constraints_7 params w_281_0 w_283_0 w_285_0 w_287_0 w_289_0 w_291_0 w_293_0 w_295_0 w_297_0 w_299_0 w_301_0 w_303_0 w_305_0 w_307_0 w_309_0 w_311_0 w_280_0 w_282_0 w_284_0 w_286_0 w_288_0 w_290_0 w_292_0 w_294_0 w_296_0 w_298_0 w_300_0 w_302_0 w_304_0 w_306_0 w_308_0 w_310_0 w_312_0 ∧
    generatedRoot.constraints_8 params w_313_0 w_315_0 w_317_0 w_319_0 w_321_0 w_323_0 w_325_0 w_327_0 w_329_0 w_331_0 w_333_0 w_335_0 w_337_0 w_339_0 w_341_0 w_343_0 w_312_0 w_314_0 w_316_0 w_318_0 w_320_0 w_322_0 w_324_0 w_326_0 w_328_0 w_330_0 w_332_0 w_334_0 w_336_0 w_338_0 w_340_0 w_342_0 w_344_0 outputs

def generatedRoot (params : Params) (inputs : (generatedRoot.InputType)) (outputs : Bool) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end W_Requirement_0
