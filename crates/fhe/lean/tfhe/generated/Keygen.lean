import MxxIR
import MxxRuntime
import MxxSecurity.Sampling

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Keygen

structure Params where
  unit : Unit

theorem index_le_720_1024 : 720 ≤ 1024 := by decide

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix 5234636801 1024 1 8640
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix 5234636801 1024 1 12
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix 5234636801 1024 1 6
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix 5234636801 1024 1 1
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix 5846006549229237458670166626046031780481016238081 1024 1 5760
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix 5846006549229237458670166626046031780481016238081 1024 1 1

abbrev parallel_generatedRoot_3.constraints_0 (w_1_0 : Int) (w_2_0 : Int) (outputs : Int) : Prop :=
  w_1_0 ≠ 0 ∧
  outputs = w_2_0

def parallel_generatedRoot_3 (backend : MxxRuntime.BackendContext) (params : Params) (_ : Nat) (inputs : Int) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs
  let w_1_0 : Int := 2
  let w_2_0 : Int := (w_0_0 % w_1_0)
  parallel_generatedRoot_3.constraints_0 w_1_0 w_2_0 outputs

abbrev parallel_generatedRoot_6.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat0) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_4_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat2) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (12 * (Int.ofNat i_0)) ∧
  (12 * (Int.ofNat i_0)) < (12 + (12 * (Int.ofNat i_0))) ∧
  (12 + (12 * (Int.ofNat i_0))) ≤ 8640 ∧
  MxxRuntime.sliceMatrix w_0_0 0 1 (12 * (Int.ofNat i_0)) (12 + (12 * (Int.ofNat i_0))) w_1_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 6 w_4_0 ∧
  MxxRuntime.concatColumns w_5_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (12 * (Int.ofNat i_0)) ∧
  (12 * (Int.ofNat i_0)) < (12 + (12 * (Int.ofNat i_0))) ∧
  (12 + (12 * (Int.ofNat i_0))) ≤ 8640 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (12 * (Int.ofNat i_0)) (12 + (12 * (Int.ofNat i_0))) w_12_0 ∧
  MxxRuntime.concatColumns w_6_0 w_5_0 w_14_0 ∧
  outputs = (w_8_0, (w_15_0, ()))

def parallel_generatedRoot_6 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × Int × mxxMat3 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : Int := inputs.2.1
  let w_9_0 : mxxMat3 := inputs.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.1
  ∃ (w_1_0 : mxxMat1),
    let w_3_0 : mxxMat3 := MxxRuntime.liftInteger w_2_0
    ∃ (w_4_0 : mxxMat2),
      let w_5_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
      let w_6_0 : mxxMat2 := (0 : mxxMat2)
      ∃ (w_7_0 : mxxMat1),
        let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_1_0 w_7_0
        let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_1_0
        ∃ (w_12_0 : mxxMat1),
          let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_10_0 w_12_0
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
            parallel_generatedRoot_6.constraints_0 backend i_0 w_0_0 w_11_0 w_1_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 outputs

abbrev parallel_generatedRoot_8.constraints_0 (i_0 : Nat) (w_0_0 : mxxMat4) (w_1_0 : mxxMat5) (w_2_0 : Fin 1024 → Int) (outputs : Fin 1024 → Int) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 5760 ∧
  MxxRuntime.sliceMatrix w_0_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_1_0 ∧
  MxxRuntime.polynomialValues false w_1_0 w_2_0 ∧
  outputs = w_2_0

def parallel_generatedRoot_8 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : mxxMat4) (outputs : Fin 1024 → Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat4 := inputs
  ∃ (w_1_0 : mxxMat5),
    ∃ (w_2_0 : Fin 1024 → Int),
      parallel_generatedRoot_8.constraints_0 i_0 w_0_0 w_1_0 w_2_0 outputs

abbrev parallel_generatedRoot_10.constraints_0 (w_0_0 : Fin 5760 → Fin 1024 → Int) (w_8_0 : Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Fin 1024 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_7_0 : Int) (w_9_0 : Int) (outputs : Int) : Prop :=
  w_2_0 ≠ 0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5760 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_3_0 w_4_0 ∧
  w_5_0 ≠ 0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_6_0 w_7_0 ∧
  w_8_0 ≠ 0 ∧
  outputs = w_9_0

def parallel_generatedRoot_10 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : (Fin 5760 → Fin 1024 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 5760 → Fin 1024 → Int := inputs.1
  let w_8_0 : Int := inputs.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 : Int := 1024
  let w_3_0 : Int := (w_1_0 / w_2_0)
  ∃ (w_4_0 : Fin 1024 → Int),
    let w_5_0 : Int := 1024
    let w_6_0 : Int := (w_1_0 % w_5_0)
    ∃ (w_7_0 : Int),
      let w_9_0 : Int := (w_7_0 % w_8_0)
      parallel_generatedRoot_10.constraints_0 w_0_0 w_8_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 outputs

abbrev parallel_generatedRoot_37.constraints_1 (w_53_0 : Fin 1024 → Int) (w_36_0 : Int) (w_39_0 : Int) (w_40_0 : Int) (w_42_0 : Int) (w_44_0 : Int) (w_46_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (w_52_0 : Int) (w_54_0 : Int) (w_55_0 : Int) (w_60_0 : Int) (w_62_0 : Int) (w_63_0 : Int) (w_65_0 : Int) (w_66_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_39_0 ∧
  w_39_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_53_0 w_39_0 w_54_0 ∧
  0 ≤ w_36_0 ∧
  w_36_0 < 8 ∧
  8 = 8 ∧
  MxxRuntime.select w_36_0 [w_40_0, w_42_0, w_44_0, w_46_0, w_48_0, w_50_0, w_52_0, w_54_0] w_55_0 ∧
  0 ≤ w_60_0 ∧
  w_60_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_60_0 [w_62_0, w_55_0] w_63_0 ∧
  w_65_0 ≠ 0 ∧
  outputs = w_66_0

abbrev parallel_generatedRoot_37.constraints_0 (w_1_0 : Fin 1024 → Int) (w_37_0 : Fin 1024 → Int) (w_41_0 : Fin 1024 → Int) (w_43_0 : Fin 1024 → Int) (w_45_0 : Fin 1024 → Int) (w_47_0 : Fin 1024 → Int) (w_49_0 : Fin 1024 → Int) (w_51_0 : Fin 1024 → Int) (w_53_0 : Fin 1024 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_6_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_28_0 : Int) (w_31_0 : Int) (w_32_0 : Int) (w_35_0 : Int) (w_36_0 : Int) (w_38_0 : Int) (w_39_0 : Int) (w_40_0 : Int) (w_42_0 : Int) (w_44_0 : Int) (w_46_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (w_52_0 : Int) (w_54_0 : Int) (w_55_0 : Int) (w_60_0 : Int) (w_62_0 : Int) (w_63_0 : Int) (w_65_0 : Int) (w_66_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  w_6_0 ≠ 0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 8 ∧
  8 = 8 ∧
  MxxRuntime.select w_7_0 [w_10_0, w_13_0, w_16_0, w_19_0, w_22_0, w_25_0, w_28_0, w_31_0] w_32_0 ∧
  w_35_0 ≠ 0 ∧
  w_38_0 ≠ 0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_37_0 w_39_0 w_40_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_41_0 w_39_0 w_42_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_39_0 w_44_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_45_0 w_39_0 w_46_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_47_0 w_39_0 w_48_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_49_0 w_39_0 w_50_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_51_0 w_39_0 w_52_0 ∧
  parallel_generatedRoot_37.constraints_1 w_53_0 w_36_0 w_39_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_55_0 w_60_0 w_62_0 w_63_0 w_65_0 w_66_0 outputs

def parallel_generatedRoot_37 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 1024 → Int) × Int × Int × Int × Int × Int × Int × Int × Int × (Fin 1024 → Int) × (Fin 1024 → Int) × (Fin 1024 → Int) × (Fin 1024 → Int) × (Fin 1024 → Int) × (Fin 1024 → Int) × (Fin 1024 → Int) × (Fin 1024 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 1024 → Int := inputs.2.1
  let w_8_0 : Int := inputs.2.2.1
  let w_11_0 : Int := inputs.2.2.2.1
  let w_14_0 : Int := inputs.2.2.2.2.1
  let w_17_0 : Int := inputs.2.2.2.2.2.1
  let w_20_0 : Int := inputs.2.2.2.2.2.2.1
  let w_23_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_26_0 : Int := inputs.2.2.2.2.2.2.2.2.1
  let w_29_0 : Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_37_0 : Fin 1024 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : Fin 1024 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 1024 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_45_0 : Fin 1024 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_47_0 : Fin 1024 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_49_0 : Fin 1024 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : Fin 1024 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : Fin 1024 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 : Int := 8
  let w_4_0 : Int := (w_2_0 / w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : Int := 8
    let w_7_0 : Int := (w_2_0 % w_6_0)
    let w_9_0 : Int := 0
    let w_10_0 : Int := (w_8_0 + w_9_0)
    let w_12_0 : Int := 0
    let w_13_0 : Int := (w_11_0 + w_12_0)
    let w_15_0 : Int := 0
    let w_16_0 : Int := (w_14_0 + w_15_0)
    let w_18_0 : Int := 0
    let w_19_0 : Int := (w_17_0 + w_18_0)
    let w_21_0 : Int := 0
    let w_22_0 : Int := (w_20_0 + w_21_0)
    let w_24_0 : Int := 0
    let w_25_0 : Int := (w_23_0 + w_24_0)
    let w_27_0 : Int := 0
    let w_28_0 : Int := (w_26_0 + w_27_0)
    let w_30_0 : Int := 0
    let w_31_0 : Int := (w_29_0 + w_30_0)
    ∃ (w_32_0 : Int),
      let w_33_0 : Int := (w_5_0 * w_32_0)
      let w_34_0 : Int := (w_0_0 + w_33_0)
      let w_35_0 : Int := 1024
      let w_36_0 : Int := (w_2_0 / w_35_0)
      let w_38_0 : Int := 1024
      let w_39_0 : Int := (w_2_0 % w_38_0)
      ∃ (w_40_0 : Int),
        ∃ (w_42_0 : Int),
          ∃ (w_44_0 : Int),
            ∃ (w_46_0 : Int),
              ∃ (w_48_0 : Int),
                ∃ (w_50_0 : Int),
                  ∃ (w_52_0 : Int),
                    ∃ (w_54_0 : Int),
                      ∃ (w_55_0 : Int),
                        let w_56_0 : Int := 2
                        let w_57_0 : Int := (w_55_0 * w_56_0)
                        let w_58_0 : Int := 5234636801
                        let w_59_0 : Bool := decide (w_57_0 ≤ w_58_0)
                        let w_60_0 : Int := if w_59_0 then 1 else 0
                        let w_61_0 : Int := 5234636801
                        let w_62_0 : Int := (w_55_0 - w_61_0)
                        ∃ (w_63_0 : Int),
                          let w_64_0 : Int := (w_34_0 + w_63_0)
                          let w_65_0 : Int := 4294967296
                          let w_66_0 : Int := (w_64_0 % w_65_0)
                          parallel_generatedRoot_37.constraints_0 w_1_0 w_37_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_10_0 w_13_0 w_16_0 w_19_0 w_22_0 w_25_0 w_28_0 w_31_0 w_32_0 w_35_0 w_36_0 w_38_0 w_39_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_55_0 w_60_0 w_62_0 w_63_0 w_65_0 w_66_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_0 : mxxMat0
  w_1_0 : mxxMat3
  w_2_0 : Fin 1024 → Int
  w_3_0 : Fin 720 → Int
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_6_0 : Fin 720 → mxxMat1
  w_6_1 : Fin 720 → mxxMat1
  w_7_0 : mxxMat4
  w_8_0 : Fin 5760 → Fin 1024 → Int
  w_10_0 : Fin 5898240 → Int
  w_12_0 : Fin 1024 → Int
  w_21_0 : mxxMat3
  w_22_0 : Fin 1024 → Int
  w_23_0 : mxxMat3
  w_24_0 : Fin 1024 → Int
  w_25_0 : mxxMat3
  w_26_0 : Fin 1024 → Int
  w_27_0 : mxxMat3
  w_28_0 : Fin 1024 → Int
  w_29_0 : mxxMat3
  w_30_0 : Fin 1024 → Int
  w_31_0 : mxxMat3
  w_32_0 : Fin 1024 → Int
  w_33_0 : mxxMat3
  w_34_0 : Fin 1024 → Int
  w_35_0 : mxxMat3
  w_36_0 : Fin 1024 → Int
  w_37_0 : Fin 8192 → Int
  w_38_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (params : Params) (w_3_0 : Fin 720 → Int) (w_6_0 : Fin 720 → mxxMat1) (w_6_1 : Fin 720 → mxxMat1) (w_10_0 : Fin 5898240 → Int) (w_11_0 : Fin 8192 → Int) (w_12_0 : Fin 1024 → Int) (w_13_0 : Int) (w_14_0 : Int) (w_15_0 : Int) (w_16_0 : Int) (w_17_0 : Int) (w_18_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_22_0 : Fin 1024 → Int) (w_24_0 : Fin 1024 → Int) (w_26_0 : Fin 1024 → Int) (w_28_0 : Fin 1024 → Int) (w_30_0 : Fin 1024 → Int) (w_32_0 : Fin 1024 → Int) (w_34_0 : Fin 1024 → Int) (w_36_0 : Fin 1024 → Int) (w_37_0 : Fin 8192 → Int) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_41_0 : mxxMat3) (outputs : (Fin 720 → mxxMat1) × (Fin 720 → mxxMat1) × (Fin 5898240 → Int) × (Fin 8192 → Int) × mxxMat3 × mxxMat3 × (Fin 5898240 → Int) × (Fin 8192 → Int) × (Fin 720 → Int) × Unit) : Prop :=
  (∀ i : Fin 8192, parallel_generatedRoot_37 backend params i ((w_11_0 i), (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_18_0, (w_19_0, (w_20_0, (w_22_0, (w_24_0, (w_26_0, (w_28_0, (w_30_0, (w_32_0, (w_34_0, (w_36_0, ())))))))))))))))))) (w_37_0 i)) ∧
  MxxRuntime.uniformResidueSample w_38_0 ∧
  MxxRuntime.gaussianSample ((156/1)) (2496) w_40_0 ∧
  outputs = (w_6_0, (w_6_1, (w_10_0, (w_37_0, (w_38_0, (w_41_0, (w_10_0, (w_37_0, (w_3_0, ())))))))))

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat3) (w_2_0 : Fin 1024 → Int) (w_3_0 : Fin 720 → Int) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_6_0 : Fin 720 → mxxMat1) (w_6_1 : Fin 720 → mxxMat1) (w_7_0 : mxxMat4) (w_8_0 : Fin 5760 → Fin 1024 → Int) (w_9_0 : Int) (w_10_0 : Fin 5898240 → Int) (w_11_0 : Fin 8192 → Int) (w_12_0 : Fin 1024 → Int) (w_13_0 : Int) (w_14_0 : Int) (w_15_0 : Int) (w_16_0 : Int) (w_17_0 : Int) (w_18_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_21_0 : mxxMat3) (w_22_0 : Fin 1024 → Int) (w_23_0 : mxxMat3) (w_24_0 : Fin 1024 → Int) (w_25_0 : mxxMat3) (w_26_0 : Fin 1024 → Int) (w_27_0 : mxxMat3) (w_28_0 : Fin 1024 → Int) (w_29_0 : mxxMat3) (w_30_0 : Fin 1024 → Int) (w_31_0 : mxxMat3) (w_32_0 : Fin 1024 → Int) (w_33_0 : mxxMat3) (w_34_0 : Fin 1024 → Int) (w_35_0 : mxxMat3) (w_36_0 : Fin 1024 → Int) (w_37_0 : Fin 8192 → Int) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_41_0 : mxxMat3) (outputs : (Fin 720 → mxxMat1) × (Fin 720 → mxxMat1) × (Fin 5898240 → Int) × (Fin 8192 → Int) × mxxMat3 × mxxMat3 × (Fin 5898240 → Int) × (Fin 8192 → Int) × (Fin 720 → Int) × Unit) : Prop :=
  MxxRuntime.uniformResidueSample w_0_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_1_0 ∧
  MxxRuntime.polynomialValues false w_1_0 w_2_0 ∧
  (720) = 720 ∧
  (∀ i : Fin 720, parallel_generatedRoot_3 backend params i (w_2_0 (Fin.castLE index_le_720_1024 i)) (w_3_0 i)) ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_4_0 ∧
  MxxRuntime.gaussianSample ((156/1)) (2496) w_5_0 ∧
  (720) = 720 ∧
  (∀ i : Fin 720, parallel_generatedRoot_6 backend params i (w_0_0, ((w_3_0 i), (w_4_0, (w_5_0, ())))) ((w_6_0 i), ((w_6_1 i), ()))) ∧
  MxxRuntime.uniformResidueSample w_7_0 ∧
  (5760) = 5760 ∧
  (∀ i : Fin 5760, parallel_generatedRoot_8 backend params i w_7_0 (w_8_0 i)) ∧
  (5898240) = 5898240 ∧
  (∀ i : Fin 5898240, parallel_generatedRoot_10 backend params i (w_8_0, (w_9_0, ())) (w_10_0 i)) ∧
  MxxRuntime.polynomialValues false w_4_0 w_12_0 ∧
  MxxRuntime.gaussianSample ((32768/1)) (524288) w_21_0 ∧
  MxxRuntime.polynomialValues false w_21_0 w_22_0 ∧
  MxxRuntime.gaussianSample ((32768/1)) (524288) w_23_0 ∧
  MxxRuntime.polynomialValues false w_23_0 w_24_0 ∧
  MxxRuntime.gaussianSample ((32768/1)) (524288) w_25_0 ∧
  MxxRuntime.polynomialValues false w_25_0 w_26_0 ∧
  MxxRuntime.gaussianSample ((32768/1)) (524288) w_27_0 ∧
  MxxRuntime.polynomialValues false w_27_0 w_28_0 ∧
  MxxRuntime.gaussianSample ((32768/1)) (524288) w_29_0 ∧
  MxxRuntime.polynomialValues false w_29_0 w_30_0 ∧
  MxxRuntime.gaussianSample ((32768/1)) (524288) w_31_0 ∧
  MxxRuntime.polynomialValues false w_31_0 w_32_0 ∧
  MxxRuntime.gaussianSample ((32768/1)) (524288) w_33_0 ∧
  MxxRuntime.polynomialValues false w_33_0 w_34_0 ∧
  MxxRuntime.gaussianSample ((32768/1)) (524288) w_35_0 ∧
  MxxRuntime.polynomialValues false w_35_0 w_36_0 ∧
  (8192) = 8192 ∧
  generatedRoot.constraints_1 backend params w_3_0 w_6_0 w_6_1 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_37_0 w_38_0 w_40_0 w_41_0 outputs

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (params : Params) (_ : Unit) (outputs : (Fin 720 → mxxMat1) × (Fin 720 → mxxMat1) × (Fin 5898240 → Int) × (Fin 8192 → Int) × mxxMat3 × mxxMat3 × (Fin 5898240 → Int) × (Fin 8192 → Int) × (Fin 720 → Int) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 := witness.w_0_0
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Int := 4294967296
  let w_10_0 := witness.w_10_0
  let w_11_0 : Fin 8192 → Int := MxxRuntime.intMatrixVectorProduct false w_10_0 w_3_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : Int := 65536
  let w_14_0 : Int := 262144
  let w_15_0 : Int := 1048576
  let w_16_0 : Int := 4194304
  let w_17_0 : Int := 16777216
  let w_18_0 : Int := 67108864
  let w_19_0 : Int := 268435456
  let w_20_0 : Int := 1073741824
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_38_0 w_4_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_40_0
  generatedRoot.constraints_0 backend params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_40_0 w_41_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (params : Params) (_ : Unit) (outputs : (Fin 720 → mxxMat1) × (Fin 720 → mxxMat1) × (Fin 5898240 → Int) × (Fin 8192 → Int) × mxxMat3 × mxxMat3 × (Fin 5898240 → Int) × (Fin 8192 → Int) × (Fin 720 → Int) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend params () outputs witness

noncomputable def generatedRoot.comp (backend : MxxRuntime.BackendContext) (params : Params) (_ : Unit) : MxxSecurity.Comp ((Fin 720 → mxxMat1) × (Fin 720 → mxxMat1) × (Fin 5898240 → Int) × (Fin 8192 → Int) × mxxMat3 × mxxMat3 × (Fin 5898240 → Int) × (Fin 8192 → Int) × (Fin 720 → Int) × Unit) := do
  let w_0_0 : mxxMat0 ← MxxSecurity.sampleResidue
  let w_1_0 : mxxMat3 ← MxxSecurity.sampleInterval (0) (1)
  let w_4_0 : mxxMat3 ← MxxSecurity.sampleInterval (0) (1)
  let w_5_0 : mxxMat0 ← MxxSecurity.sampleGaussian ((156/1)) (2496)
  let w_7_0 : mxxMat4 ← MxxSecurity.sampleResidue
  let w_21_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((32768/1)) (524288)
  let w_23_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((32768/1)) (524288)
  let w_25_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((32768/1)) (524288)
  let w_27_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((32768/1)) (524288)
  let w_29_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((32768/1)) (524288)
  let w_31_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((32768/1)) (524288)
  let w_33_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((32768/1)) (524288)
  let w_35_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((32768/1)) (524288)
  let w_38_0 : mxxMat3 ← MxxSecurity.sampleResidue
  let w_40_0 : mxxMat3 ← MxxSecurity.sampleGaussian ((156/1)) (2496)
  pure (MxxSecurity.determined fun outputs : (Fin 720 → mxxMat1) × (Fin 720 → mxxMat1) × (Fin 5898240 → Int) × (Fin 8192 → Int) × mxxMat3 × mxxMat3 × (Fin 5898240 → Int) × (Fin 8192 → Int) × (Fin 720 → Int) × Unit ↦
    ∃ witness : generatedRoot.Witness, witness.w_0_0 = w_0_0 ∧ witness.w_1_0 = w_1_0 ∧ witness.w_4_0 = w_4_0 ∧ witness.w_5_0 = w_5_0 ∧ witness.w_7_0 = w_7_0 ∧ witness.w_21_0 = w_21_0 ∧ witness.w_23_0 = w_23_0 ∧ witness.w_25_0 = w_25_0 ∧ witness.w_27_0 = w_27_0 ∧ witness.w_29_0 = w_29_0 ∧ witness.w_31_0 = w_31_0 ∧ witness.w_33_0 = w_33_0 ∧ witness.w_35_0 = w_35_0 ∧ witness.w_38_0 = w_38_0 ∧ witness.w_40_0 = w_40_0 ∧ generatedRoot.body backend params () outputs witness)


end Keygen
