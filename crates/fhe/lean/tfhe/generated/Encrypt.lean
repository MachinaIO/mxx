import MxxIR
import MxxRuntime
import MxxSecurity.Sampling

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Encrypt

structure Params where
  unit : Unit

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix 5234636801 1024 1 1

abbrev parallel_generatedRoot_8.constraints_0 (w_0_0 : Fin 1024 → Int) (w_23_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : Int) (w_21_0 : Int) (w_24_0 : Int) (outputs : Int) : Prop :=
  w_4_0 ≠ 0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_5_0 w_6_0 ∧
  w_15_0 ≠ 0 ∧
  w_21_0 ≠ 0 ∧
  w_23_0 ≠ 0 ∧
  outputs = w_24_0

def parallel_generatedRoot_8 (params : Params) (i_0 : Nat) (inputs : (Fin 1024 → Int) × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Fin 1024 → Int := inputs.1
  let w_23_0 : Int := inputs.2.1
  let w_1_0 : Int := 1024
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 : Int := (w_1_0 - w_2_0)
  let w_4_0 : Int := 1024
  let w_5_0 : Int := (w_3_0 % w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 0
    let w_8_0 : Bool := decide (w_2_0 = w_7_0)
    let w_9_0 : Int := if w_8_0 then 1 else 0
    let w_10_0 : Int := 2
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : Int := 1
    let w_13_0 : Int := (w_11_0 - w_12_0)
    let w_14_0 : Int := (w_6_0 * w_13_0)
    let w_15_0 : Int := 5234636801
    let w_16_0 : Int := (w_14_0 % w_15_0)
    let w_17_0 : Int := 4294967296
    let w_18_0 : Int := (w_16_0 * w_17_0)
    let w_19_0 : Int := 2617318400
    let w_20_0 : Int := (w_18_0 + w_19_0)
    let w_21_0 : Int := 5234636801
    let w_22_0 : Int := (w_20_0 / w_21_0)
    let w_24_0 : Int := (w_22_0 % w_23_0)
    parallel_generatedRoot_8.constraints_0 w_0_0 w_23_0 w_4_0 w_5_0 w_6_0 w_15_0 w_21_0 w_24_0 outputs

abbrev parallel_generatedRoot_17.constraints_0 (w_0_0 : Fin 1024 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (w_7_0 : Int) (w_9_0 : Int) (w_11_0 : Int) (w_12_0 : Int) (w_15_0 : Int) (w_18_0 : Int) (w_21_0 : Int) (w_24_0 : Int) (w_27_0 : Int) (w_30_0 : Int) (w_33_0 : Int) (w_36_0 : Int) (w_37_0 : Int) (w_39_0 : Int) (w_40_0 : Int) (outputs : Int) : Prop :=
  w_2_0 ≠ 0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 1024 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_3_0 w_4_0 ∧
  w_7_0 ≠ 0 ∧
  w_9_0 ≠ 0 ∧
  w_11_0 ≠ 0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  8 = 8 ∧
  MxxRuntime.select w_12_0 [w_15_0, w_18_0, w_21_0, w_24_0, w_27_0, w_30_0, w_33_0, w_36_0] w_37_0 ∧
  w_37_0 ≠ 0 ∧
  w_39_0 ≠ 0 ∧
  outputs = w_40_0

def parallel_generatedRoot_17 (params : Params) (i_0 : Nat) (inputs : (Fin 1024 → Int) × Int × Int × Int × Int × Int × Int × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Fin 1024 → Int := inputs.1
  let w_13_0 : Int := inputs.2.1
  let w_16_0 : Int := inputs.2.2.1
  let w_19_0 : Int := inputs.2.2.2.1
  let w_22_0 : Int := inputs.2.2.2.2.1
  let w_25_0 : Int := inputs.2.2.2.2.2.1
  let w_28_0 : Int := inputs.2.2.2.2.2.2.1
  let w_31_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_34_0 : Int := inputs.2.2.2.2.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 : Int := 8
  let w_3_0 : Int := (w_1_0 / w_2_0)
  ∃ (w_4_0 : Int),
    let w_5_0 : Int := 32768
    let w_6_0 : Int := (w_4_0 + w_5_0)
    let w_7_0 : Int := 65536
    let w_8_0 : Int := (w_6_0 / w_7_0)
    let w_9_0 : Int := 65536
    let w_10_0 : Int := (w_8_0 % w_9_0)
    let w_11_0 : Int := 8
    let w_12_0 : Int := (w_1_0 % w_11_0)
    let w_14_0 : Int := 0
    let w_15_0 : Int := (w_13_0 + w_14_0)
    let w_17_0 : Int := 0
    let w_18_0 : Int := (w_16_0 + w_17_0)
    let w_20_0 : Int := 0
    let w_21_0 : Int := (w_19_0 + w_20_0)
    let w_23_0 : Int := 0
    let w_24_0 : Int := (w_22_0 + w_23_0)
    let w_26_0 : Int := 0
    let w_27_0 : Int := (w_25_0 + w_26_0)
    let w_29_0 : Int := 0
    let w_30_0 : Int := (w_28_0 + w_29_0)
    let w_32_0 : Int := 0
    let w_33_0 : Int := (w_31_0 + w_32_0)
    let w_35_0 : Int := 0
    let w_36_0 : Int := (w_34_0 + w_35_0)
    ∃ (w_37_0 : Int),
      let w_38_0 : Int := (w_10_0 / w_37_0)
      let w_39_0 : Int := 4
      let w_40_0 : Int := (w_38_0 % w_39_0)
      parallel_generatedRoot_17.constraints_0 w_0_0 w_2_0 w_3_0 w_4_0 w_7_0 w_9_0 w_11_0 w_12_0 w_15_0 w_18_0 w_21_0 w_24_0 w_27_0 w_30_0 w_33_0 w_36_0 w_37_0 w_39_0 w_40_0 outputs

abbrev parallel_generatedRoot_20.constraints_0 (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  outputs = w_4_0

def parallel_generatedRoot_20 (params : Params) (_ : Nat) (inputs : Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_1_0 : Int := inputs.1
  let w_3_0 : Int := inputs.2.1
  let w_0_0 : Int := 0
  let w_2_0 : Int := (w_0_0 - w_1_0)
  let w_4_0 : Int := (w_2_0 % w_3_0)
  parallel_generatedRoot_20.constraints_0 w_3_0 w_4_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : mxxMat0
  w_4_0 : mxxMat0
  w_6_0 : Fin 1024 → Int
  w_8_0 : Fin 1024 → Int
  w_17_0 : Fin 8192 → Int
  w_20_0 : Fin 720 → Int
  w_23_0 : mxxMat0
  w_34_0 : Fin 1024 → Int
  w_35_0 : Int
  w_47_0 : Int

abbrev generatedRoot.constraints_0 (params : Params) (w_1_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat0) (w_6_0 : Fin 1024 → Int) (w_7_0 : Int) (w_8_0 : Fin 1024 → Int) (w_9_0 : Int) (w_10_0 : Int) (w_11_0 : Int) (w_12_0 : Int) (w_13_0 : Int) (w_14_0 : Int) (w_15_0 : Int) (w_16_0 : Int) (w_17_0 : Fin 8192 → Int) (w_18_0 : Fin 720 → Int) (w_19_0 : Int) (w_20_0 : Fin 720 → Int) (w_23_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Fin 1024 → Int) (w_35_0 : Int) (w_36_0 : Int) (w_42_0 : Int) (w_46_0 : Fin 1 → Int) (w_47_0 : Int) (w_49_0 : Int) (outputs : (Fin 720 → Int) × Int × Unit) : Prop :=
  MxxRuntime.uniformIntervalSample (0) (1) w_1_0 ∧
  MxxRuntime.gaussianSample ((156/1)) (2496) w_4_0 ∧
  MxxRuntime.polynomialValues false w_5_0 w_6_0 ∧
  (1024) = 1024 ∧
  (∀ i : Fin 1024, parallel_generatedRoot_8 params i (w_6_0, (w_7_0, ())) (w_8_0 i)) ∧
  (8192) = 8192 ∧
  (∀ i : Fin 8192, parallel_generatedRoot_17 params i (w_8_0, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, ()))))))))) (w_17_0 i)) ∧
  (720) = 720 ∧
  (∀ i : Fin 720, parallel_generatedRoot_20 params i ((w_18_0 i), (w_19_0, ())) (w_20_0 i)) ∧
  MxxRuntime.gaussianSample ((156/1)) (2496) w_23_0 ∧
  MxxRuntime.polynomialValues false w_33_0 w_34_0 ∧
  MxxRuntime.familyGetStatic w_34_0 (0) w_35_0 ∧
  w_36_0 ≠ 0 ∧
  w_42_0 ≠ 0 ∧
  w_7_0 ≠ 0 ∧
  MxxRuntime.familyGetStatic w_46_0 (0) w_47_0 ∧
  w_19_0 ≠ 0 ∧
  outputs = (w_20_0, (w_49_0, ()))

abbrev generatedRoot.body (params : Params) (inputs : (Fin 5898240 → Int) × mxxMat0 × mxxMat0 × Int × (Fin 8192 → Int) × Unit) (outputs : (Fin 720 → Int) × Int × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Fin 5898240 → Int := inputs.1
  let w_2_0 : mxxMat0 := inputs.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.1
  let w_25_0 : Int := inputs.2.2.2.1
  let w_45_0 : Fin 8192 → Int := inputs.2.2.2.2.1
  let w_1_0 := witness.w_1_0
  let w_3_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_1_0 w_2_0
  let w_4_0 := witness.w_4_0
  let w_5_0 : mxxMat0 := MxxRuntime.matrixAdd w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : Int := 4294967296
  let w_8_0 := witness.w_8_0
  let w_9_0 : Int := 1
  let w_10_0 : Int := 4
  let w_11_0 : Int := 16
  let w_12_0 : Int := 64
  let w_13_0 : Int := 256
  let w_14_0 : Int := 1024
  let w_15_0 : Int := 4096
  let w_16_0 : Int := 16384
  let w_17_0 := witness.w_17_0
  let w_18_0 : Fin 720 → Int := MxxRuntime.intMatrixVectorProduct true w_0_0 w_17_0
  let w_19_0 : Int := 4294967296
  let w_20_0 := witness.w_20_0
  let w_22_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_1_0 w_21_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_26_0 : Int := 2
  let w_27_0 : Int := (w_25_0 * w_26_0)
  let w_28_0 : Int := 1
  let w_29_0 : Int := (w_27_0 - w_28_0)
  let w_30_0 : Int := 654329600
  let w_31_0 : Int := (w_29_0 * w_30_0)
  let w_32_0 : mxxMat0 := MxxRuntime.liftInteger w_31_0
  let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_24_0 w_32_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : Int := 5234636801
  let w_37_0 : Int := (w_35_0 % w_36_0)
  let w_38_0 : Int := 4294967296
  let w_39_0 : Int := (w_37_0 * w_38_0)
  let w_40_0 : Int := 2617318400
  let w_41_0 : Int := (w_39_0 + w_40_0)
  let w_42_0 : Int := 5234636801
  let w_43_0 : Int := (w_41_0 / w_42_0)
  let w_44_0 : Int := (w_43_0 % w_7_0)
  let w_46_0 : Fin 1 → Int := MxxRuntime.intMatrixVectorProduct true w_45_0 w_17_0
  let w_47_0 := witness.w_47_0
  let w_48_0 : Int := (w_44_0 - w_47_0)
  let w_49_0 : Int := (w_48_0 % w_19_0)
  generatedRoot.constraints_0 params w_1_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_23_0 w_33_0 w_34_0 w_35_0 w_36_0 w_42_0 w_46_0 w_47_0 w_49_0 outputs

def generatedRoot (params : Params) (inputs : (Fin 5898240 → Int) × mxxMat0 × mxxMat0 × Int × (Fin 8192 → Int) × Unit) (outputs : (Fin 720 → Int) × Int × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness

noncomputable def generatedRoot.comp (params : Params) (inputs : (Fin 5898240 → Int) × mxxMat0 × mxxMat0 × Int × (Fin 8192 → Int) × Unit) : MxxSecurity.Comp ((Fin 720 → Int) × Int × Unit) := do
  let w_1_0 : mxxMat0 ← MxxSecurity.sampleInterval (0) (1)
  let w_4_0 : mxxMat0 ← MxxSecurity.sampleGaussian ((156/1)) (2496)
  let w_23_0 : mxxMat0 ← MxxSecurity.sampleGaussian ((156/1)) (2496)
  pure (MxxSecurity.determined fun outputs : (Fin 720 → Int) × Int × Unit ↦
    ∃ witness : generatedRoot.Witness, witness.w_1_0 = w_1_0 ∧ witness.w_4_0 = w_4_0 ∧ witness.w_23_0 = w_23_0 ∧ generatedRoot.body params inputs outputs witness)


end Encrypt
