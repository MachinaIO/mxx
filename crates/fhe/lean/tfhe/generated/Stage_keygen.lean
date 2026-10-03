import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_keygen

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_2.constraints_0 (w_1_0 : Int) (w_2_0 : Int) (outputs : Int) : Prop :=
  w_1_0 ≠ 0 ∧
  outputs = w_2_0

def parallel_generatedRoot_2 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (inputs : Int) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs
  let w_1_0 : Int := 2
  let w_2_0 : Int := (w_0_0 % w_1_0)
  parallel_generatedRoot_2.constraints_0 w_1_0 w_2_0 outputs

abbrev parallel_generatedRoot_4.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) (w_3_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 8) (w_4_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 8) (w_5_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 8) (w_6_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) (w_7_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) (w_10_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) (w_12_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) (w_13_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) (outputs : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16 × Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16 × Unit) : Prop :=
  MxxRuntime.uniformResidueSample w_0_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 256 8 w_3_0 ∧
  MxxRuntime.concatColumns w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gaussianSample ((156/1)) (2496) w_10_0 ∧
  MxxRuntime.concatColumns w_5_0 w_4_0 w_12_0 ∧
  outputs = (w_7_0, (w_13_0, ()))

def parallel_generatedRoot_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (inputs : Int × Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1 × Unit) (outputs : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16 × Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Int := inputs.1
  let w_8_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1 := inputs.2.1
  ∃ (w_0_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16),
    let w_2_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1 := MxxRuntime.liftInteger w_1_0
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 8),
      let w_4_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 8 := MxxRuntime.matrixMulScalarLeft w_2_0 w_3_0
      let w_5_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 8 := (0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 8)
      ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16),
        let w_7_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16 := MxxRuntime.matrixAdd w_0_0 w_6_0
        let w_9_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16 := MxxRuntime.matrixMulScalarLeft w_8_0 w_0_0
        ∃ (w_10_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16),
          let w_11_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16 := MxxRuntime.matrixAdd w_9_0 w_10_0
          ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16),
            let w_13_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16 := MxxRuntime.matrixAdd w_11_0 w_12_0
            parallel_generatedRoot_4.constraints_0 backend w_0_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_10_0 w_12_0 w_13_0 outputs

abbrev parallel_generatedRoot_33.constraints_1 (w_53_0 : Fin 2048 → Int) (w_36_0 : Int) (w_39_0 : Int) (w_40_0 : Int) (w_42_0 : Int) (w_44_0 : Int) (w_46_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (w_52_0 : Int) (w_54_0 : Int) (w_55_0 : Int) (w_60_0 : Int) (w_62_0 : Int) (w_63_0 : Int) (w_65_0 : Int) (w_66_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_39_0 ∧
  w_39_0 < 2048 ∧
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

abbrev parallel_generatedRoot_33.constraints_0 (w_1_0 : Fin 2048 → Int) (w_37_0 : Fin 2048 → Int) (w_41_0 : Fin 2048 → Int) (w_43_0 : Fin 2048 → Int) (w_45_0 : Fin 2048 → Int) (w_47_0 : Fin 2048 → Int) (w_49_0 : Fin 2048 → Int) (w_51_0 : Fin 2048 → Int) (w_53_0 : Fin 2048 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (w_6_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_13_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_22_0 : Int) (w_25_0 : Int) (w_28_0 : Int) (w_31_0 : Int) (w_32_0 : Int) (w_35_0 : Int) (w_36_0 : Int) (w_38_0 : Int) (w_39_0 : Int) (w_40_0 : Int) (w_42_0 : Int) (w_44_0 : Int) (w_46_0 : Int) (w_48_0 : Int) (w_50_0 : Int) (w_52_0 : Int) (w_54_0 : Int) (w_55_0 : Int) (w_60_0 : Int) (w_62_0 : Int) (w_63_0 : Int) (w_65_0 : Int) (w_66_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 2048 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  w_6_0 ≠ 0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 8 ∧
  8 = 8 ∧
  MxxRuntime.select w_7_0 [w_10_0, w_13_0, w_16_0, w_19_0, w_22_0, w_25_0, w_28_0, w_31_0] w_32_0 ∧
  w_35_0 ≠ 0 ∧
  w_38_0 ≠ 0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 2048 ∧
  MxxRuntime.familyGetDynamic w_37_0 w_39_0 w_40_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 2048 ∧
  MxxRuntime.familyGetDynamic w_41_0 w_39_0 w_42_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 2048 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_39_0 w_44_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 2048 ∧
  MxxRuntime.familyGetDynamic w_45_0 w_39_0 w_46_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 2048 ∧
  MxxRuntime.familyGetDynamic w_47_0 w_39_0 w_48_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 2048 ∧
  MxxRuntime.familyGetDynamic w_49_0 w_39_0 w_50_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 2048 ∧
  MxxRuntime.familyGetDynamic w_51_0 w_39_0 w_52_0 ∧
  parallel_generatedRoot_33.constraints_1 w_53_0 w_36_0 w_39_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_55_0 w_60_0 w_62_0 w_63_0 w_65_0 w_66_0 outputs

def parallel_generatedRoot_33 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Int × (Fin 2048 → Int) × Int × Int × Int × Int × Int × Int × Int × Int × (Fin 2048 → Int) × (Fin 2048 → Int) × (Fin 2048 → Int) × (Fin 2048 → Int) × (Fin 2048 → Int) × (Fin 2048 → Int) × (Fin 2048 → Int) × (Fin 2048 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 2048 → Int := inputs.2.1
  let w_8_0 : Int := inputs.2.2.1
  let w_11_0 : Int := inputs.2.2.2.1
  let w_14_0 : Int := inputs.2.2.2.2.1
  let w_17_0 : Int := inputs.2.2.2.2.2.1
  let w_20_0 : Int := inputs.2.2.2.2.2.2.1
  let w_23_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_26_0 : Int := inputs.2.2.2.2.2.2.2.2.1
  let w_29_0 : Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_37_0 : Fin 2048 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : Fin 2048 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 2048 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_45_0 : Fin 2048 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_47_0 : Fin 2048 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_49_0 : Fin 2048 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : Fin 2048 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : Fin 2048 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
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
      let w_35_0 : Int := 2048
      let w_36_0 : Int := (w_2_0 / w_35_0)
      let w_38_0 : Int := 2048
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
                        let w_58_0 : Int := 4611255024196841473
                        let w_59_0 : Bool := decide (w_57_0 ≤ w_58_0)
                        let w_60_0 : Int := if w_59_0 then 1 else 0
                        let w_61_0 : Int := 4611255024196841473
                        let w_62_0 : Int := (w_55_0 - w_61_0)
                        ∃ (w_63_0 : Int),
                          let w_64_0 : Int := (w_34_0 + w_63_0)
                          let w_65_0 : Int := 4294967296
                          let w_66_0 : Int := (w_64_0 % w_65_0)
                          parallel_generatedRoot_33.constraints_0 w_1_0 w_37_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_10_0 w_13_0 w_16_0 w_19_0 w_22_0 w_25_0 w_28_0 w_31_0 w_32_0 w_35_0 w_36_0 w_38_0 w_39_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_55_0 w_60_0 w_62_0 w_63_0 w_65_0 w_66_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_1_0 : Fin 2048 → Int
  w_2_0 : Fin 630 → Int
  w_3_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_4_0 : Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16
  w_4_1 : Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16
  w_6_0 : Fin 10321920 → Int
  w_8_0 : Fin 2048 → Int
  w_17_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_18_0 : Fin 2048 → Int
  w_19_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_20_0 : Fin 2048 → Int
  w_21_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_22_0 : Fin 2048 → Int
  w_23_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_24_0 : Fin 2048 → Int
  w_25_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_26_0 : Fin 2048 → Int
  w_27_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_28_0 : Fin 2048 → Int
  w_29_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_30_0 : Fin 2048 → Int
  w_31_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_32_0 : Fin 2048 → Int
  w_33_0 : Fin 16384 → Int

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_0_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_1_0 : Fin 2048 → Int) (w_2_0 : Fin 630 → Int) (w_3_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_4_0 : Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) (w_4_1 : Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) (w_6_0 : Fin 10321920 → Int) (w_7_0 : Fin 16384 → Int) (w_8_0 : Fin 2048 → Int) (w_9_0 : Int) (w_10_0 : Int) (w_11_0 : Int) (w_12_0 : Int) (w_13_0 : Int) (w_14_0 : Int) (w_15_0 : Int) (w_16_0 : Int) (w_17_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_18_0 : Fin 2048 → Int) (w_19_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_20_0 : Fin 2048 → Int) (w_21_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_22_0 : Fin 2048 → Int) (w_23_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_24_0 : Fin 2048 → Int) (w_25_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_26_0 : Fin 2048 → Int) (w_27_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_28_0 : Fin 2048 → Int) (w_29_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_30_0 : Fin 2048 → Int) (w_31_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_32_0 : Fin 2048 → Int) (w_33_0 : Fin 16384 → Int) (outputs : (Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) × (Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) × (Fin 10321920 → Int) × (Fin 16384 → Int) × (Fin 630 → Int) × Unit) : Prop :=
  MxxRuntime.uniformIntervalSample (0) (1) w_0_0 ∧
  MxxRuntime.polynomialValues false w_0_0 w_1_0 ∧
  (630) = 630 ∧
  (∀ i : Fin 630, parallel_generatedRoot_2 backend hashModel params i (w_1_0 ⟨i.val, by omega⟩) (w_2_0 i)) ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_3_0 ∧
  (630) = 630 ∧
  (∀ i : Fin 630, parallel_generatedRoot_4 backend hashModel params i ((w_2_0 i), (w_3_0, ())) ((w_4_0 i), ((w_4_1 i), ()))) ∧
  MxxRuntime.hashIntFamily (hashModel) (4294967296) ([109, 120, 120, 47, 104, 97, 115, 104, 45, 105, 110, 116, 45, 102, 97, 109, 105, 108, 121, 47, 118, 49, 0, 116, 102, 104, 101, 47, 107, 101, 121, 103, 101, 110, 47, 107, 115, 107, 45, 97, 47, 118, 49]) ([]) (w_5_0) w_6_0 ∧
  MxxRuntime.polynomialValues false w_3_0 w_8_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_17_0 ∧
  MxxRuntime.polynomialValues false w_17_0 w_18_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_19_0 ∧
  MxxRuntime.polynomialValues false w_19_0 w_20_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_21_0 ∧
  MxxRuntime.polynomialValues false w_21_0 w_22_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_23_0 ∧
  MxxRuntime.polynomialValues false w_23_0 w_24_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_25_0 ∧
  MxxRuntime.polynomialValues false w_25_0 w_26_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_27_0 ∧
  MxxRuntime.polynomialValues false w_27_0 w_28_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_29_0 ∧
  MxxRuntime.polynomialValues false w_29_0 w_30_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_31_0 ∧
  MxxRuntime.polynomialValues false w_31_0 w_32_0 ∧
  (16384) = 16384 ∧
  (∀ i : Fin 16384, parallel_generatedRoot_33 backend hashModel params i ((w_7_0 i), (w_8_0, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_18_0, (w_20_0, (w_22_0, (w_24_0, (w_26_0, (w_28_0, (w_30_0, (w_32_0, ())))))))))))))))))) (w_33_0 i)) ∧
  outputs = (w_4_0, (w_4_1, (w_6_0, (w_33_0, (w_2_0, ())))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray) (outputs : (Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) × (Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) × (Fin 10321920 → Int) × (Fin 16384 → Int) × (Fin 630 → Int) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_5_0 : ByteArray := inputs
  let w_0_0 := witness.w_0_0
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_6_0 := witness.w_6_0
  let w_7_0 : Fin 16384 → Int := MxxRuntime.intMatrixVectorProduct false w_6_0 w_2_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Int := 65536
  let w_10_0 : Int := 262144
  let w_11_0 : Int := 1048576
  let w_12_0 : Int := 4194304
  let w_13_0 : Int := 16777216
  let w_14_0 : Int := 67108864
  let w_15_0 : Int := 268435456
  let w_16_0 : Int := 1073741824
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
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
  generatedRoot.constraints_0 backend hashModel params w_5_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray) (outputs : (Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) × (Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) × (Fin 10321920 → Int) × (Fin 16384 → Int) × (Fin 630 → Int) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end Stage_keygen
