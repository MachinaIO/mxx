import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_zero

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Fin 1 → mxxMat0
  w_4_0 : Fin 1 → mxxMat0
  w_6_0 : Fin 1 → mxxMat0
  w_8_0 : Fin 1 → mxxMat0
  w_10_0 : Fin 1 → mxxMat0
  w_12_0 : Fin 1 → mxxMat0
  w_14_0 : Fin 1 → mxxMat0
  w_16_0 : Fin 1 → mxxMat0
  w_18_0 : Fin 1 → mxxMat0
  w_20_0 : Fin 1 → mxxMat0
  w_22_0 : Fin 1 → mxxMat0
  w_24_0 : Fin 1 → mxxMat0
  w_26_0 : Fin 1 → mxxMat0
  w_28_0 : Fin 1 → mxxMat0
  w_37_0 : Fin 8 → mxxMat0
  w_46_0 : Fin 8 → mxxMat0
  w_51_0 : Fin 4 → mxxMat0
  w_56_0 : Fin 4 → mxxMat0
  w_59_0 : Fin 2 → mxxMat0
  w_62_0 : Fin 2 → mxxMat0
  w_67_0 : Fin 4 → mxxMat0
  w_72_0 : Fin 4 → mxxMat0

abbrev generatedRoot.constraints_0 (w_0_0 : mxxMat1) (w_1_0 : mxxMat0) (w_2_0 : Fin 1 → mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : Fin 1 → mxxMat0) (w_5_0 : mxxMat0) (w_6_0 : Fin 1 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : Fin 1 → mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 1 → mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : Fin 1 → mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : Fin 1 → mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : Fin 1 → mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : Fin 1 → mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : Fin 1 → mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : Fin 1 → mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : Fin 1 → mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : Fin 1 → mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : Fin 1 → mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Fin 8 → mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : Fin 8 → mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_49_0 : mxxMat0) (w_50_0 : mxxMat0) (w_51_0 : Fin 4 → mxxMat0) (w_52_0 : mxxMat0) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : Fin 4 → mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : Fin 2 → mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : Fin 2 → mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : Fin 4 → mxxMat0) (w_68_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : Fin 4 → mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_76_0 : mxxMat0) (outputs : mxxMat1 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.familyPack (1) [w_1_0] w_2_0 ∧
  MxxRuntime.familyPack (1) [w_3_0] w_4_0 ∧
  MxxRuntime.familyPack (1) [w_5_0] w_6_0 ∧
  MxxRuntime.familyPack (1) [w_7_0] w_8_0 ∧
  MxxRuntime.familyPack (1) [w_9_0] w_10_0 ∧
  MxxRuntime.familyPack (1) [w_11_0] w_12_0 ∧
  MxxRuntime.familyPack (1) [w_13_0] w_14_0 ∧
  MxxRuntime.familyPack (1) [w_15_0] w_16_0 ∧
  MxxRuntime.familyPack (1) [w_17_0] w_18_0 ∧
  MxxRuntime.familyPack (1) [w_19_0] w_20_0 ∧
  MxxRuntime.familyPack (1) [w_21_0] w_22_0 ∧
  MxxRuntime.familyPack (1) [w_23_0] w_24_0 ∧
  MxxRuntime.familyPack (1) [w_25_0] w_26_0 ∧
  MxxRuntime.familyPack (1) [w_27_0] w_28_0 ∧
  MxxRuntime.familyPack (8) [w_29_0, w_30_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0] w_37_0 ∧
  MxxRuntime.familyPack (8) [w_38_0, w_39_0, w_40_0, w_41_0, w_42_0, w_43_0, w_44_0, w_45_0] w_46_0 ∧
  MxxRuntime.familyPack (4) [w_47_0, w_48_0, w_49_0, w_50_0] w_51_0 ∧
  MxxRuntime.familyPack (4) [w_52_0, w_53_0, w_54_0, w_55_0] w_56_0 ∧
  MxxRuntime.familyPack (2) [w_57_0, w_58_0] w_59_0 ∧
  MxxRuntime.familyPack (2) [w_60_0, w_61_0] w_62_0 ∧
  MxxRuntime.familyPack (4) [w_63_0, w_64_0, w_65_0, w_66_0] w_67_0 ∧
  MxxRuntime.familyPack (4) [w_68_0, w_69_0, w_70_0, w_71_0] w_72_0 ∧
  outputs = (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, (w_10_0, (w_12_0, (w_14_0, (w_16_0, (w_18_0, (w_20_0, (w_22_0, (w_24_0, (w_26_0, (w_28_0, (w_37_0, (w_46_0, (w_51_0, (w_56_0, (w_59_0, (w_62_0, (w_67_0, (w_72_0, (w_73_0, (w_74_0, (w_75_0, (w_76_0, ())))))))))))))))))))))))))))

abbrev generatedRoot.body (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Unit) (outputs : mxxMat1 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_4_0 := witness.w_4_0
  let w_5_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat0 := (0 : mxxMat0)
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat0 := (0 : mxxMat0)
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat0 := (0 : mxxMat0)
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := (0 : mxxMat0)
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat0 := (0 : mxxMat0)
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := (0 : mxxMat0)
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat0 := (0 : mxxMat0)
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat0 := (0 : mxxMat0)
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat0 := (0 : mxxMat0)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat0 := (0 : mxxMat0)
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat0 := (0 : mxxMat0)
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat0 := (0 : mxxMat0)
  let w_30_0 : mxxMat0 := (0 : mxxMat0)
  let w_31_0 : mxxMat0 := (0 : mxxMat0)
  let w_32_0 : mxxMat0 := (0 : mxxMat0)
  let w_33_0 : mxxMat0 := (0 : mxxMat0)
  let w_34_0 : mxxMat0 := (0 : mxxMat0)
  let w_35_0 : mxxMat0 := (0 : mxxMat0)
  let w_36_0 : mxxMat0 := (0 : mxxMat0)
  let w_37_0 := witness.w_37_0
  let w_38_0 : mxxMat0 := (0 : mxxMat0)
  let w_39_0 : mxxMat0 := (0 : mxxMat0)
  let w_40_0 : mxxMat0 := (0 : mxxMat0)
  let w_41_0 : mxxMat0 := (0 : mxxMat0)
  let w_42_0 : mxxMat0 := (0 : mxxMat0)
  let w_43_0 : mxxMat0 := (0 : mxxMat0)
  let w_44_0 : mxxMat0 := (0 : mxxMat0)
  let w_45_0 : mxxMat0 := (0 : mxxMat0)
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := (0 : mxxMat0)
  let w_48_0 : mxxMat0 := (0 : mxxMat0)
  let w_49_0 : mxxMat0 := (0 : mxxMat0)
  let w_50_0 : mxxMat0 := (0 : mxxMat0)
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat0 := (0 : mxxMat0)
  let w_53_0 : mxxMat0 := (0 : mxxMat0)
  let w_54_0 : mxxMat0 := (0 : mxxMat0)
  let w_55_0 : mxxMat0 := (0 : mxxMat0)
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := (0 : mxxMat0)
  let w_58_0 : mxxMat0 := (0 : mxxMat0)
  let w_59_0 := witness.w_59_0
  let w_60_0 : mxxMat0 := (0 : mxxMat0)
  let w_61_0 : mxxMat0 := (0 : mxxMat0)
  let w_62_0 := witness.w_62_0
  let w_63_0 : mxxMat0 := (0 : mxxMat0)
  let w_64_0 : mxxMat0 := (0 : mxxMat0)
  let w_65_0 : mxxMat0 := (0 : mxxMat0)
  let w_66_0 : mxxMat0 := (0 : mxxMat0)
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat0 := (0 : mxxMat0)
  let w_69_0 : mxxMat0 := (0 : mxxMat0)
  let w_70_0 : mxxMat0 := (0 : mxxMat0)
  let w_71_0 : mxxMat0 := (0 : mxxMat0)
  let w_72_0 := witness.w_72_0
  let w_73_0 : mxxMat0 := (0 : mxxMat0)
  let w_74_0 : mxxMat0 := (0 : mxxMat0)
  let w_75_0 : mxxMat0 := (0 : mxxMat0)
  let w_76_0 : mxxMat0 := (0 : mxxMat0)
  generatedRoot.constraints_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : mxxMat1 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params () outputs witness


end Stage_zero
