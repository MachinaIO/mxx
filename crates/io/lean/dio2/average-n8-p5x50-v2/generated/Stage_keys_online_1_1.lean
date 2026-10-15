import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_keys_online_1_1

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993
local notation "mxxNat1" => 36550556566445887949135852474979514307313305075533710697673106717124183540354
local notation "mxxNat2" => 84715014674746765542951856319120207364894094043218421459706511318570933778708
local notation "mxxNat3" => 904625697163682872125158399399175198226681342373069820760217620276403972497

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 3 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 50 25
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 3 50
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 8 25 1
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 8 2 1

abbrev sequential_generatedRoot_87.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 4 → mxxMat0) × ByteArray × (Fin 4 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit

abbrev sequential_generatedRoot_87.constraints_4 (w_75_0 : Fin 4 → mxxMat0) (w_79_0 : Fin 4 → mxxMat0) (w_81_0 : Fin 4 → mxxMat0) (w_83_0 : Fin 4 → mxxMat0) (w_41_0 : mxxMat1) (w_45_0 : mxxMat0) (w_58_0 : mxxMat1) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_generatedRoot_87.constraints_3 (w_41_0 : mxxMat1) (w_52_0 : mxxMat1) (w_58_0 : mxxMat1) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 0 1 0 25 w_59_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 25 w_60_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_41_0 1 2 0 25 w_64_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 1 2 0 25 w_67_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev sequential_generatedRoot_87.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat1) (w_41_0 : mxxMat1) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat2) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat2) (w_56_0 : mxxMat3) (w_57_0 : mxxMat3) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_41_0 0 1 0 25 w_42_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_35_0 0 1 0 25 w_43_0 ∧
  MxxRuntime.concatRows w_47_0 w_48_0 w_49_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 1 2 0 25 w_53_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 2 3 0 25 w_54_0 ∧
  MxxRuntime.concatRows w_53_0 w_54_0 w_55_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_55_0 w_56_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_56_0 w_57_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev sequential_generatedRoot_87.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat1) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat4) (w_32_0 : mxxMat2) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat1) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat2) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) : Prop :=
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_26_0 w_27_0 ∧
  MxxRuntime.concatRows w_23_0 w_25_0 w_28_concat_1 ∧
  MxxRuntime.concatRows w_28_concat_1 w_27_0 w_28_0 ∧
  MxxRuntime.concatColumns w_21_0 w_28_0 w_29_0 ∧
  MxxRuntime.concatRows w_30_0 w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_32_0 w_33_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_33_0 w_34_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_35_0 1 2 0 25 w_36_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_35_0 2 3 0 25 w_37_0 ∧
  MxxRuntime.concatRows w_36_0 w_37_0 w_38_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_38_0 w_39_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_39_0 w_40_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0

abbrev sequential_generatedRoot_87.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 4 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 4 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat1) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat1) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0))

def sequential_generatedRoot_87 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_87.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_46_0 : mxxMat0 := inputs.2.1
  let w_63_0 : mxxMat0 := inputs.2.2.1
  let w_66_0 : mxxMat0 := inputs.2.2.2.1
  let w_69_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat1),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat1),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat1),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat1),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat3),
                                                  ∃ (w_34_0 : mxxMat3),
                                                    let w_35_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat3),
                                                            ∃ (w_40_0 : mxxMat3),
                                                              let w_41_0 : mxxMat1 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat3),
                                                                      ∃ (w_51_0 : mxxMat3),
                                                                        let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat3),
                                                                                ∃ (w_57_0 : mxxMat3),
                                                                                  let w_58_0 : mxxMat1 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat0),
                                                                                    ∃ (w_60_0 : mxxMat0),
                                                                                      let w_61_0 : mxxMat0 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat0 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat0),
                                                                                        let w_65_0 : mxxMat0 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat0),
                                                                                          let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat0),
                                                                                            let w_71_0 : mxxMat0 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat0),
                                                                                              let w_74_0 : mxxMat0 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat0),
                                                                                                ∃ (w_80_0 : mxxMat0),
                                                                                                  ∃ (w_82_0 : mxxMat0),
                                                                                                    ∃ (w_84_0 : mxxMat0),
                                                                                                      sequential_generatedRoot_87.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_87.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_87.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_87.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_87.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev parallel_generatedRoot_143.InputType : Type :=
  Int × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat5 × mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit

abbrev parallel_generatedRoot_143.constraints_3 (w_45_0 : mxxMat1) (w_46_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_45_0 2 3 0 25 w_54_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 2 3 0 25 w_55_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_45_0 0 1 0 25 w_56_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 25 w_57_0 ∧
  outputs = (w_46_0, (w_53_0, (w_54_0, (w_55_0, (w_56_0, (w_57_0, ()))))))

abbrev parallel_generatedRoot_143.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_39_0 : ByteArray) (w_41_0 : mxxMat0) (w_48_0 : mxxMat0) (w_24_0 : mxxMat1) (w_30_0 : mxxMat1) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat1) (w_37_concat_1 : mxxMat2) (w_38_0 : mxxMat4) (w_40_0 : mxxMat0) (w_42_0 : mxxMat2) (w_43_0 : mxxMat3) (w_44_0 : mxxMat3) (w_45_0 : mxxMat1) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_49_0 : mxxMat2) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_30_0 2 3 0 25 w_36_0 ∧
  MxxRuntime.concatRows w_34_0 w_35_0 w_37_concat_1 ∧
  MxxRuntime.concatRows w_37_concat_1 w_36_0 w_37_0 ∧
  MxxRuntime.concatColumns w_24_0 w_37_0 w_38_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_39_0) w_40_0 ∧
  MxxRuntime.concatRows w_40_0 w_41_0 w_42_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_42_0 w_43_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_43_0 w_44_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_45_0 1 2 0 25 w_46_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_39_0) w_47_0 ∧
  MxxRuntime.concatRows w_47_0 w_48_0 w_49_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 1 2 0 25 w_53_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0

abbrev parallel_generatedRoot_143.constraints_1 (backend : MxxRuntime.BackendContext) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_16_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat1) (w_24_concat_1 : mxxMat2) (w_27_0 : mxxMat2) (w_28_0 : mxxMat3) (w_29_0 : mxxMat3) (w_30_0 : mxxMat1) (w_31_0 : mxxMat0) (w_35_0 : mxxMat0) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_16_0 1 2 0 25 w_22_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_16_0 2 3 0 25 w_23_0 ∧
  MxxRuntime.concatRows w_21_0 w_22_0 w_24_concat_1 ∧
  MxxRuntime.concatRows w_24_concat_1 w_23_0 w_24_0 ∧
  MxxRuntime.concatRows w_25_0 w_26_0 w_27_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_27_0 w_28_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_28_0 w_29_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_30_0 0 1 0 25 w_31_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_30_0 1 2 0 25 w_35_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0

abbrev parallel_generatedRoot_143.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_6_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat1) (w_5_concat_1 : mxxMat2) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat1) (w_9_concat_1 : mxxMat2) (w_10_0 : mxxMat4) (w_13_0 : mxxMat2) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_16_0 : mxxMat1) (w_17_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_0_0 w_3_0 w_5_concat_1 ∧
  MxxRuntime.concatRows w_5_concat_1 w_4_0 w_5_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_8_0 ∧
  MxxRuntime.concatRows w_6_0 w_7_0 w_9_concat_1 ∧
  MxxRuntime.concatRows w_9_concat_1 w_8_0 w_9_0 ∧
  MxxRuntime.concatColumns w_5_0 w_9_0 w_10_0 ∧
  MxxRuntime.concatRows w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 0 25 w_17_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

def parallel_generatedRoot_143 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_143.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_12_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_18_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_19_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_5_0 : mxxMat1),
        ∃ (w_5_concat_1 : mxxMat2),
          ∃ (w_7_0 : mxxMat0),
            ∃ (w_8_0 : mxxMat0),
              ∃ (w_9_0 : mxxMat1),
                ∃ (w_9_concat_1 : mxxMat2),
                  ∃ (w_10_0 : mxxMat4),
                    ∃ (w_13_0 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_15_0 : mxxMat3),
                          let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_10_0 w_15_0
                          ∃ (w_17_0 : mxxMat0),
                            let w_20_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_18_0 w_19_0
                            let w_21_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_20_0
                            ∃ (w_22_0 : mxxMat0),
                              ∃ (w_23_0 : mxxMat0),
                                ∃ (w_24_0 : mxxMat1),
                                  ∃ (w_24_concat_1 : mxxMat2),
                                    ∃ (w_27_0 : mxxMat2),
                                      ∃ (w_28_0 : mxxMat3),
                                        ∃ (w_29_0 : mxxMat3),
                                          let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_10_0 w_29_0
                                          ∃ (w_31_0 : mxxMat0),
                                            let w_33_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_32_0 w_19_0
                                            let w_34_0 : mxxMat0 := MxxRuntime.matrixAdd w_31_0 w_33_0
                                            ∃ (w_35_0 : mxxMat0),
                                              ∃ (w_36_0 : mxxMat0),
                                                ∃ (w_37_0 : mxxMat1),
                                                  ∃ (w_37_concat_1 : mxxMat2),
                                                    ∃ (w_38_0 : mxxMat4),
                                                      ∃ (w_40_0 : mxxMat0),
                                                        ∃ (w_42_0 : mxxMat2),
                                                          ∃ (w_43_0 : mxxMat3),
                                                            ∃ (w_44_0 : mxxMat3),
                                                              let w_45_0 : mxxMat1 := MxxRuntime.matrixMul w_38_0 w_44_0
                                                              ∃ (w_46_0 : mxxMat0),
                                                                ∃ (w_47_0 : mxxMat0),
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat3),
                                                                      ∃ (w_51_0 : mxxMat3),
                                                                        let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_38_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat0),
                                                                              ∃ (w_56_0 : mxxMat0),
                                                                                ∃ (w_57_0 : mxxMat0),
                                                                                  parallel_generatedRoot_143.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_12_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 ∧
    parallel_generatedRoot_143.constraints_1 backend w_25_0 w_26_0 w_16_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_concat_1 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 ∧
    parallel_generatedRoot_143.constraints_2 backend hashModel i_0 w_39_0 w_41_0 w_48_0 w_24_0 w_30_0 w_34_0 w_35_0 w_36_0 w_37_0 w_37_concat_1 w_38_0 w_40_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 ∧
    parallel_generatedRoot_143.constraints_3 w_45_0 w_46_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 outputs

abbrev parallel_parallel_generatedRoot_144_6.constraints_0 (w_0_0 : Fin 16 → mxxMat0) (w_3_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 16 → mxxMat0) (w_7_0 : Fin 16 → mxxMat0) (w_9_0 : Fin 16 → mxxMat0) (w_11_0 : Fin 16 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_1_0 w_12_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, (w_10_0, (w_12_0, ()))))))

def parallel_parallel_generatedRoot_144_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat0 := inputs.1
  let w_3_0 : Fin 16 → mxxMat0 := inputs.2.1
  let w_5_0 : Fin 16 → mxxMat0 := inputs.2.2.1
  let w_7_0 : Fin 16 → mxxMat0 := inputs.2.2.2.1
  let w_9_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.1
  let w_11_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_10_0 : mxxMat0),
            ∃ (w_12_0 : mxxMat0),
              parallel_parallel_generatedRoot_144_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_144_25.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_144_25.constraints_0 (w_1_0 : Fin 2 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_10_0 : Fin 2 → mxxMat0) (w_14_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_22_0 : Fin 2 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_2_0 w_19_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_2_0 w_23_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def sequential_parallel_generatedRoot_144_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_144_25.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_5_0 : mxxMat0 := inputs.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat0),
    let w_4_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat0),
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat0),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat0),
          let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat0),
            let w_20_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat0),
              let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_parallel_generatedRoot_144_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_144.Witness where
  w_6_0 : Fin 2 → mxxMat0
  w_6_1 : Fin 2 → mxxMat0
  w_6_2 : Fin 2 → mxxMat0
  w_6_3 : Fin 2 → mxxMat0
  w_6_4 : Fin 2 → mxxMat0
  w_6_5 : Fin 2 → mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat0
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_13_0 : mxxMat0
  w_14_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat0
  w_19_0 : mxxMat0
  w_20_0 : mxxMat0
  w_22_0 : mxxMat0
  w_23_0 : mxxMat0
  w_25_0 : mxxMat0
  w_25_1 : mxxMat0
  w_25_2 : mxxMat0
  w_25_3 : mxxMat0
  w_25_4 : mxxMat0
  w_25_5 : mxxMat0

abbrev parallel_generatedRoot_144.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat0) (w_1_0 : Fin 16 → mxxMat0) (w_2_0 : Fin 16 → mxxMat0) (w_3_0 : Fin 16 → mxxMat0) (w_4_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 16 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat0) (w_6_3 : Fin 2 → mxxMat0) (w_6_4 : Fin 2 → mxxMat0) (w_6_5 : Fin 2 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_25_1 : mxxMat0) (w_25_2 : mxxMat0) (w_25_3 : mxxMat0) (w_25_4 : mxxMat0) (w_25_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_144_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
  MxxRuntime.familyGetStatic w_6_0 (0) w_7_0 ∧
  MxxRuntime.familyGetStatic w_6_0 (1) w_8_0 ∧
  MxxRuntime.familyGetStatic w_6_1 (0) w_10_0 ∧
  MxxRuntime.familyGetStatic w_6_1 (1) w_11_0 ∧
  MxxRuntime.familyGetStatic w_6_2 (0) w_13_0 ∧
  MxxRuntime.familyGetStatic w_6_2 (1) w_14_0 ∧
  MxxRuntime.familyGetStatic w_6_3 (0) w_16_0 ∧
  MxxRuntime.familyGetStatic w_6_3 (1) w_17_0 ∧
  MxxRuntime.familyGetStatic w_6_4 (0) w_19_0 ∧
  MxxRuntime.familyGetStatic w_6_4 (1) w_20_0 ∧
  MxxRuntime.familyGetStatic w_6_5 (0) w_22_0 ∧
  MxxRuntime.familyGetStatic w_6_5 (1) w_23_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_144_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_144.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_144.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat0 := inputs.1
  let w_1_0 : Fin 16 → mxxMat0 := inputs.2.1
  let w_2_0 : Fin 16 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Fin 16 → mxxMat0 := inputs.2.2.2.1
  let w_4_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.1
  let w_5_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_6_4 := witness.w_6_4
  let w_6_5 := witness.w_6_5
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat0 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_25_2 := witness.w_25_2
  let w_25_3 := witness.w_25_3
  let w_25_4 := witness.w_25_4
  let w_25_5 := witness.w_25_5
  parallel_generatedRoot_144.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_144 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_144.Witness,
    parallel_generatedRoot_144.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_147.InputType : Type :=
  Int × mxxMat0 × ByteArray × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × mxxMat5 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × ByteArray × mxxMat0 × mxxMat0 × Unit

abbrev parallel_generatedRoot_147.constraints_4 (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 25 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_59_0 0 1 0 25 w_64_0 ∧
  outputs = (w_53_0, (w_60_0, (w_61_0, (w_62_0, (w_63_0, (w_64_0, ()))))))

abbrev parallel_generatedRoot_147.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_46_0 : ByteArray) (w_55_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_56_0 : mxxMat2) (w_57_0 : mxxMat3) (w_58_0 : mxxMat3) (w_59_0 : mxxMat1) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 1 2 0 25 w_53_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_46_0) w_54_0 ∧
  MxxRuntime.concatRows w_54_0 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_56_0 w_57_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_57_0 w_58_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_59_0 1 2 0 25 w_60_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 2 3 0 25 w_61_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_59_0 2 3 0 25 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3

abbrev parallel_generatedRoot_147.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_38_0 : Fin 8 → mxxMat0) (w_46_0 : ByteArray) (w_48_0 : mxxMat0) (w_12_0 : Int) (w_28_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_39_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat1) (w_44_concat_1 : mxxMat2) (w_45_0 : mxxMat4) (w_47_0 : mxxMat0) (w_49_0 : mxxMat2) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_36_0 0 1 0 25 w_37_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_38_0 w_12_0 w_39_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_36_0 1 2 0 25 w_42_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_36_0 2 3 0 25 w_43_0 ∧
  MxxRuntime.concatRows w_41_0 w_42_0 w_44_concat_1 ∧
  MxxRuntime.concatRows w_44_concat_1 w_43_0 w_44_0 ∧
  MxxRuntime.concatColumns w_28_0 w_44_0 w_45_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_46_0) w_47_0 ∧
  MxxRuntime.concatRows w_47_0 w_48_0 w_49_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3

abbrev parallel_generatedRoot_147.constraints_1 (backend : MxxRuntime.BackendContext) (w_21_0 : Fin 8 → mxxMat0) (w_29_0 : Fin 8 → mxxMat0) (w_31_0 : Fin 8 → mxxMat0) (w_12_0 : Int) (w_19_0 : mxxMat1) (w_20_0 : mxxMat0) (w_22_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_28_concat_1 : mxxMat2) (w_30_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat2) (w_34_0 : mxxMat3) (w_35_0 : mxxMat3) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 25 w_20_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_12_0 w_22_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_19_0 1 2 0 25 w_26_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_19_0 2 3 0 25 w_27_0 ∧
  MxxRuntime.concatRows w_25_0 w_26_0 w_28_concat_1 ∧
  MxxRuntime.concatRows w_28_concat_1 w_27_0 w_28_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_29_0 w_12_0 w_30_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_31_0 w_12_0 w_32_0 ∧
  MxxRuntime.concatRows w_30_0 w_32_0 w_33_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_33_0 w_34_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_34_0 w_35_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev parallel_generatedRoot_147.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_6_0 : mxxMat0) (w_11_0 : Fin 8 → mxxMat0) (w_14_0 : Fin 8 → mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat1) (w_5_concat_1 : mxxMat2) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat1) (w_9_concat_1 : mxxMat2) (w_10_0 : mxxMat4) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat2) (w_17_0 : mxxMat3) (w_18_0 : mxxMat3) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_0_0 w_3_0 w_5_concat_1 ∧
  MxxRuntime.concatRows w_5_concat_1 w_4_0 w_5_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_8_0 ∧
  MxxRuntime.concatRows w_6_0 w_7_0 w_9_concat_1 ∧
  MxxRuntime.concatRows w_9_concat_1 w_8_0 w_9_0 ∧
  MxxRuntime.concatColumns w_5_0 w_9_0 w_10_0 ∧
  (2) ≠ 0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_12_0 w_15_0 ∧
  MxxRuntime.concatRows w_13_0 w_15_0 w_16_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_16_0 w_17_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_17_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

def parallel_generatedRoot_147 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_147.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.1
  let w_14_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_21_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_23_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_5_0 : mxxMat1),
        ∃ (w_5_concat_1 : mxxMat2),
          ∃ (w_7_0 : mxxMat0),
            ∃ (w_8_0 : mxxMat0),
              ∃ (w_9_0 : mxxMat1),
                ∃ (w_9_concat_1 : mxxMat2),
                  ∃ (w_10_0 : mxxMat4),
                    let w_12_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (2))
                    ∃ (w_13_0 : mxxMat0),
                      ∃ (w_15_0 : mxxMat0),
                        ∃ (w_16_0 : mxxMat2),
                          ∃ (w_17_0 : mxxMat3),
                            ∃ (w_18_0 : mxxMat3),
                              let w_19_0 : mxxMat1 := MxxRuntime.matrixMul w_10_0 w_18_0
                              ∃ (w_20_0 : mxxMat0),
                                ∃ (w_22_0 : mxxMat0),
                                  let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_22_0 w_23_0
                                  let w_25_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_24_0
                                  ∃ (w_26_0 : mxxMat0),
                                    ∃ (w_27_0 : mxxMat0),
                                      ∃ (w_28_0 : mxxMat1),
                                        ∃ (w_28_concat_1 : mxxMat2),
                                          ∃ (w_30_0 : mxxMat0),
                                            ∃ (w_32_0 : mxxMat0),
                                              ∃ (w_33_0 : mxxMat2),
                                                ∃ (w_34_0 : mxxMat3),
                                                  ∃ (w_35_0 : mxxMat3),
                                                    let w_36_0 : mxxMat1 := MxxRuntime.matrixMul w_10_0 w_35_0
                                                    ∃ (w_37_0 : mxxMat0),
                                                      ∃ (w_39_0 : mxxMat0),
                                                        let w_40_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_39_0 w_23_0
                                                        let w_41_0 : mxxMat0 := MxxRuntime.matrixAdd w_37_0 w_40_0
                                                        ∃ (w_42_0 : mxxMat0),
                                                          ∃ (w_43_0 : mxxMat0),
                                                            ∃ (w_44_0 : mxxMat1),
                                                              ∃ (w_44_concat_1 : mxxMat2),
                                                                ∃ (w_45_0 : mxxMat4),
                                                                  ∃ (w_47_0 : mxxMat0),
                                                                    ∃ (w_49_0 : mxxMat2),
                                                                      ∃ (w_50_0 : mxxMat3),
                                                                        ∃ (w_51_0 : mxxMat3),
                                                                          let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_45_0 w_51_0
                                                                          ∃ (w_53_0 : mxxMat0),
                                                                            ∃ (w_54_0 : mxxMat0),
                                                                              ∃ (w_56_0 : mxxMat2),
                                                                                ∃ (w_57_0 : mxxMat3),
                                                                                  ∃ (w_58_0 : mxxMat3),
                                                                                    let w_59_0 : mxxMat1 := MxxRuntime.matrixMul w_45_0 w_58_0
                                                                                    ∃ (w_60_0 : mxxMat0),
                                                                                      ∃ (w_61_0 : mxxMat0),
                                                                                        ∃ (w_62_0 : mxxMat0),
                                                                                          ∃ (w_63_0 : mxxMat0),
                                                                                            ∃ (w_64_0 : mxxMat0),
                                                                                              parallel_generatedRoot_147.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_14_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_147.constraints_1 backend w_21_0 w_29_0 w_31_0 w_12_0 w_19_0 w_20_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_concat_1 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 ∧
    parallel_generatedRoot_147.constraints_2 backend hashModel i_0 w_38_0 w_46_0 w_48_0 w_12_0 w_28_0 w_36_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_concat_1 w_45_0 w_47_0 w_49_0 w_50_0 w_51_0 ∧
    parallel_generatedRoot_147.constraints_3 backend hashModel i_0 w_46_0 w_55_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 ∧
    parallel_generatedRoot_147.constraints_4 w_52_0 w_53_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_parallel_generatedRoot_148_6.constraints_0 (w_0_0 : Fin 16 → mxxMat0) (w_3_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 16 → mxxMat0) (w_7_0 : Fin 16 → mxxMat0) (w_9_0 : Fin 16 → mxxMat0) (w_11_0 : Fin 16 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_1_0 w_12_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, (w_10_0, (w_12_0, ()))))))

def parallel_parallel_generatedRoot_148_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat0 := inputs.1
  let w_3_0 : Fin 16 → mxxMat0 := inputs.2.1
  let w_5_0 : Fin 16 → mxxMat0 := inputs.2.2.1
  let w_7_0 : Fin 16 → mxxMat0 := inputs.2.2.2.1
  let w_9_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.1
  let w_11_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_10_0 : mxxMat0),
            ∃ (w_12_0 : mxxMat0),
              parallel_parallel_generatedRoot_148_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_148_25.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_148_25.constraints_0 (w_1_0 : Fin 2 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_10_0 : Fin 2 → mxxMat0) (w_14_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_22_0 : Fin 2 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_2_0 w_19_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_2_0 w_23_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def sequential_parallel_generatedRoot_148_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_148_25.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_5_0 : mxxMat0 := inputs.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat0),
    let w_4_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat0),
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat0),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat0),
          let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat0),
            let w_20_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat0),
              let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_parallel_generatedRoot_148_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_148.Witness where
  w_6_0 : Fin 2 → mxxMat0
  w_6_1 : Fin 2 → mxxMat0
  w_6_2 : Fin 2 → mxxMat0
  w_6_3 : Fin 2 → mxxMat0
  w_6_4 : Fin 2 → mxxMat0
  w_6_5 : Fin 2 → mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat0
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_13_0 : mxxMat0
  w_14_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat0
  w_19_0 : mxxMat0
  w_20_0 : mxxMat0
  w_22_0 : mxxMat0
  w_23_0 : mxxMat0
  w_25_0 : mxxMat0
  w_25_1 : mxxMat0
  w_25_2 : mxxMat0
  w_25_3 : mxxMat0
  w_25_4 : mxxMat0
  w_25_5 : mxxMat0

abbrev parallel_generatedRoot_148.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat0) (w_1_0 : Fin 16 → mxxMat0) (w_2_0 : Fin 16 → mxxMat0) (w_3_0 : Fin 16 → mxxMat0) (w_4_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 16 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat0) (w_6_3 : Fin 2 → mxxMat0) (w_6_4 : Fin 2 → mxxMat0) (w_6_5 : Fin 2 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_25_1 : mxxMat0) (w_25_2 : mxxMat0) (w_25_3 : mxxMat0) (w_25_4 : mxxMat0) (w_25_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_148_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
  MxxRuntime.familyGetStatic w_6_0 (0) w_7_0 ∧
  MxxRuntime.familyGetStatic w_6_0 (1) w_8_0 ∧
  MxxRuntime.familyGetStatic w_6_1 (0) w_10_0 ∧
  MxxRuntime.familyGetStatic w_6_1 (1) w_11_0 ∧
  MxxRuntime.familyGetStatic w_6_2 (0) w_13_0 ∧
  MxxRuntime.familyGetStatic w_6_2 (1) w_14_0 ∧
  MxxRuntime.familyGetStatic w_6_3 (0) w_16_0 ∧
  MxxRuntime.familyGetStatic w_6_3 (1) w_17_0 ∧
  MxxRuntime.familyGetStatic w_6_4 (0) w_19_0 ∧
  MxxRuntime.familyGetStatic w_6_4 (1) w_20_0 ∧
  MxxRuntime.familyGetStatic w_6_5 (0) w_22_0 ∧
  MxxRuntime.familyGetStatic w_6_5 (1) w_23_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_148_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_148.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_148.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat0 := inputs.1
  let w_1_0 : Fin 16 → mxxMat0 := inputs.2.1
  let w_2_0 : Fin 16 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Fin 16 → mxxMat0 := inputs.2.2.2.1
  let w_4_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.1
  let w_5_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_6_4 := witness.w_6_4
  let w_6_5 := witness.w_6_5
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat0 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_25_2 := witness.w_25_2
  let w_25_3 := witness.w_25_3
  let w_25_4 := witness.w_25_4
  let w_25_5 := witness.w_25_5
  parallel_generatedRoot_148.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_148 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_148.Witness,
    parallel_generatedRoot_148.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_149_18.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_149_18.constraints_4 (w_75_0 : Fin 8 → mxxMat0) (w_79_0 : Fin 8 → mxxMat0) (w_81_0 : Fin 8 → mxxMat0) (w_83_0 : Fin 8 → mxxMat0) (w_41_0 : mxxMat1) (w_45_0 : mxxMat0) (w_58_0 : mxxMat1) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_parallel_generatedRoot_149_18.constraints_3 (w_41_0 : mxxMat1) (w_52_0 : mxxMat1) (w_58_0 : mxxMat1) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 0 1 0 25 w_59_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 25 w_60_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_41_0 1 2 0 25 w_64_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 1 2 0 25 w_67_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev sequential_parallel_generatedRoot_149_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat1) (w_41_0 : mxxMat1) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat2) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat2) (w_56_0 : mxxMat3) (w_57_0 : mxxMat3) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_41_0 0 1 0 25 w_42_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_35_0 0 1 0 25 w_43_0 ∧
  MxxRuntime.concatRows w_47_0 w_48_0 w_49_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 1 2 0 25 w_53_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 2 3 0 25 w_54_0 ∧
  MxxRuntime.concatRows w_53_0 w_54_0 w_55_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_55_0 w_56_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_56_0 w_57_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev sequential_parallel_generatedRoot_149_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat1) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat4) (w_32_0 : mxxMat2) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat1) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat2) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) : Prop :=
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_26_0 w_27_0 ∧
  MxxRuntime.concatRows w_23_0 w_25_0 w_28_concat_1 ∧
  MxxRuntime.concatRows w_28_concat_1 w_27_0 w_28_0 ∧
  MxxRuntime.concatColumns w_21_0 w_28_0 w_29_0 ∧
  MxxRuntime.concatRows w_30_0 w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_32_0 w_33_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_33_0 w_34_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_35_0 1 2 0 25 w_36_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_35_0 2 3 0 25 w_37_0 ∧
  MxxRuntime.concatRows w_36_0 w_37_0 w_38_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_38_0 w_39_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_39_0 w_40_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0

abbrev sequential_parallel_generatedRoot_149_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat1) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat1) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_1))

def sequential_parallel_generatedRoot_149_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_149_18.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_46_0 : mxxMat0 := inputs.2.1
  let w_63_0 : mxxMat0 := inputs.2.2.1
  let w_66_0 : mxxMat0 := inputs.2.2.2.1
  let w_69_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat1),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat1),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat1),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat1),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat3),
                                                  ∃ (w_34_0 : mxxMat3),
                                                    let w_35_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat3),
                                                            ∃ (w_40_0 : mxxMat3),
                                                              let w_41_0 : mxxMat1 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat3),
                                                                      ∃ (w_51_0 : mxxMat3),
                                                                        let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat3),
                                                                                ∃ (w_57_0 : mxxMat3),
                                                                                  let w_58_0 : mxxMat1 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat0),
                                                                                    ∃ (w_60_0 : mxxMat0),
                                                                                      let w_61_0 : mxxMat0 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat0 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat0),
                                                                                        let w_65_0 : mxxMat0 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat0),
                                                                                          let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat0),
                                                                                            let w_71_0 : mxxMat0 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat0),
                                                                                              let w_74_0 : mxxMat0 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat0),
                                                                                                ∃ (w_80_0 : mxxMat0),
                                                                                                  ∃ (w_82_0 : mxxMat0),
                                                                                                    ∃ (w_84_0 : mxxMat0),
                                                                                                      sequential_parallel_generatedRoot_149_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_149_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_149_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_149_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_149_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev parallel_generatedRoot_149.InputType : Type :=
  (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_149.Witness where
  w_2_0 : mxxMat0
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_8_0 : mxxMat0
  w_18_0 : mxxMat0
  w_18_1 : mxxMat0
  w_18_2 : mxxMat0
  w_18_3 : mxxMat0
  w_18_4 : mxxMat0
  w_18_5 : mxxMat0
  w_18_6 : mxxMat0
  w_18_7 : mxxMat0
  w_18_8 : mxxMat0
  w_18_9 : mxxMat0
  w_19_0 : mxxMat0
  w_21_0 : mxxMat0
  w_22_0 : mxxMat0
  w_23_0 : mxxMat1
  w_23_concat_1 : mxxMat2
  w_24_0 : mxxMat0
  w_25_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat1
  w_27_concat_1 : mxxMat2
  w_28_0 : mxxMat4
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat1
  w_32_concat_1 : mxxMat2
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat1
  w_36_concat_1 : mxxMat2
  w_37_0 : mxxMat4
  w_38_0 : mxxMat2
  w_39_0 : mxxMat3
  w_40_0 : mxxMat3
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat2
  w_45_0 : mxxMat3
  w_46_0 : mxxMat3
  w_48_0 : mxxMat0
  w_50_0 : mxxMat2
  w_51_0 : mxxMat3
  w_52_0 : mxxMat3
  w_54_0 : mxxMat0
  w_55_0 : mxxMat0
  w_56_0 : mxxMat2
  w_57_0 : mxxMat3
  w_58_0 : mxxMat3
  w_60_0 : mxxMat0
  w_62_0 : mxxMat0
  w_64_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0
  w_70_0 : mxxMat0
  w_71_0 : mxxMat0

abbrev parallel_generatedRoot_149.constraints_4 (w_49_0 : mxxMat0) (w_53_0 : mxxMat1) (w_59_0 : mxxMat1) (w_61_0 : mxxMat0) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_59_0 0 1 0 25 w_70_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_53_0 0 1 0 25 w_71_0 ∧
  outputs = (w_49_0, (w_61_0, (w_63_0, (w_65_0, (w_69_0, (w_73_0, ()))))))

abbrev parallel_generatedRoot_149.constraints_3 (w_41_0 : mxxMat1) (w_47_0 : mxxMat1) (w_59_0 : mxxMat1) (w_62_0 : mxxMat0) (w_64_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_47_0 2 3 0 25 w_62_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_59_0 2 3 0 25 w_64_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 25 w_66_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_41_0 0 1 0 25 w_67_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0

abbrev parallel_generatedRoot_149.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat0) (w_18_9 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat2) (w_51_0 : mxxMat3) (w_52_0 : mxxMat3) (w_53_0 : mxxMat1) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat2) (w_57_0 : mxxMat3) (w_58_0 : mxxMat3) (w_59_0 : mxxMat1) (w_60_0 : mxxMat0) : Prop :=
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_47_0 1 2 0 25 w_48_0 ∧
  MxxRuntime.concatRows w_18_7 w_18_9 w_50_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_50_0 w_51_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_51_0 w_52_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_53_0 1 2 0 25 w_54_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_53_0 2 3 0 25 w_55_0 ∧
  MxxRuntime.concatRows w_54_0 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_56_0 w_57_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_57_0 w_58_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_59_0 1 2 0 25 w_60_0

abbrev parallel_generatedRoot_149.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_6 : mxxMat0) (w_18_8 : mxxMat0) (w_32_0 : mxxMat1) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_36_concat_1 : mxxMat2) (w_37_0 : mxxMat4) (w_38_0 : mxxMat2) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) (w_41_0 : mxxMat1) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat2) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_15_0 w_33_0 ∧
  0 ≤ (7) ∧
  (7) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_16_0 w_34_0 ∧
  0 ≤ (7) ∧
  (7) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_17_0 w_35_0 ∧
  MxxRuntime.concatRows w_33_0 w_34_0 w_36_concat_1 ∧
  MxxRuntime.concatRows w_36_concat_1 w_35_0 w_36_0 ∧
  MxxRuntime.concatColumns w_32_0 w_36_0 w_37_0 ∧
  MxxRuntime.concatRows w_18_6 w_18_8 w_38_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_38_0 w_39_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_39_0 w_40_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_41_0 1 2 0 25 w_42_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_43_0 ∧
  MxxRuntime.concatRows w_42_0 w_43_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ 1 ∧
  1 < 2

abbrev parallel_generatedRoot_149.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_3_0 : Fin 8 → mxxMat0) (w_5_0 : Fin 8 → mxxMat0) (w_7_0 : Fin 8 → mxxMat0) (w_9_0 : Fin 8 → mxxMat0) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_0_0 : mxxMat0) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_18_2 : mxxMat0) (w_18_3 : mxxMat0) (w_18_4 : mxxMat0) (w_18_5 : mxxMat0) (w_18_6 : mxxMat0) (w_18_7 : mxxMat0) (w_18_8 : mxxMat0) (w_18_9 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : Int) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_23_concat_1 : mxxMat2) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat1) (w_27_concat_1 : mxxMat2) (w_28_0 : mxxMat4) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat1) (w_32_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_149_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_9_0 (7) w_19_0 ∧
  0 ≤ (7) ∧
  (7) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_20_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_10_0) w_21_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_20_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_10_0) w_22_0 ∧
  MxxRuntime.concatRows w_19_0 w_21_0 w_23_concat_1 ∧
  MxxRuntime.concatRows w_23_concat_1 w_22_0 w_23_0 ∧
  MxxRuntime.familyGetStatic w_11_0 (7) w_24_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_20_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_10_0) w_25_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_20_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_10_0) w_26_0 ∧
  MxxRuntime.concatRows w_24_0 w_25_0 w_27_concat_1 ∧
  MxxRuntime.concatRows w_27_concat_1 w_26_0 w_27_0 ∧
  MxxRuntime.concatColumns w_23_0 w_27_0 w_28_0 ∧
  0 ≤ (7) ∧
  (7) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_12_0 w_29_0 ∧
  0 ≤ (7) ∧
  (7) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_13_0 w_30_0 ∧
  0 ≤ (7) ∧
  (7) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_14_0 w_31_0 ∧
  MxxRuntime.concatRows w_29_0 w_30_0 w_32_concat_1 ∧
  MxxRuntime.concatRows w_32_concat_1 w_31_0 w_32_0 ∧
  0 ≤ (7) ∧
  (7) < 8

abbrev parallel_generatedRoot_149.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_149.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_149.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 8 → mxxMat0 := inputs.1
  let w_3_0 : Fin 8 → mxxMat0 := inputs.2.1
  let w_5_0 : Fin 8 → mxxMat0 := inputs.2.2.1
  let w_7_0 : Fin 8 → mxxMat0 := inputs.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.1
  let w_10_0 : ByteArray := inputs.2.2.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_12_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  let w_2_0 := witness.w_2_0
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_18_0 := witness.w_18_0
  let w_18_1 := witness.w_18_1
  let w_18_2 := witness.w_18_2
  let w_18_3 := witness.w_18_3
  let w_18_4 := witness.w_18_4
  let w_18_5 := witness.w_18_5
  let w_18_6 := witness.w_18_6
  let w_18_7 := witness.w_18_7
  let w_18_8 := witness.w_18_8
  let w_18_9 := witness.w_18_9
  let w_19_0 := witness.w_19_0
  let w_20_0 : Int := (MxxIR.selectInt (7) [(1), (3), (5), (7), (9), (11), (13), (15)])
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_23_concat_1 := witness.w_23_concat_1
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_27_concat_1 := witness.w_27_concat_1
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_32_concat_1 := witness.w_32_concat_1
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_36_concat_1 := witness.w_36_concat_1
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat1 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat1 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat1 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat1 := MxxRuntime.matrixMul w_28_0 w_58_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_3 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_4 w_62_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_5 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 : mxxMat0 := MxxRuntime.matrixAdd w_70_0 w_71_0
  let w_73_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_1 w_72_0
  parallel_generatedRoot_149.constraints_0 backend hashModel tape path params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_149.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_149.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_149.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_149.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_149 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_149.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_149.Witness,
    parallel_generatedRoot_149.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_150.constraints_0 (w_0_0 : Fin 8 → mxxMat0) (w_5_0 : Fin 8 → mxxMat0) (w_9_0 : Fin 8 → mxxMat0) (w_13_0 : Fin 8 → mxxMat0) (w_17_0 : Fin 8 → mxxMat0) (w_21_0 : Fin 8 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_1_0 w_14_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_17_0 w_1_0 w_18_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_1_0 w_22_0 ∧
  (8) ≠ 0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def parallel_generatedRoot_150 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 8 → mxxMat0 := inputs.1
  let w_5_0 : Fin 8 → mxxMat0 := inputs.2.1
  let w_9_0 : Fin 8 → mxxMat0 := inputs.2.2.1
  let w_13_0 : Fin 8 → mxxMat0 := inputs.2.2.2.1
  let w_17_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((Int.fdiv ((Int.ofNat i_0)) (8)) + (Int.fmod ((Int.ofNat i_0)) (8)))
  ∃ (w_2_0 : mxxMat0),
    let w_3_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
    let w_4_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_2_0 w_3_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
      let w_8_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_6_0 w_7_0
      ∃ (w_10_0 : mxxMat0),
        let w_11_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
        let w_12_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_10_0 w_11_0
        ∃ (w_14_0 : mxxMat0),
          let w_15_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
          let w_16_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_14_0 w_15_0
          ∃ (w_18_0 : mxxMat0),
            let w_19_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
            let w_20_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_18_0 w_19_0
            ∃ (w_22_0 : mxxMat0),
              let w_23_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
              let w_24_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_22_0 w_23_0
              parallel_generatedRoot_150.constraints_0 w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 outputs

abbrev sequential_generatedRoot_169.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × Unit

abbrev sequential_generatedRoot_169.constraints_0 (w_1_0 : Fin 8 → mxxMat0) (w_6_0 : Fin 8 → mxxMat0) (w_10_0 : Fin 8 → mxxMat0) (w_14_0 : Fin 8 → mxxMat0) (w_18_0 : Fin 8 → mxxMat0) (w_22_0 : Fin 8 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_2_0 w_19_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_2_0 w_23_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def sequential_generatedRoot_169 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_169.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_5_0 : mxxMat0 := inputs.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : mxxMat0),
    let w_4_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat0),
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat0),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat0),
          let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat0),
            let w_20_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat0),
              let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_generatedRoot_169.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev parallel_generatedRoot_171.constraints_0 (i_0 : Nat) (w_2_0 : mxxMat5) (outputs : mxxMat5) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) ≤ 4294967295 ∧
  outputs = w_2_0

def parallel_generatedRoot_171 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat5) (outputs : mxxMat5) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : mxxMat5 := inputs
  let w_0_0 : mxxMat5 := (MxxRuntime.liftInteger ((8) ^ ((Int.ofNat i_0)).toNat) : mxxMat5)
  let w_2_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_0_0 w_1_0
  parallel_generatedRoot_171.constraints_0 i_0 w_2_0 outputs

abbrev sequential_generatedRoot_172.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_1_0 : mxxMat0) (w_3_0 : Fin 1 → mxxMat5) (w_15_0 : mxxMat0) (w_20_0 : mxxMat0) (w_2_0 : mxxMat0) (w_4_0 : Int) (w_5_0 : mxxMat5) (w_6_0 : mxxMat5) (w_7_0 : mxxMat5) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_13_0 : mxxMat5) (w_16_0 : mxxMat0) (w_18_0 : mxxMat5) (w_21_0 : mxxMat0) (w_23_0 : mxxMat5) (outputs : mxxMat5 × mxxMat5 × mxxMat5 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 * (Int.ofNat i_0)) ∧
  (25 * (Int.ofNat i_0)) < (25 + (25 * (Int.ofNat i_0))) ∧
  (25 + (25 * (Int.ofNat i_0))) ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (25 * (Int.ofNat i_0)) (25 + (25 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 1 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 * (Int.ofNat i_0)) ∧
  (25 * (Int.ofNat i_0)) < (25 + (25 * (Int.ofNat i_0))) ∧
  (25 + (25 * (Int.ofNat i_0))) ≤ 25 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 (25 * (Int.ofNat i_0)) (25 + (25 * (Int.ofNat i_0))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 * (Int.ofNat i_0)) ∧
  (25 * (Int.ofNat i_0)) < (25 + (25 * (Int.ofNat i_0))) ∧
  (25 + (25 * (Int.ofNat i_0))) ≤ 25 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (25 * (Int.ofNat i_0)) (25 + (25 * (Int.ofNat i_0))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_generatedRoot_172 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat5 × mxxMat5 × mxxMat5 × mxxMat0 × (Fin 1 → mxxMat5) × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat5 × mxxMat5 × mxxMat5 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat5 := inputs.1
  let w_14_0 : mxxMat5 := inputs.2.1
  let w_19_0 : mxxMat5 := inputs.2.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat5 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat0),
    let w_4_0 : Int := (Int.ofNat i_0)
    ∃ (w_5_0 : mxxMat5),
      ∃ (w_6_0 : mxxMat5),
        ∃ (w_7_0 : mxxMat5),
          let w_8_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
          let w_9_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat6),
            ∃ (w_11_0 : mxxMat6),
              let w_12_0 : mxxMat5 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat5 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat0),
                let w_17_0 : mxxMat5 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat5 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat0),
                  let w_22_0 : mxxMat5 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat5 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_generatedRoot_172.constraints_0 backend i_0 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev generatedRoot.InputType : Type :=
  (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × ByteArray × (Fin 4 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × mxxMat0 × (Fin 2 → mxxMat5) × mxxMat0 × ByteArray × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 16 → Int) × (Fin 16 → Int) × mxxMat0 × mxxMat0 × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit

abbrev generatedRoot.OutputType : Type :=
  mxxMat7 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : mxxMat0
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_8_0 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0
  w_19_0 : mxxMat0
  w_22_0 : mxxMat1
  w_22_concat_1 : mxxMat2
  w_24_0 : mxxMat0
  w_27_0 : mxxMat0
  w_30_0 : mxxMat0
  w_33_0 : mxxMat1
  w_33_concat_1 : mxxMat2
  w_34_0 : mxxMat4
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_37_0 : mxxMat2
  w_38_0 : mxxMat3
  w_39_0 : mxxMat3
  w_41_0 : mxxMat0
  w_44_0 : mxxMat5
  w_49_0 : mxxMat0
  w_50_0 : mxxMat0
  w_51_0 : mxxMat1
  w_51_concat_1 : mxxMat2
  w_52_0 : mxxMat0
  w_53_0 : mxxMat0
  w_54_0 : mxxMat2
  w_55_0 : mxxMat3
  w_56_0 : mxxMat3
  w_58_0 : mxxMat0
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat1
  w_64_concat_1 : mxxMat2
  w_65_0 : mxxMat4
  w_67_0 : mxxMat0
  w_69_0 : mxxMat0
  w_70_0 : mxxMat2
  w_71_0 : mxxMat3
  w_72_0 : mxxMat3
  w_74_0 : mxxMat0
  w_75_0 : mxxMat0
  w_76_0 : mxxMat0
  w_77_0 : mxxMat0
  w_79_0 : mxxMat0
  w_80_0 : mxxMat2
  w_81_0 : mxxMat3
  w_82_0 : mxxMat3
  w_84_0 : mxxMat0
  w_85_0 : mxxMat0
  w_86_0 : mxxMat0
  w_87_0 : mxxMat0
  w_87_1 : mxxMat0
  w_87_2 : mxxMat0
  w_87_3 : mxxMat0
  w_87_4 : mxxMat0
  w_87_5 : mxxMat0
  w_87_6 : mxxMat0
  w_87_7 : mxxMat0
  w_87_8 : mxxMat0
  w_87_9 : mxxMat0
  w_88_0 : mxxMat0
  w_90_0 : mxxMat0
  w_91_0 : mxxMat0
  w_92_0 : mxxMat1
  w_92_concat_1 : mxxMat2
  w_93_0 : mxxMat0
  w_94_0 : mxxMat0
  w_95_0 : mxxMat0
  w_96_0 : mxxMat1
  w_96_concat_1 : mxxMat2
  w_97_0 : mxxMat4
  w_98_0 : mxxMat0
  w_99_0 : mxxMat0
  w_100_0 : mxxMat0
  w_101_0 : mxxMat1
  w_101_concat_1 : mxxMat2
  w_102_0 : mxxMat0
  w_103_0 : mxxMat0
  w_104_0 : mxxMat0
  w_105_0 : mxxMat1
  w_105_concat_1 : mxxMat2
  w_106_0 : mxxMat4
  w_107_0 : mxxMat2
  w_108_0 : mxxMat3
  w_109_0 : mxxMat3
  w_111_0 : mxxMat0
  w_112_0 : mxxMat0
  w_113_0 : mxxMat2
  w_114_0 : mxxMat3
  w_115_0 : mxxMat3
  w_117_0 : mxxMat0
  w_120_0 : mxxMat5
  w_123_0 : mxxMat6
  w_124_0 : mxxMat6
  w_135_0 : mxxMat0
  w_136_0 : mxxMat0
  w_138_0 : mxxMat0
  w_139_0 : mxxMat0
  w_143_0 : Fin 16 → mxxMat0
  w_143_1 : Fin 16 → mxxMat0
  w_143_2 : Fin 16 → mxxMat0
  w_143_3 : Fin 16 → mxxMat0
  w_143_4 : Fin 16 → mxxMat0
  w_143_5 : Fin 16 → mxxMat0
  w_144_0 : Fin 8 → mxxMat0
  w_144_1 : Fin 8 → mxxMat0
  w_144_2 : Fin 8 → mxxMat0
  w_144_3 : Fin 8 → mxxMat0
  w_144_4 : Fin 8 → mxxMat0
  w_144_5 : Fin 8 → mxxMat0
  w_147_0 : Fin 16 → mxxMat0
  w_147_1 : Fin 16 → mxxMat0
  w_147_2 : Fin 16 → mxxMat0
  w_147_3 : Fin 16 → mxxMat0
  w_147_4 : Fin 16 → mxxMat0
  w_147_5 : Fin 16 → mxxMat0
  w_148_0 : Fin 8 → mxxMat0
  w_148_1 : Fin 8 → mxxMat0
  w_148_2 : Fin 8 → mxxMat0
  w_148_3 : Fin 8 → mxxMat0
  w_148_4 : Fin 8 → mxxMat0
  w_148_5 : Fin 8 → mxxMat0
  w_149_0 : Fin 8 → mxxMat0
  w_149_1 : Fin 8 → mxxMat0
  w_149_2 : Fin 8 → mxxMat0
  w_149_3 : Fin 8 → mxxMat0
  w_149_4 : Fin 8 → mxxMat0
  w_149_5 : Fin 8 → mxxMat0
  w_150_0 : Fin 8 → mxxMat0
  w_150_1 : Fin 8 → mxxMat0
  w_150_2 : Fin 8 → mxxMat0
  w_150_3 : Fin 8 → mxxMat0
  w_150_4 : Fin 8 → mxxMat0
  w_150_5 : Fin 8 → mxxMat0
  w_151_0 : mxxMat0
  w_152_0 : mxxMat0
  w_154_0 : mxxMat0
  w_155_0 : mxxMat0
  w_157_0 : mxxMat0
  w_158_0 : mxxMat0
  w_160_0 : mxxMat0
  w_161_0 : mxxMat0
  w_163_0 : mxxMat0
  w_164_0 : mxxMat0
  w_166_0 : mxxMat0
  w_167_0 : mxxMat0
  w_169_0 : mxxMat0
  w_169_1 : mxxMat0
  w_169_2 : mxxMat0
  w_169_3 : mxxMat0
  w_169_4 : mxxMat0
  w_169_5 : mxxMat0
  w_171_0 : Fin 1 → mxxMat5
  w_172_0 : mxxMat5
  w_172_1 : mxxMat5
  w_172_2 : mxxMat5
  w_174_0 : mxxMat0
  w_178_0 : mxxMat7
  w_179_0 : mxxMat2
  w_180_0 : mxxMat3
  w_181_0 : mxxMat3
  w_183_0 : mxxMat0
  w_184_0 : mxxMat0
  w_185_0 : mxxMat2
  w_186_0 : mxxMat3
  w_187_0 : mxxMat3
  w_189_0 : mxxMat0
  w_191_0 : mxxMat0
  w_193_0 : mxxMat0
  w_194_0 : mxxMat0
  w_197_0 : mxxMat0
  w_198_0 : mxxMat0

abbrev generatedRoot.constraints_9 (w_110_0 : mxxMat1) (w_116_0 : mxxMat1) (w_118_0 : mxxMat0) (w_169_0 : mxxMat0) (w_169_1 : mxxMat0) (w_169_2 : mxxMat0) (w_169_3 : mxxMat0) (w_169_4 : mxxMat0) (w_169_5 : mxxMat0) (w_175_0 : mxxMat0) (w_178_0 : mxxMat7) (w_190_0 : mxxMat0) (w_192_0 : mxxMat0) (w_196_0 : mxxMat0) (w_197_0 : mxxMat0) (w_198_0 : mxxMat0) (w_200_0 : mxxMat0) (outputs : (generatedRoot.OutputType)) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_116_0 0 1 0 25 w_197_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_110_0 0 1 0 25 w_198_0 ∧
  outputs = (w_178_0, (w_190_0, (w_118_0, (w_192_0, (w_175_0, (w_196_0, (w_200_0, (w_169_0, (w_169_1, (w_169_2, (w_169_3, (w_169_4, (w_169_5, ())))))))))))))

abbrev generatedRoot.constraints_8 (w_182_0 : mxxMat1) (w_188_0 : mxxMat1) (w_189_0 : mxxMat0) (w_191_0 : mxxMat0) (w_193_0 : mxxMat0) (w_194_0 : mxxMat0) : Prop :=
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_188_0 1 2 0 25 w_189_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_188_0 2 3 0 25 w_191_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_188_0 0 1 0 25 w_193_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_182_0 0 1 0 25 w_194_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_7 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_87_6 : mxxMat0) (w_87_8 : mxxMat0) (w_116_0 : mxxMat1) (w_126_0 : mxxMat5) (w_169_1 : mxxMat0) (w_169_3 : mxxMat0) (w_169_5 : mxxMat0) (w_170_0 : mxxMat5) (w_171_0 : Fin 1 → mxxMat5) (w_172_0 : mxxMat5) (w_172_1 : mxxMat5) (w_172_2 : mxxMat5) (w_173_0 : mxxMat5) (w_174_0 : mxxMat0) (w_177_0 : mxxMat5) (w_178_0 : mxxMat7) (w_179_0 : mxxMat2) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat1) (w_183_0 : mxxMat0) (w_184_0 : mxxMat0) (w_185_0 : mxxMat2) (w_186_0 : mxxMat3) (w_187_0 : mxxMat3) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_171 backend hashModel tape (path ++ [171, i.val]) params i w_170_0 (w_171_0 i)) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat5 × mxxMat5 × mxxMat5 × Unit) => sequential_generatedRoot_172 backend hashModel tape (path ++ [172, i]) params i (current.1, (current.2.1, (current.2.2.1, (w_169_5, (w_171_0, (w_169_1, (w_169_3, ()))))))) next) (Int.toNat (1)) (w_126_0, (w_126_0, (w_126_0, ()))) (w_172_0, (w_172_1, (w_172_2, ()))) ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_116_0 2 3 0 25 w_174_0 ∧
  MxxRuntime.concatRows w_173_0 w_177_0 w_178_0 ∧
  MxxRuntime.concatRows w_87_6 w_87_8 w_179_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_179_0 w_180_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_180_0 w_181_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_182_0 1 2 0 25 w_183_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_182_0 2 3 0 25 w_184_0 ∧
  MxxRuntime.concatRows w_183_0 w_184_0 w_185_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_185_0 w_186_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_186_0 w_187_0 ∧
  0 ≤ 1

abbrev generatedRoot.constraints_6 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_10_0 : ByteArray) (w_12_0 : Fin 2 → mxxMat0) (w_23_0 : Fin 2 → mxxMat0) (w_43_0 : Fin 2 → mxxMat5) (w_66_0 : ByteArray) (w_127_0 : Fin 8 → mxxMat0) (w_128_0 : Fin 8 → mxxMat0) (w_129_0 : Fin 8 → mxxMat0) (w_130_0 : Fin 8 → mxxMat0) (w_131_0 : Fin 8 → mxxMat0) (w_132_0 : Fin 8 → mxxMat0) (w_133_0 : Fin 16 → Int) (w_134_0 : Fin 16 → Int) (w_137_0 : mxxMat0) (w_140_0 : mxxMat0) (w_141_0 : Fin 16 → mxxMat0) (w_142_0 : Fin 16 → mxxMat0) (w_145_0 : Fin 16 → mxxMat0) (w_146_0 : Fin 16 → mxxMat0) (w_123_0 : mxxMat6) (w_124_0 : mxxMat6) (w_135_0 : mxxMat0) (w_136_0 : mxxMat0) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_143_0 : Fin 16 → mxxMat0) (w_143_1 : Fin 16 → mxxMat0) (w_143_2 : Fin 16 → mxxMat0) (w_143_3 : Fin 16 → mxxMat0) (w_143_4 : Fin 16 → mxxMat0) (w_143_5 : Fin 16 → mxxMat0) (w_144_0 : Fin 8 → mxxMat0) (w_144_1 : Fin 8 → mxxMat0) (w_144_2 : Fin 8 → mxxMat0) (w_144_3 : Fin 8 → mxxMat0) (w_144_4 : Fin 8 → mxxMat0) (w_144_5 : Fin 8 → mxxMat0) (w_147_0 : Fin 16 → mxxMat0) (w_147_1 : Fin 16 → mxxMat0) (w_147_2 : Fin 16 → mxxMat0) (w_147_3 : Fin 16 → mxxMat0) (w_147_4 : Fin 16 → mxxMat0) (w_147_5 : Fin 16 → mxxMat0) (w_148_0 : Fin 8 → mxxMat0) (w_148_1 : Fin 8 → mxxMat0) (w_148_2 : Fin 8 → mxxMat0) (w_148_3 : Fin 8 → mxxMat0) (w_148_4 : Fin 8 → mxxMat0) (w_148_5 : Fin 8 → mxxMat0) (w_149_0 : Fin 8 → mxxMat0) (w_149_1 : Fin 8 → mxxMat0) (w_149_2 : Fin 8 → mxxMat0) (w_149_3 : Fin 8 → mxxMat0) (w_149_4 : Fin 8 → mxxMat0) (w_149_5 : Fin 8 → mxxMat0) (w_150_0 : Fin 8 → mxxMat0) (w_150_1 : Fin 8 → mxxMat0) (w_150_2 : Fin 8 → mxxMat0) (w_150_3 : Fin 8 → mxxMat0) (w_150_4 : Fin 8 → mxxMat0) (w_150_5 : Fin 8 → mxxMat0) (w_151_0 : mxxMat0) (w_152_0 : mxxMat0) (w_153_0 : mxxMat0) (w_154_0 : mxxMat0) (w_155_0 : mxxMat0) (w_156_0 : mxxMat0) (w_157_0 : mxxMat0) (w_158_0 : mxxMat0) (w_159_0 : mxxMat0) (w_160_0 : mxxMat0) (w_161_0 : mxxMat0) (w_162_0 : mxxMat0) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat0) (w_166_0 : mxxMat0) (w_167_0 : mxxMat0) (w_168_0 : mxxMat0) (w_169_0 : mxxMat0) (w_169_1 : mxxMat0) (w_169_2 : mxxMat0) (w_169_3 : mxxMat0) (w_169_4 : mxxMat0) (w_169_5 : mxxMat0) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 2048 w_123_0 w_124_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_10_0) w_135_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_10_0) w_136_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_10_0) w_138_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_10_0) w_139_0 ∧
  (16) = 16 ∧
  (∀ i : Fin 16, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat5, 0 ≤ (w_134_0 i) ∧
  (w_134_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_12_0 (w_134_0 i) w_12_0_gathered ∧
  0 ≤ (w_134_0 i) ∧
  (w_134_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_23_0 (w_134_0 i) w_23_0_gathered ∧
  0 ≤ (w_134_0 i) ∧
  (w_134_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_43_0 (w_134_0 i) w_43_0_gathered ∧
  parallel_generatedRoot_143 backend hashModel tape (path ++ [143, i.val]) params i ((w_134_0 i), (w_12_0_gathered, (w_10_0, (w_23_0_gathered, (w_135_0, (w_136_0, (w_137_0, (w_43_0_gathered, (w_138_0, (w_139_0, (w_140_0, (w_66_0, ((w_141_0 i), ((w_142_0 i), ())))))))))))))) ((w_143_0 i), ((w_143_1 i), ((w_143_2 i), ((w_143_3 i), ((w_143_4 i), ((w_143_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_144 backend hashModel tape (path ++ [144, i.val]) params i (w_143_0, (w_143_1, (w_143_2, (w_143_3, (w_143_4, (w_143_5, ())))))) ((w_144_0 i), ((w_144_1 i), ((w_144_2 i), ((w_144_3 i), ((w_144_4 i), ((w_144_5 i), ()))))))) ∧
  (16) = 16 ∧
  (∀ i : Fin 16, ∃ w_12_0_gathered : mxxMat0, ∃ w_23_0_gathered : mxxMat0, ∃ w_43_0_gathered : mxxMat5, 0 ≤ (w_133_0 i) ∧
  (w_133_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_12_0 (w_133_0 i) w_12_0_gathered ∧
  0 ≤ (w_133_0 i) ∧
  (w_133_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_23_0 (w_133_0 i) w_23_0_gathered ∧
  0 ≤ (w_133_0 i) ∧
  (w_133_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_43_0 (w_133_0 i) w_43_0_gathered ∧
  parallel_generatedRoot_147 backend hashModel tape (path ++ [147, i.val]) params i ((w_133_0 i), (w_12_0_gathered, (w_10_0, (w_23_0_gathered, (w_144_0, (w_144_2, (w_144_4, (w_43_0_gathered, (w_144_1, (w_144_3, (w_144_5, (w_66_0, ((w_145_0 i), ((w_146_0 i), ())))))))))))))) ((w_147_0 i), ((w_147_1 i), ((w_147_2 i), ((w_147_3 i), ((w_147_4 i), ((w_147_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_148 backend hashModel tape (path ++ [148, i.val]) params i (w_147_0, (w_147_1, (w_147_2, (w_147_3, (w_147_4, (w_147_5, ())))))) ((w_148_0 i), ((w_148_1 i), ((w_148_2 i), ((w_148_3 i), ((w_148_4 i), ((w_148_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_149 backend hashModel tape (path ++ [149, i.val]) params i (w_127_0, (w_128_0, (w_129_0, (w_130_0, (w_131_0, (w_10_0, (w_132_0, ((w_148_4 i), ((w_148_0 i), ((w_148_2 i), ((w_148_5 i), ((w_148_1 i), ((w_148_3 i), ()))))))))))))) ((w_149_0 i), ((w_149_1 i), ((w_149_2 i), ((w_149_3 i), ((w_149_4 i), ((w_149_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_150 backend hashModel tape (path ++ [150, i.val]) params i (w_149_0, (w_149_1, (w_149_2, (w_149_3, (w_149_4, (w_149_5, ())))))) ((w_150_0 i), ((w_150_1 i), ((w_150_2 i), ((w_150_3 i), ((w_150_4 i), ((w_150_5 i), ()))))))) ∧
  MxxRuntime.familyGetStatic w_150_0 (0) w_151_0 ∧
  MxxRuntime.familyGetStatic w_150_0 (1) w_152_0 ∧
  MxxRuntime.familyGetStatic w_150_1 (0) w_154_0 ∧
  MxxRuntime.familyGetStatic w_150_1 (1) w_155_0 ∧
  MxxRuntime.familyGetStatic w_150_2 (0) w_157_0 ∧
  MxxRuntime.familyGetStatic w_150_2 (1) w_158_0 ∧
  MxxRuntime.familyGetStatic w_150_3 (0) w_160_0 ∧
  MxxRuntime.familyGetStatic w_150_3 (1) w_161_0 ∧
  MxxRuntime.familyGetStatic w_150_4 (0) w_163_0 ∧
  MxxRuntime.familyGetStatic w_150_4 (1) w_164_0 ∧
  MxxRuntime.familyGetStatic w_150_5 (0) w_166_0 ∧
  MxxRuntime.familyGetStatic w_150_5 (1) w_167_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_169 backend hashModel tape (path ++ [169, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_150_0, (w_150_1, (w_150_2, (w_150_3, (w_150_4, (w_150_5, ())))))))))))) next) (Int.toNat (6)) (w_153_0, (w_156_0, (w_159_0, (w_162_0, (w_165_0, (w_168_0, ())))))) (w_169_0, (w_169_1, (w_169_2, (w_169_3, (w_169_4, (w_169_5, ())))))) ∧
  (1) = 1

abbrev generatedRoot.constraints_5 (backend : MxxRuntime.BackendContext) (w_87_7 : mxxMat0) (w_87_9 : mxxMat0) (w_101_0 : mxxMat1) (w_102_0 : mxxMat0) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat1) (w_105_concat_1 : mxxMat2) (w_106_0 : mxxMat4) (w_107_0 : mxxMat2) (w_108_0 : mxxMat3) (w_109_0 : mxxMat3) (w_110_0 : mxxMat1) (w_111_0 : mxxMat0) (w_112_0 : mxxMat0) (w_113_0 : mxxMat2) (w_114_0 : mxxMat3) (w_115_0 : mxxMat3) (w_116_0 : mxxMat1) (w_117_0 : mxxMat0) (w_119_0 : mxxMat5) (w_120_0 : mxxMat5) (w_122_0 : mxxMat5) (w_123_0 : mxxMat6) : Prop :=
  MxxRuntime.concatRows w_102_0 w_103_0 w_105_concat_1 ∧
  MxxRuntime.concatRows w_105_concat_1 w_104_0 w_105_0 ∧
  MxxRuntime.concatColumns w_101_0 w_105_0 w_106_0 ∧
  MxxRuntime.concatRows w_87_7 w_87_9 w_107_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_107_0 w_108_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_108_0 w_109_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_110_0 1 2 0 25 w_111_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_110_0 2 3 0 25 w_112_0 ∧
  MxxRuntime.concatRows w_111_0 w_112_0 w_113_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_113_0 w_114_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_114_0 w_115_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_116_0 1 2 0 25 w_117_0 ∧
  MxxRuntime.centeredRebaseRuns w_119_0 w_120_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_122_0 w_123_0

abbrev generatedRoot.constraints_4 (hashModel : MxxRuntime.HashModel) (w_10_0 : ByteArray) (w_11_0 : Fin 4 → mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_76_0 : mxxMat0) (w_84_0 : mxxMat0) (w_85_0 : mxxMat0) (w_86_0 : mxxMat0) (w_88_0 : mxxMat0) (w_89_0 : Int) (w_90_0 : mxxMat0) (w_91_0 : mxxMat0) (w_92_0 : mxxMat1) (w_92_concat_1 : mxxMat2) (w_93_0 : mxxMat0) (w_94_0 : mxxMat0) (w_95_0 : mxxMat0) (w_96_0 : mxxMat1) (w_96_concat_1 : mxxMat2) (w_97_0 : mxxMat4) (w_98_0 : mxxMat0) (w_99_0 : mxxMat0) (w_100_0 : mxxMat0) (w_101_0 : mxxMat1) (w_101_concat_1 : mxxMat2) (w_102_0 : mxxMat0) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) : Prop :=
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_89_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_10_0) w_90_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_89_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_10_0) w_91_0 ∧
  MxxRuntime.concatRows w_88_0 w_90_0 w_92_concat_1 ∧
  MxxRuntime.concatRows w_92_concat_1 w_91_0 w_92_0 ∧
  MxxRuntime.familyGetStatic w_11_0 (3) w_93_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_89_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_10_0) w_94_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_89_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_10_0) w_95_0 ∧
  MxxRuntime.concatRows w_93_0 w_94_0 w_96_concat_1 ∧
  MxxRuntime.concatRows w_96_concat_1 w_95_0 w_96_0 ∧
  MxxRuntime.concatColumns w_92_0 w_96_0 w_97_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_74_0 w_98_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_75_0 w_99_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_76_0 w_100_0 ∧
  MxxRuntime.concatRows w_98_0 w_99_0 w_101_concat_1 ∧
  MxxRuntime.concatRows w_101_concat_1 w_100_0 w_101_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_84_0 w_102_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_85_0 w_103_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_86_0 w_104_0

abbrev generatedRoot.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_1_0 : Fin 4 → mxxMat0) (w_3_0 : Fin 4 → mxxMat0) (w_5_0 : Fin 4 → mxxMat0) (w_7_0 : Fin 4 → mxxMat0) (w_9_0 : Fin 4 → mxxMat0) (w_10_0 : ByteArray) (w_11_0 : Fin 4 → mxxMat0) (w_66_0 : ByteArray) (w_78_0 : Fin 1 → mxxMat0) (w_0_0 : mxxMat0) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_73_0 : mxxMat1) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_76_0 : mxxMat0) (w_77_0 : mxxMat0) (w_79_0 : mxxMat0) (w_80_0 : mxxMat2) (w_81_0 : mxxMat3) (w_82_0 : mxxMat3) (w_83_0 : mxxMat1) (w_84_0 : mxxMat0) (w_85_0 : mxxMat0) (w_86_0 : mxxMat0) (w_87_0 : mxxMat0) (w_87_1 : mxxMat0) (w_87_2 : mxxMat0) (w_87_3 : mxxMat0) (w_87_4 : mxxMat0) (w_87_5 : mxxMat0) (w_87_6 : mxxMat0) (w_87_7 : mxxMat0) (w_87_8 : mxxMat0) (w_87_9 : mxxMat0) (w_88_0 : mxxMat0) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_73_0 2 3 0 25 w_76_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_66_0) w_77_0 ∧
  MxxRuntime.familyGetStatic w_78_0 (0) w_79_0 ∧
  MxxRuntime.concatRows w_77_0 w_79_0 w_80_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_80_0 w_81_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_81_0 w_82_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_83_0 0 1 0 25 w_84_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_83_0 1 2 0 25 w_85_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_83_0 2 3 0 25 w_86_0 ∧
  0 ≤ (3) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_87 backend hashModel tape (path ++ [87, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_74_0, (w_75_0, (w_76_0, (w_84_0, (w_85_0, (w_86_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_87_0, (w_87_1, (w_87_2, (w_87_3, (w_87_4, (w_87_5, (w_87_6, (w_87_7, (w_87_8, (w_87_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_9_0 (3) w_88_0

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_66_0 : ByteArray) (w_68_0 : Fin 1 → mxxMat0) (w_51_0 : mxxMat1) (w_57_0 : mxxMat1) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat1) (w_64_concat_1 : mxxMat2) (w_65_0 : mxxMat4) (w_67_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat2) (w_71_0 : mxxMat3) (w_72_0 : mxxMat3) (w_73_0 : mxxMat1) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) : Prop :=
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_57_0 2 3 0 25 w_63_0 ∧
  MxxRuntime.concatRows w_61_0 w_62_0 w_64_concat_1 ∧
  MxxRuntime.concatRows w_64_concat_1 w_63_0 w_64_0 ∧
  MxxRuntime.concatColumns w_51_0 w_64_0 w_65_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_66_0) w_67_0 ∧
  MxxRuntime.familyGetStatic w_68_0 (0) w_69_0 ∧
  MxxRuntime.concatRows w_67_0 w_69_0 w_70_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_70_0 w_71_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_71_0 w_72_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_73_0 0 1 0 25 w_74_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_73_0 1 2 0 25 w_75_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_10_0 : ByteArray) (w_40_0 : mxxMat1) (w_48_0 : mxxMat0) (w_49_0 : mxxMat0) (w_50_0 : mxxMat0) (w_51_0 : mxxMat1) (w_51_concat_1 : mxxMat2) (w_52_0 : mxxMat0) (w_53_0 : mxxMat0) (w_54_0 : mxxMat2) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_57_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat0) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_40_0 1 2 0 25 w_49_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_40_0 2 3 0 25 w_50_0 ∧
  MxxRuntime.concatRows w_48_0 w_49_0 w_51_concat_1 ∧
  MxxRuntime.concatRows w_51_concat_1 w_50_0 w_51_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_10_0) w_52_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_10_0) w_53_0 ∧
  MxxRuntime.concatRows w_52_0 w_53_0 w_54_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_55_0 w_56_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_57_0 0 1 0 25 w_58_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_57_0 1 2 0 25 w_62_0 ∧
  0 ≤ 2

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : Fin 4 → mxxMat0) (w_3_0 : Fin 4 → mxxMat0) (w_5_0 : Fin 4 → mxxMat0) (w_7_0 : Fin 4 → mxxMat0) (w_10_0 : ByteArray) (w_12_0 : Fin 2 → mxxMat0) (w_23_0 : Fin 2 → mxxMat0) (w_43_0 : Fin 2 → mxxMat5) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_22_concat_1 : mxxMat2) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat1) (w_33_concat_1 : mxxMat2) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat2) (w_38_0 : mxxMat3) (w_39_0 : mxxMat3) (w_40_0 : mxxMat1) (w_41_0 : mxxMat0) (w_44_0 : mxxMat5) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_12_0 (1) w_13_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_10_0) w_16_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_10_0) w_19_0 ∧
  MxxRuntime.concatRows w_15_0 w_18_0 w_22_concat_1 ∧
  MxxRuntime.concatRows w_22_concat_1 w_21_0 w_22_0 ∧
  MxxRuntime.familyGetStatic w_23_0 (1) w_24_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_10_0) w_27_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_10_0) w_30_0 ∧
  MxxRuntime.concatRows w_26_0 w_29_0 w_33_concat_1 ∧
  MxxRuntime.concatRows w_33_concat_1 w_32_0 w_33_0 ∧
  MxxRuntime.concatColumns w_22_0 w_33_0 w_34_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_10_0) w_35_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_10_0) w_36_0 ∧
  MxxRuntime.concatRows w_35_0 w_36_0 w_37_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_37_0 w_38_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_38_0 w_39_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_40_0 0 1 0 25 w_41_0 ∧
  MxxRuntime.familyGetStatic w_43_0 (1) w_44_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (generatedRoot.InputType)) (outputs : (generatedRoot.OutputType)) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 4 → mxxMat0 := inputs.1
  let w_3_0 : Fin 4 → mxxMat0 := inputs.2.1
  let w_5_0 : Fin 4 → mxxMat0 := inputs.2.2.1
  let w_7_0 : Fin 4 → mxxMat0 := inputs.2.2.2.1
  let w_9_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.1
  let w_10_0 : ByteArray := inputs.2.2.2.2.2.1
  let w_11_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_12_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_23_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_42_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 2 → mxxMat5 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_59_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_66_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_68_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_78_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_127_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_128_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_129_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_130_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_131_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_132_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_133_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_134_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_137_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_140_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_141_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_142_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_145_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_146_0 : Fin 16 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  let w_2_0 := witness.w_2_0
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : Int := 2
  let w_15_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : Int := 2
  let w_18_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : Int := 2
  let w_21_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_22_concat_1 := witness.w_22_concat_1
  let w_24_0 := witness.w_24_0
  let w_25_0 : Int := 2
  let w_26_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_24_0 w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : Int := 2
  let w_29_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_27_0 w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 : Int := 2
  let w_32_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_30_0 w_31_0
  let w_33_0 := witness.w_33_0
  let w_33_concat_1 := witness.w_33_concat_1
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat1 := MxxRuntime.matrixMul w_34_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : Int := 2
  let w_46_0 : mxxMat5 := MxxRuntime.multiplyMonomial w_44_0 w_45_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_42_0 w_46_0
  let w_48_0 : mxxMat0 := MxxRuntime.matrixAdd w_41_0 w_47_0
  let w_49_0 := witness.w_49_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_51_concat_1 := witness.w_51_concat_1
  let w_52_0 := witness.w_52_0
  let w_53_0 := witness.w_53_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat1 := MxxRuntime.matrixMul w_34_0 w_56_0
  let w_58_0 := witness.w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_59_0 w_46_0
  let w_61_0 : mxxMat0 := MxxRuntime.matrixAdd w_58_0 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_64_concat_1 := witness.w_64_concat_1
  let w_65_0 := witness.w_65_0
  let w_67_0 := witness.w_67_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 := witness.w_72_0
  let w_73_0 : mxxMat1 := MxxRuntime.matrixMul w_65_0 w_72_0
  let w_74_0 := witness.w_74_0
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 := witness.w_82_0
  let w_83_0 : mxxMat1 := MxxRuntime.matrixMul w_65_0 w_82_0
  let w_84_0 := witness.w_84_0
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 := witness.w_87_0
  let w_87_1 := witness.w_87_1
  let w_87_2 := witness.w_87_2
  let w_87_3 := witness.w_87_3
  let w_87_4 := witness.w_87_4
  let w_87_5 := witness.w_87_5
  let w_87_6 := witness.w_87_6
  let w_87_7 := witness.w_87_7
  let w_87_8 := witness.w_87_8
  let w_87_9 := witness.w_87_9
  let w_88_0 := witness.w_88_0
  let w_89_0 : Int := (MxxIR.selectInt (3) [(1), (3), (5), (15)])
  let w_90_0 := witness.w_90_0
  let w_91_0 := witness.w_91_0
  let w_92_0 := witness.w_92_0
  let w_92_concat_1 := witness.w_92_concat_1
  let w_93_0 := witness.w_93_0
  let w_94_0 := witness.w_94_0
  let w_95_0 := witness.w_95_0
  let w_96_0 := witness.w_96_0
  let w_96_concat_1 := witness.w_96_concat_1
  let w_97_0 := witness.w_97_0
  let w_98_0 := witness.w_98_0
  let w_99_0 := witness.w_99_0
  let w_100_0 := witness.w_100_0
  let w_101_0 := witness.w_101_0
  let w_101_concat_1 := witness.w_101_concat_1
  let w_102_0 := witness.w_102_0
  let w_103_0 := witness.w_103_0
  let w_104_0 := witness.w_104_0
  let w_105_0 := witness.w_105_0
  let w_105_concat_1 := witness.w_105_concat_1
  let w_106_0 := witness.w_106_0
  let w_107_0 := witness.w_107_0
  let w_108_0 := witness.w_108_0
  let w_109_0 := witness.w_109_0
  let w_110_0 : mxxMat1 := MxxRuntime.matrixMul w_106_0 w_109_0
  let w_111_0 := witness.w_111_0
  let w_112_0 := witness.w_112_0
  let w_113_0 := witness.w_113_0
  let w_114_0 := witness.w_114_0
  let w_115_0 := witness.w_115_0
  let w_116_0 : mxxMat1 := MxxRuntime.matrixMul w_97_0 w_115_0
  let w_117_0 := witness.w_117_0
  let w_118_0 : mxxMat0 := MxxRuntime.matrixAdd w_87_3 w_117_0
  let w_119_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)
  let w_120_0 := witness.w_120_0
  let w_121_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_122_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_120_0 w_121_0
  let w_123_0 := witness.w_123_0
  let w_124_0 := witness.w_124_0
  let w_125_0 : mxxMat5 := MxxRuntime.matrixMul w_118_0 w_124_0
  let w_126_0 : mxxMat5 := (0 : mxxMat5)
  let w_135_0 := witness.w_135_0
  let w_136_0 := witness.w_136_0
  let w_138_0 := witness.w_138_0
  let w_139_0 := witness.w_139_0
  let w_143_0 := witness.w_143_0
  let w_143_1 := witness.w_143_1
  let w_143_2 := witness.w_143_2
  let w_143_3 := witness.w_143_3
  let w_143_4 := witness.w_143_4
  let w_143_5 := witness.w_143_5
  let w_144_0 := witness.w_144_0
  let w_144_1 := witness.w_144_1
  let w_144_2 := witness.w_144_2
  let w_144_3 := witness.w_144_3
  let w_144_4 := witness.w_144_4
  let w_144_5 := witness.w_144_5
  let w_147_0 := witness.w_147_0
  let w_147_1 := witness.w_147_1
  let w_147_2 := witness.w_147_2
  let w_147_3 := witness.w_147_3
  let w_147_4 := witness.w_147_4
  let w_147_5 := witness.w_147_5
  let w_148_0 := witness.w_148_0
  let w_148_1 := witness.w_148_1
  let w_148_2 := witness.w_148_2
  let w_148_3 := witness.w_148_3
  let w_148_4 := witness.w_148_4
  let w_148_5 := witness.w_148_5
  let w_149_0 := witness.w_149_0
  let w_149_1 := witness.w_149_1
  let w_149_2 := witness.w_149_2
  let w_149_3 := witness.w_149_3
  let w_149_4 := witness.w_149_4
  let w_149_5 := witness.w_149_5
  let w_150_0 := witness.w_150_0
  let w_150_1 := witness.w_150_1
  let w_150_2 := witness.w_150_2
  let w_150_3 := witness.w_150_3
  let w_150_4 := witness.w_150_4
  let w_150_5 := witness.w_150_5
  let w_151_0 := witness.w_151_0
  let w_152_0 := witness.w_152_0
  let w_153_0 : mxxMat0 := MxxRuntime.matrixAdd w_151_0 w_152_0
  let w_154_0 := witness.w_154_0
  let w_155_0 := witness.w_155_0
  let w_156_0 : mxxMat0 := MxxRuntime.matrixAdd w_154_0 w_155_0
  let w_157_0 := witness.w_157_0
  let w_158_0 := witness.w_158_0
  let w_159_0 : mxxMat0 := MxxRuntime.matrixAdd w_157_0 w_158_0
  let w_160_0 := witness.w_160_0
  let w_161_0 := witness.w_161_0
  let w_162_0 : mxxMat0 := MxxRuntime.matrixAdd w_160_0 w_161_0
  let w_163_0 := witness.w_163_0
  let w_164_0 := witness.w_164_0
  let w_165_0 : mxxMat0 := MxxRuntime.matrixAdd w_163_0 w_164_0
  let w_166_0 := witness.w_166_0
  let w_167_0 := witness.w_167_0
  let w_168_0 : mxxMat0 := MxxRuntime.matrixAdd w_166_0 w_167_0
  let w_169_0 := witness.w_169_0
  let w_169_1 := witness.w_169_1
  let w_169_2 := witness.w_169_2
  let w_169_3 := witness.w_169_3
  let w_169_4 := witness.w_169_4
  let w_169_5 := witness.w_169_5
  let w_170_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat5)
  let w_171_0 := witness.w_171_0
  let w_172_0 := witness.w_172_0
  let w_172_1 := witness.w_172_1
  let w_172_2 := witness.w_172_2
  let w_173_0 : mxxMat5 := MxxRuntime.matrixAdd w_125_0 w_172_1
  let w_174_0 := witness.w_174_0
  let w_175_0 : mxxMat0 := MxxRuntime.matrixAdd w_87_5 w_174_0
  let w_176_0 : mxxMat5 := MxxRuntime.matrixMul w_175_0 w_124_0
  let w_177_0 : mxxMat5 := MxxRuntime.matrixAdd w_176_0 w_172_2
  let w_178_0 := witness.w_178_0
  let w_179_0 := witness.w_179_0
  let w_180_0 := witness.w_180_0
  let w_181_0 := witness.w_181_0
  let w_182_0 : mxxMat1 := MxxRuntime.matrixMul w_106_0 w_181_0
  let w_183_0 := witness.w_183_0
  let w_184_0 := witness.w_184_0
  let w_185_0 := witness.w_185_0
  let w_186_0 := witness.w_186_0
  let w_187_0 := witness.w_187_0
  let w_188_0 : mxxMat1 := MxxRuntime.matrixMul w_97_0 w_187_0
  let w_189_0 := witness.w_189_0
  let w_190_0 : mxxMat0 := MxxRuntime.matrixAdd w_87_2 w_189_0
  let w_191_0 := witness.w_191_0
  let w_192_0 : mxxMat0 := MxxRuntime.matrixAdd w_87_4 w_191_0
  let w_193_0 := witness.w_193_0
  let w_194_0 := witness.w_194_0
  let w_195_0 : mxxMat0 := MxxRuntime.matrixAdd w_193_0 w_194_0
  let w_196_0 : mxxMat0 := MxxRuntime.matrixAdd w_87_0 w_195_0
  let w_197_0 := witness.w_197_0
  let w_198_0 := witness.w_198_0
  let w_199_0 : mxxMat0 := MxxRuntime.matrixAdd w_197_0 w_198_0
  let w_200_0 : mxxMat0 := MxxRuntime.matrixAdd w_87_1 w_199_0
  generatedRoot.constraints_0 backend hashModel w_1_0 w_3_0 w_5_0 w_7_0 w_10_0 w_12_0 w_23_0 w_43_0 w_2_0 w_4_0 w_6_0 w_8_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_21_0 w_22_0 w_22_concat_1 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_33_concat_1 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_44_0 ∧
    generatedRoot.constraints_1 backend hashModel w_10_0 w_40_0 w_48_0 w_49_0 w_50_0 w_51_0 w_51_concat_1 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_62_0 ∧
    generatedRoot.constraints_2 backend hashModel w_66_0 w_68_0 w_51_0 w_57_0 w_61_0 w_62_0 w_63_0 w_64_0 w_64_concat_1 w_65_0 w_67_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 ∧
    generatedRoot.constraints_3 backend hashModel tape path params w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_66_0 w_78_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_87_1 w_87_2 w_87_3 w_87_4 w_87_5 w_87_6 w_87_7 w_87_8 w_87_9 w_88_0 ∧
    generatedRoot.constraints_4 hashModel w_10_0 w_11_0 w_74_0 w_75_0 w_76_0 w_84_0 w_85_0 w_86_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_92_concat_1 w_93_0 w_94_0 w_95_0 w_96_0 w_96_concat_1 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_101_concat_1 w_102_0 w_103_0 w_104_0 ∧
    generatedRoot.constraints_5 backend w_87_7 w_87_9 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_concat_1 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_119_0 w_120_0 w_122_0 w_123_0 ∧
    generatedRoot.constraints_6 backend hashModel tape path params w_10_0 w_12_0 w_23_0 w_43_0 w_66_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_137_0 w_140_0 w_141_0 w_142_0 w_145_0 w_146_0 w_123_0 w_124_0 w_135_0 w_136_0 w_138_0 w_139_0 w_143_0 w_143_1 w_143_2 w_143_3 w_143_4 w_143_5 w_144_0 w_144_1 w_144_2 w_144_3 w_144_4 w_144_5 w_147_0 w_147_1 w_147_2 w_147_3 w_147_4 w_147_5 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_149_0 w_149_1 w_149_2 w_149_3 w_149_4 w_149_5 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_169_1 w_169_2 w_169_3 w_169_4 w_169_5 ∧
    generatedRoot.constraints_7 backend hashModel tape path params w_87_6 w_87_8 w_116_0 w_126_0 w_169_1 w_169_3 w_169_5 w_170_0 w_171_0 w_172_0 w_172_1 w_172_2 w_173_0 w_174_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 ∧
    generatedRoot.constraints_8 w_182_0 w_188_0 w_189_0 w_191_0 w_193_0 w_194_0 ∧
    generatedRoot.constraints_9 w_110_0 w_116_0 w_118_0 w_169_0 w_169_1 w_169_2 w_169_3 w_169_4 w_169_5 w_175_0 w_178_0 w_190_0 w_192_0 w_196_0 w_197_0 w_198_0 w_200_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (generatedRoot.InputType)) (outputs : (generatedRoot.OutputType)) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_keys_online_1_1
