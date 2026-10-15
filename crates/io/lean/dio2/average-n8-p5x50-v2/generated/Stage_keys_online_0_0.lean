import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_keys_online_0_0

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993
local notation "mxxNat1" => 29270270146341990167195247313302370525046749248022274291861926128744555509284
local notation "mxxNat2" => 102443552169918673375612164323677696578152944978499947864697498749418303762829
local notation "mxxNat3" => 1224430409422355437355602951403039070751149016291967033667224370016023747552
local notation "mxxNat4" => 1577331461107503857687361648278377550964861539914994428956938744873164216254
local notation "mxxNat5" => 231620637999516441280060882203998849893024391639068770271694775176939662912
local notation "mxxNat6" => 286408645742909803465705372805876867840931182364166978575391001671831167302
local notation "mxxNat7" => 298711634382445948711902742905408453456759239282082071569621589367657095967
local notation "mxxNat8" => 904625697163682872125158399399175198226681342373069820760217620276403972497

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

/-- The modulus of a matrix type. -/
abbrev modulus_1 : Nat := 1125899906842273

/-- The modulus of a matrix type. -/
abbrev modulus_2 : Nat := 1125899906842177

/-- The modulus of a matrix type. -/
abbrev modulus_3 : Nat := 1125899906842033

/-- The modulus of a matrix type. -/
abbrev modulus_4 : Nat := 1125899906841713

/-- The modulus of a matrix type. -/
abbrev modulus_5 : Nat := 1125899906841377

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 3 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 3 50
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 8 50 25
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 8 25 25
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 8 1 22500
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix modulus_0 8 1 900
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix modulus_0 8 1 112500
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix modulus_0 8 1 4500
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix modulus_0 8 1 625
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix modulus_0 8 25 1
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix modulus_0 8 625 25
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix modulus_0 8 1 50
local notation "mxxMat15" => Mxx.Primitives.ExactMatrix modulus_1 8 1 25
local notation "mxxMat16" => Mxx.Primitives.ExactMatrix modulus_0 8 25 625
local notation "mxxMat17" => Mxx.Primitives.ExactMatrix modulus_0 8 625 625
local notation "mxxMat18" => Mxx.Primitives.ExactMatrix modulus_1 8 1 1
local notation "mxxMat19" => Mxx.Primitives.ExactMatrix modulus_2 8 1 25
local notation "mxxMat20" => Mxx.Primitives.ExactMatrix modulus_2 8 1 1
local notation "mxxMat21" => Mxx.Primitives.ExactMatrix modulus_3 8 1 25
local notation "mxxMat22" => Mxx.Primitives.ExactMatrix modulus_3 8 1 1
local notation "mxxMat23" => Mxx.Primitives.ExactMatrix modulus_4 8 1 25
local notation "mxxMat24" => Mxx.Primitives.ExactMatrix modulus_4 8 1 1
local notation "mxxMat25" => Mxx.Primitives.ExactMatrix modulus_5 8 1 25
local notation "mxxMat26" => Mxx.Primitives.ExactMatrix modulus_5 8 1 1
local notation "mxxMat27" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54
local notation "mxxMat28" => Mxx.Primitives.ExactMatrix modulus_0 8 2 46
local notation "mxxMat29" => Mxx.Primitives.ExactMatrix modulus_0 8 2 100

abbrev parallel_generatedRoot_27.InputType : Type :=
  Int × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit

abbrev parallel_generatedRoot_27.constraints_3 (w_45_0 : mxxMat2) (w_46_0 : mxxMat0) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_27.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_39_0 : ByteArray) (w_41_0 : mxxMat0) (w_48_0 : mxxMat0) (w_24_0 : mxxMat2) (w_30_0 : mxxMat2) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat2) (w_37_concat_1 : mxxMat3) (w_38_0 : mxxMat4) (w_40_0 : mxxMat0) (w_42_0 : mxxMat3) (w_43_0 : mxxMat5) (w_44_0 : mxxMat5) (w_45_0 : mxxMat2) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_30_0 2 3 0 25 w_36_0 ∧
  MxxRuntime.concatRows w_34_0 w_35_0 w_37_concat_1 ∧
  MxxRuntime.concatRows w_37_concat_1 w_36_0 w_37_0 ∧
  MxxRuntime.concatColumns w_24_0 w_37_0 w_38_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_39_0) w_40_0 ∧
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_39_0) w_47_0 ∧
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

abbrev parallel_generatedRoot_27.constraints_1 (backend : MxxRuntime.BackendContext) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_16_0 : mxxMat2) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat2) (w_24_concat_1 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat5) (w_29_0 : mxxMat5) (w_30_0 : mxxMat2) (w_31_0 : mxxMat0) (w_35_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_27.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_6_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat2) (w_5_concat_1 : mxxMat3) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat2) (w_9_concat_1 : mxxMat3) (w_10_0 : mxxMat4) (w_13_0 : mxxMat3) (w_14_0 : mxxMat5) (w_15_0 : mxxMat5) (w_16_0 : mxxMat2) (w_17_0 : mxxMat0) : Prop :=
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

def parallel_generatedRoot_27 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_27.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_12_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_18_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_5_0 : mxxMat2),
        ∃ (w_5_concat_1 : mxxMat3),
          ∃ (w_7_0 : mxxMat0),
            ∃ (w_8_0 : mxxMat0),
              ∃ (w_9_0 : mxxMat2),
                ∃ (w_9_concat_1 : mxxMat3),
                  ∃ (w_10_0 : mxxMat4),
                    ∃ (w_13_0 : mxxMat3),
                      ∃ (w_14_0 : mxxMat5),
                        ∃ (w_15_0 : mxxMat5),
                          let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_15_0
                          ∃ (w_17_0 : mxxMat0),
                            let w_20_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_18_0 w_19_0
                            let w_21_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_20_0
                            ∃ (w_22_0 : mxxMat0),
                              ∃ (w_23_0 : mxxMat0),
                                ∃ (w_24_0 : mxxMat2),
                                  ∃ (w_24_concat_1 : mxxMat3),
                                    ∃ (w_27_0 : mxxMat3),
                                      ∃ (w_28_0 : mxxMat5),
                                        ∃ (w_29_0 : mxxMat5),
                                          let w_30_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_29_0
                                          ∃ (w_31_0 : mxxMat0),
                                            let w_33_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_32_0 w_19_0
                                            let w_34_0 : mxxMat0 := MxxRuntime.matrixAdd w_31_0 w_33_0
                                            ∃ (w_35_0 : mxxMat0),
                                              ∃ (w_36_0 : mxxMat0),
                                                ∃ (w_37_0 : mxxMat2),
                                                  ∃ (w_37_concat_1 : mxxMat3),
                                                    ∃ (w_38_0 : mxxMat4),
                                                      ∃ (w_40_0 : mxxMat0),
                                                        ∃ (w_42_0 : mxxMat3),
                                                          ∃ (w_43_0 : mxxMat5),
                                                            ∃ (w_44_0 : mxxMat5),
                                                              let w_45_0 : mxxMat2 := MxxRuntime.matrixMul w_38_0 w_44_0
                                                              ∃ (w_46_0 : mxxMat0),
                                                                ∃ (w_47_0 : mxxMat0),
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_38_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat0),
                                                                              ∃ (w_56_0 : mxxMat0),
                                                                                ∃ (w_57_0 : mxxMat0),
                                                                                  parallel_generatedRoot_27.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_12_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 ∧
    parallel_generatedRoot_27.constraints_1 backend w_25_0 w_26_0 w_16_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_concat_1 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 ∧
    parallel_generatedRoot_27.constraints_2 backend hashModel i_0 w_39_0 w_41_0 w_48_0 w_24_0 w_30_0 w_34_0 w_35_0 w_36_0 w_37_0 w_37_concat_1 w_38_0 w_40_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 ∧
    parallel_generatedRoot_27.constraints_3 w_45_0 w_46_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 outputs

abbrev parallel_parallel_generatedRoot_28_6.constraints_0 (w_0_0 : Fin 16 → mxxMat0) (w_3_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 16 → mxxMat0) (w_7_0 : Fin 16 → mxxMat0) (w_9_0 : Fin 16 → mxxMat0) (w_11_0 : Fin 16 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_28_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              parallel_parallel_generatedRoot_28_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_28_25.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_28_25.constraints_0 (w_1_0 : Fin 2 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_10_0 : Fin 2 → mxxMat0) (w_14_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_22_0 : Fin 2 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_28_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_28_25.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              sequential_parallel_generatedRoot_28_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_28.Witness where
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

abbrev parallel_generatedRoot_28.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat0) (w_1_0 : Fin 16 → mxxMat0) (w_2_0 : Fin 16 → mxxMat0) (w_3_0 : Fin 16 → mxxMat0) (w_4_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 16 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat0) (w_6_3 : Fin 2 → mxxMat0) (w_6_4 : Fin 2 → mxxMat0) (w_6_5 : Fin 2 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_25_1 : mxxMat0) (w_25_2 : mxxMat0) (w_25_3 : mxxMat0) (w_25_4 : mxxMat0) (w_25_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_28_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_28_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_28.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_28.Witness) : Prop :=
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
  parallel_generatedRoot_28.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_28 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_28.Witness,
    parallel_generatedRoot_28.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_31.InputType : Type :=
  Int × mxxMat0 × ByteArray × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × mxxMat1 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × ByteArray × mxxMat0 × mxxMat0 × Unit

abbrev parallel_generatedRoot_31.constraints_4 (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_31.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_46_0 : ByteArray) (w_55_0 : mxxMat0) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_56_0 : mxxMat3) (w_57_0 : mxxMat5) (w_58_0 : mxxMat5) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 1 2 0 25 w_53_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 52, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_46_0) w_54_0 ∧
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

abbrev parallel_generatedRoot_31.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_38_0 : Fin 8 → mxxMat0) (w_46_0 : ByteArray) (w_48_0 : mxxMat0) (w_12_0 : Int) (w_28_0 : mxxMat2) (w_36_0 : mxxMat2) (w_37_0 : mxxMat0) (w_39_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat2) (w_44_concat_1 : mxxMat3) (w_45_0 : mxxMat4) (w_47_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) : Prop :=
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 52, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_46_0) w_47_0 ∧
  MxxRuntime.concatRows w_47_0 w_48_0 w_49_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3

abbrev parallel_generatedRoot_31.constraints_1 (backend : MxxRuntime.BackendContext) (w_21_0 : Fin 8 → mxxMat0) (w_29_0 : Fin 8 → mxxMat0) (w_31_0 : Fin 8 → mxxMat0) (w_12_0 : Int) (w_19_0 : mxxMat2) (w_20_0 : mxxMat0) (w_22_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_30_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat3) (w_34_0 : mxxMat5) (w_35_0 : mxxMat5) : Prop :=
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

abbrev parallel_generatedRoot_31.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_6_0 : mxxMat0) (w_11_0 : Fin 8 → mxxMat0) (w_14_0 : Fin 8 → mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat2) (w_5_concat_1 : mxxMat3) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat2) (w_9_concat_1 : mxxMat3) (w_10_0 : mxxMat4) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat5) (w_18_0 : mxxMat5) : Prop :=
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

def parallel_generatedRoot_31 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_31.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.1
  let w_14_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_21_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_5_0 : mxxMat2),
        ∃ (w_5_concat_1 : mxxMat3),
          ∃ (w_7_0 : mxxMat0),
            ∃ (w_8_0 : mxxMat0),
              ∃ (w_9_0 : mxxMat2),
                ∃ (w_9_concat_1 : mxxMat3),
                  ∃ (w_10_0 : mxxMat4),
                    let w_12_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (2))
                    ∃ (w_13_0 : mxxMat0),
                      ∃ (w_15_0 : mxxMat0),
                        ∃ (w_16_0 : mxxMat3),
                          ∃ (w_17_0 : mxxMat5),
                            ∃ (w_18_0 : mxxMat5),
                              let w_19_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_18_0
                              ∃ (w_20_0 : mxxMat0),
                                ∃ (w_22_0 : mxxMat0),
                                  let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_22_0 w_23_0
                                  let w_25_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_24_0
                                  ∃ (w_26_0 : mxxMat0),
                                    ∃ (w_27_0 : mxxMat0),
                                      ∃ (w_28_0 : mxxMat2),
                                        ∃ (w_28_concat_1 : mxxMat3),
                                          ∃ (w_30_0 : mxxMat0),
                                            ∃ (w_32_0 : mxxMat0),
                                              ∃ (w_33_0 : mxxMat3),
                                                ∃ (w_34_0 : mxxMat5),
                                                  ∃ (w_35_0 : mxxMat5),
                                                    let w_36_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_35_0
                                                    ∃ (w_37_0 : mxxMat0),
                                                      ∃ (w_39_0 : mxxMat0),
                                                        let w_40_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_39_0 w_23_0
                                                        let w_41_0 : mxxMat0 := MxxRuntime.matrixAdd w_37_0 w_40_0
                                                        ∃ (w_42_0 : mxxMat0),
                                                          ∃ (w_43_0 : mxxMat0),
                                                            ∃ (w_44_0 : mxxMat2),
                                                              ∃ (w_44_concat_1 : mxxMat3),
                                                                ∃ (w_45_0 : mxxMat4),
                                                                  ∃ (w_47_0 : mxxMat0),
                                                                    ∃ (w_49_0 : mxxMat3),
                                                                      ∃ (w_50_0 : mxxMat5),
                                                                        ∃ (w_51_0 : mxxMat5),
                                                                          let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_45_0 w_51_0
                                                                          ∃ (w_53_0 : mxxMat0),
                                                                            ∃ (w_54_0 : mxxMat0),
                                                                              ∃ (w_56_0 : mxxMat3),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  ∃ (w_58_0 : mxxMat5),
                                                                                    let w_59_0 : mxxMat2 := MxxRuntime.matrixMul w_45_0 w_58_0
                                                                                    ∃ (w_60_0 : mxxMat0),
                                                                                      ∃ (w_61_0 : mxxMat0),
                                                                                        ∃ (w_62_0 : mxxMat0),
                                                                                          ∃ (w_63_0 : mxxMat0),
                                                                                            ∃ (w_64_0 : mxxMat0),
                                                                                              parallel_generatedRoot_31.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_14_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_31.constraints_1 backend w_21_0 w_29_0 w_31_0 w_12_0 w_19_0 w_20_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_concat_1 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 ∧
    parallel_generatedRoot_31.constraints_2 backend hashModel i_0 w_38_0 w_46_0 w_48_0 w_12_0 w_28_0 w_36_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_concat_1 w_45_0 w_47_0 w_49_0 w_50_0 w_51_0 ∧
    parallel_generatedRoot_31.constraints_3 backend hashModel i_0 w_46_0 w_55_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 ∧
    parallel_generatedRoot_31.constraints_4 w_52_0 w_53_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_parallel_generatedRoot_32_6.constraints_0 (w_0_0 : Fin 16 → mxxMat0) (w_3_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 16 → mxxMat0) (w_7_0 : Fin 16 → mxxMat0) (w_9_0 : Fin 16 → mxxMat0) (w_11_0 : Fin 16 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_32_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              parallel_parallel_generatedRoot_32_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_32_25.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_32_25.constraints_0 (w_1_0 : Fin 2 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_10_0 : Fin 2 → mxxMat0) (w_14_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_22_0 : Fin 2 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_32_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_32_25.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              sequential_parallel_generatedRoot_32_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_32.Witness where
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

abbrev parallel_generatedRoot_32.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat0) (w_1_0 : Fin 16 → mxxMat0) (w_2_0 : Fin 16 → mxxMat0) (w_3_0 : Fin 16 → mxxMat0) (w_4_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 16 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat0) (w_6_3 : Fin 2 → mxxMat0) (w_6_4 : Fin 2 → mxxMat0) (w_6_5 : Fin 2 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_25_1 : mxxMat0) (w_25_2 : mxxMat0) (w_25_3 : mxxMat0) (w_25_4 : mxxMat0) (w_25_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_32_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_32_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_32.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_32.Witness) : Prop :=
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
  parallel_generatedRoot_32.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_32 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_32.Witness,
    parallel_generatedRoot_32.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_33_18.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_33_18.constraints_4 (w_75_0 : Fin 8 → mxxMat0) (w_79_0 : Fin 8 → mxxMat0) (w_81_0 : Fin 8 → mxxMat0) (w_83_0 : Fin 8 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_33_18.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_parallel_generatedRoot_33_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_parallel_generatedRoot_33_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
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

abbrev sequential_parallel_generatedRoot_33_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
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

def sequential_parallel_generatedRoot_33_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_33_18.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_parallel_generatedRoot_33_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_33_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_33_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_33_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_33_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev parallel_generatedRoot_33.InputType : Type :=
  (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_33.Witness where
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
  w_23_0 : mxxMat2
  w_23_concat_1 : mxxMat3
  w_24_0 : mxxMat0
  w_25_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat2
  w_27_concat_1 : mxxMat3
  w_28_0 : mxxMat4
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat2
  w_32_concat_1 : mxxMat3
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat2
  w_36_concat_1 : mxxMat3
  w_37_0 : mxxMat4
  w_38_0 : mxxMat3
  w_39_0 : mxxMat5
  w_40_0 : mxxMat5
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat3
  w_45_0 : mxxMat5
  w_46_0 : mxxMat5
  w_48_0 : mxxMat0
  w_50_0 : mxxMat3
  w_51_0 : mxxMat5
  w_52_0 : mxxMat5
  w_54_0 : mxxMat0
  w_55_0 : mxxMat0
  w_56_0 : mxxMat3
  w_57_0 : mxxMat5
  w_58_0 : mxxMat5
  w_60_0 : mxxMat0
  w_62_0 : mxxMat0
  w_64_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0
  w_70_0 : mxxMat0
  w_71_0 : mxxMat0

abbrev parallel_generatedRoot_33.constraints_4 (w_49_0 : mxxMat0) (w_53_0 : mxxMat2) (w_59_0 : mxxMat2) (w_61_0 : mxxMat0) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_33.constraints_3 (w_41_0 : mxxMat2) (w_47_0 : mxxMat2) (w_59_0 : mxxMat2) (w_62_0 : mxxMat0) (w_64_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_33.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat0) (w_18_9 : mxxMat0) (w_47_0 : mxxMat2) (w_48_0 : mxxMat0) (w_50_0 : mxxMat3) (w_51_0 : mxxMat5) (w_52_0 : mxxMat5) (w_53_0 : mxxMat2) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat3) (w_57_0 : mxxMat5) (w_58_0 : mxxMat5) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_33.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_6 : mxxMat0) (w_18_8 : mxxMat0) (w_32_0 : mxxMat2) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat2) (w_36_concat_1 : mxxMat3) (w_37_0 : mxxMat4) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat3) (w_45_0 : mxxMat5) (w_46_0 : mxxMat5) : Prop :=
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

abbrev parallel_generatedRoot_33.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_3_0 : Fin 8 → mxxMat0) (w_5_0 : Fin 8 → mxxMat0) (w_7_0 : Fin 8 → mxxMat0) (w_9_0 : Fin 8 → mxxMat0) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_0_0 : mxxMat0) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_18_2 : mxxMat0) (w_18_3 : mxxMat0) (w_18_4 : mxxMat0) (w_18_5 : mxxMat0) (w_18_6 : mxxMat0) (w_18_7 : mxxMat0) (w_18_8 : mxxMat0) (w_18_9 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : Int) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat2) (w_23_concat_1 : mxxMat3) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat2) (w_27_concat_1 : mxxMat3) (w_28_0 : mxxMat4) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat2) (w_32_concat_1 : mxxMat3) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_33_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
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

abbrev parallel_generatedRoot_33.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_33.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_33.Witness) : Prop :=
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
  let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat2 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat2 := MxxRuntime.matrixMul w_28_0 w_58_0
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
  parallel_generatedRoot_33.constraints_0 backend hashModel tape path params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_33.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_33.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_33.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_33.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_33 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_33.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_33.Witness,
    parallel_generatedRoot_33.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_34_18.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_34_18.constraints_4 (w_75_0 : Fin 8 → mxxMat0) (w_79_0 : Fin 8 → mxxMat0) (w_81_0 : Fin 8 → mxxMat0) (w_83_0 : Fin 8 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_34_18.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_parallel_generatedRoot_34_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_parallel_generatedRoot_34_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
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

abbrev sequential_parallel_generatedRoot_34_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
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

def sequential_parallel_generatedRoot_34_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_34_18.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_parallel_generatedRoot_34_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_34_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_34_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_34_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_34_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev parallel_generatedRoot_34.InputType : Type :=
  (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_34.Witness where
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
  w_23_0 : mxxMat2
  w_23_concat_1 : mxxMat3
  w_24_0 : mxxMat0
  w_25_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat2
  w_27_concat_1 : mxxMat3
  w_28_0 : mxxMat4
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat2
  w_32_concat_1 : mxxMat3
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat2
  w_36_concat_1 : mxxMat3
  w_37_0 : mxxMat4
  w_38_0 : mxxMat3
  w_39_0 : mxxMat5
  w_40_0 : mxxMat5
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat3
  w_45_0 : mxxMat5
  w_46_0 : mxxMat5
  w_48_0 : mxxMat0
  w_50_0 : mxxMat3
  w_51_0 : mxxMat5
  w_52_0 : mxxMat5
  w_54_0 : mxxMat0
  w_55_0 : mxxMat0
  w_56_0 : mxxMat3
  w_57_0 : mxxMat5
  w_58_0 : mxxMat5
  w_60_0 : mxxMat0
  w_62_0 : mxxMat0
  w_64_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0
  w_70_0 : mxxMat0
  w_71_0 : mxxMat0

abbrev parallel_generatedRoot_34.constraints_4 (w_49_0 : mxxMat0) (w_53_0 : mxxMat2) (w_59_0 : mxxMat2) (w_61_0 : mxxMat0) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_34.constraints_3 (w_41_0 : mxxMat2) (w_47_0 : mxxMat2) (w_59_0 : mxxMat2) (w_62_0 : mxxMat0) (w_64_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_34.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat0) (w_18_9 : mxxMat0) (w_47_0 : mxxMat2) (w_48_0 : mxxMat0) (w_50_0 : mxxMat3) (w_51_0 : mxxMat5) (w_52_0 : mxxMat5) (w_53_0 : mxxMat2) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat3) (w_57_0 : mxxMat5) (w_58_0 : mxxMat5) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_34.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_6 : mxxMat0) (w_18_8 : mxxMat0) (w_32_0 : mxxMat2) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat2) (w_36_concat_1 : mxxMat3) (w_37_0 : mxxMat4) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat3) (w_45_0 : mxxMat5) (w_46_0 : mxxMat5) : Prop :=
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

abbrev parallel_generatedRoot_34.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_3_0 : Fin 8 → mxxMat0) (w_5_0 : Fin 8 → mxxMat0) (w_7_0 : Fin 8 → mxxMat0) (w_9_0 : Fin 8 → mxxMat0) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_0_0 : mxxMat0) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_18_2 : mxxMat0) (w_18_3 : mxxMat0) (w_18_4 : mxxMat0) (w_18_5 : mxxMat0) (w_18_6 : mxxMat0) (w_18_7 : mxxMat0) (w_18_8 : mxxMat0) (w_18_9 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : Int) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat2) (w_23_concat_1 : mxxMat3) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat2) (w_27_concat_1 : mxxMat3) (w_28_0 : mxxMat4) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat2) (w_32_concat_1 : mxxMat3) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_34_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
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

abbrev parallel_generatedRoot_34.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_34.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_34.Witness) : Prop :=
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
  let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat2 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat2 := MxxRuntime.matrixMul w_28_0 w_58_0
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
  parallel_generatedRoot_34.constraints_0 backend hashModel tape path params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_34.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_34.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_34.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_34.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_34 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_34.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_34.Witness,
    parallel_generatedRoot_34.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_35.constraints_0 (w_2_0 : mxxMat0) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_14_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_2_0, (w_5_0, (w_8_0, (w_11_0, (w_14_0, (w_17_0, ()))))))

def parallel_generatedRoot_35 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_3_0 : mxxMat0 := inputs.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.1
  let w_12_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_0_0 w_1_0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  let w_7_0 : Int := (Int.ofNat i_0)
  let w_8_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_6_0 w_7_0
  let w_10_0 : Int := (Int.ofNat i_0)
  let w_11_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_9_0 w_10_0
  let w_13_0 : Int := (Int.ofNat i_0)
  let w_14_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_12_0 w_13_0
  let w_16_0 : Int := (Int.ofNat i_0)
  let w_17_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_15_0 w_16_0
  parallel_generatedRoot_35.constraints_0 w_2_0 w_5_0 w_8_0 w_11_0 w_14_0 w_17_0 outputs

abbrev sequential_generatedRoot_54.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × Unit

abbrev sequential_generatedRoot_54.constraints_0 (w_1_0 : Fin 8 → mxxMat0) (w_6_0 : Fin 8 → mxxMat0) (w_10_0 : Fin 8 → mxxMat0) (w_14_0 : Fin 8 → mxxMat0) (w_18_0 : Fin 8 → mxxMat0) (w_22_0 : Fin 8 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def sequential_generatedRoot_54 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_54.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              sequential_generatedRoot_54.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev sequential_generatedRoot_161.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 4 → mxxMat0) × ByteArray × (Fin 4 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit

abbrev sequential_generatedRoot_161.constraints_4 (w_75_0 : Fin 4 → mxxMat0) (w_79_0 : Fin 4 → mxxMat0) (w_81_0 : Fin 4 → mxxMat0) (w_83_0 : Fin 4 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_161.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_generatedRoot_161.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_161.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_161.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 4 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 4 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
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

def sequential_generatedRoot_161 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_161.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_generatedRoot_161.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_161.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_161.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_161.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_161.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_240.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × ByteArray × (Fin 1 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × Unit

abbrev sequential_generatedRoot_240.constraints_4 (w_75_0 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_81_0 : Fin 1 → mxxMat0) (w_83_0 : Fin 1 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_generatedRoot_240.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_generatedRoot_240.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_240.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_26_0 w_27_0 ∧
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

abbrev sequential_generatedRoot_240.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0))

def sequential_generatedRoot_240 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_240.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
  let w_1_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_generatedRoot_240.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_240.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_240.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_240.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_240.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_298.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × ByteArray × (Fin 1 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × Unit

abbrev sequential_generatedRoot_298.constraints_4 (w_75_0 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_81_0 : Fin 1 → mxxMat0) (w_83_0 : Fin 1 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_generatedRoot_298.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_generatedRoot_298.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_298.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_26_0 w_27_0 ∧
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

abbrev sequential_generatedRoot_298.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0))

def sequential_generatedRoot_298 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_298.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
  let w_1_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_generatedRoot_298.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_298.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_298.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_298.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_298.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_356.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × ByteArray × (Fin 1 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × Unit

abbrev sequential_generatedRoot_356.constraints_4 (w_75_0 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_81_0 : Fin 1 → mxxMat0) (w_83_0 : Fin 1 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_generatedRoot_356.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_generatedRoot_356.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_356.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_26_0 w_27_0 ∧
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

abbrev sequential_generatedRoot_356.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0))

def sequential_generatedRoot_356 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_356.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
  let w_1_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(7)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_generatedRoot_356.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_356.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_356.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_356.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_356.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_414.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × ByteArray × (Fin 1 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × Unit

abbrev sequential_generatedRoot_414.constraints_4 (w_75_0 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_81_0 : Fin 1 → mxxMat0) (w_83_0 : Fin 1 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_generatedRoot_414.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_generatedRoot_414.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_414.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_26_0 w_27_0 ∧
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

abbrev sequential_generatedRoot_414.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0))

def sequential_generatedRoot_414 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_414.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
  let w_1_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(9)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_generatedRoot_414.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_414.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_414.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_414.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_414.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_472.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × ByteArray × (Fin 1 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × Unit

abbrev sequential_generatedRoot_472.constraints_4 (w_75_0 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_81_0 : Fin 1 → mxxMat0) (w_83_0 : Fin 1 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_generatedRoot_472.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_generatedRoot_472.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_472.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_26_0 w_27_0 ∧
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

abbrev sequential_generatedRoot_472.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0))

def sequential_generatedRoot_472 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_472.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
  let w_1_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(11)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_generatedRoot_472.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_472.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_472.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_472.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_472.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_530.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × ByteArray × (Fin 1 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × Unit

abbrev sequential_generatedRoot_530.constraints_4 (w_75_0 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_81_0 : Fin 1 → mxxMat0) (w_83_0 : Fin 1 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_generatedRoot_530.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_generatedRoot_530.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_530.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_26_0 w_27_0 ∧
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

abbrev sequential_generatedRoot_530.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0))

def sequential_generatedRoot_530 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_530.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
  let w_1_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(13)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_generatedRoot_530.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_530.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_530.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_530.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_530.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_588.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × ByteArray × (Fin 1 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × Unit

abbrev sequential_generatedRoot_588.constraints_4 (w_75_0 : Fin 1 → mxxMat0) (w_79_0 : Fin 1 → mxxMat0) (w_81_0 : Fin 1 → mxxMat0) (w_83_0 : Fin 1 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_41_0 2 3 0 25 w_70_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_58_0 2 3 0 25 w_73_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_75_0 w_77_0 w_78_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_79_0 w_77_0 w_80_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_81_0 w_77_0 w_82_0 ∧
  0 ≤ w_77_0 ∧
  w_77_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_83_0 w_77_0 w_84_0 ∧
  outputs = (w_45_0, (w_62_0, (w_65_0, (w_68_0, (w_71_0, (w_74_0, (w_78_0, (w_80_0, (w_82_0, (w_84_0, ()))))))))))

abbrev sequential_generatedRoot_588.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_generatedRoot_588.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_generatedRoot_588.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_24_0 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_26_0 w_27_0 ∧
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

abbrev sequential_generatedRoot_588.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_4_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_4_0) w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_2_0 w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_4_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_5_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_4_0) w_12_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_13_concat_1 ∧
  MxxRuntime.concatRows w_13_concat_1 w_12_0 w_13_0 ∧
  MxxRuntime.concatColumns w_8_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_16_0 w_18_0 w_21_concat_1 ∧
  MxxRuntime.concatRows w_21_concat_1 w_20_0 w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0))

def sequential_generatedRoot_588 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_588.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
  let w_1_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15)])
    ∃ (w_6_0 : mxxMat0),
      ∃ (w_7_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_generatedRoot_588.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_588.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_588.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_588.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_588.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev parallel_parallel_generatedRoot_658_31.constraints_1 (hashModel : MxxRuntime.HashModel) (w_29_0 : ByteArray) (w_35_0 : mxxMat0) (w_38_0 : mxxMat0) (w_41_0 : mxxMat1) (w_43_0 : Fin 5 → Int) (w_3_0 : Int) (w_21_0 : Int) (w_23_0 : mxxMat0) (w_27_0 : mxxMat0) (w_31_0 : mxxMat0) (w_34_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_48_0 : Int) (w_49_0 : Int) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_21_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_29_0) w_36_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_3_0 [w_35_0, w_36_0] w_37_0 ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_21_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_29_0) w_39_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_3_0 [w_38_0, w_39_0] w_40_0 ∧
  0 ≤ w_48_0 ∧
  w_48_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_48_0 w_49_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_3_0 [w_41_0, w_52_0] w_53_0 ∧
  outputs = (w_23_0, (w_27_0, (w_31_0, (w_34_0, (w_37_0, (w_40_0, (w_53_0, ())))))))

abbrev parallel_parallel_generatedRoot_658_31.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_4_0 : mxxMat0) (w_5_0 : Fin 4 → mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : Fin 4 → mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : ByteArray) (w_32_0 : mxxMat0) (w_3_0 : Int) (w_21_0 : Int) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) : Prop :=
  0 ≤ ((2 * (Int.ofNat i_1))) ∧
  0 ≤ ((1 + (2 * (Int.ofNat i_1)))) ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_21_0 w_22_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_3_0 [w_4_0, w_22_0] w_23_0 ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_21_0 w_26_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_3_0 [w_24_0, w_26_0] w_27_0 ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_21_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_29_0) w_30_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_3_0 [w_28_0, w_30_0] w_31_0 ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_21_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_29_0) w_33_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_3_0 [w_32_0, w_33_0] w_34_0 ∧
  0 ≤ w_21_0 ∧
  w_21_0 < 2 ^ 64

def parallel_parallel_generatedRoot_658_31 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × mxxMat0 × (Fin 4 → mxxMat0) × Int × Int × mxxMat0 × (Fin 4 → mxxMat0) × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × (Fin 5 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Int := inputs.1
  let w_4_0 : mxxMat0 := inputs.2.1
  let w_5_0 : Fin 4 → mxxMat0 := inputs.2.2.1
  let w_6_0 : Int := inputs.2.2.2.1
  let w_10_0 : Int := inputs.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_25_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_28_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_29_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.1
  let w_32_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_35_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 5 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : Int := (Int.ofNat i_1)
  let w_2_0 : Bool := decide (w_0_0 < w_1_0)
  let w_3_0 : Int := if w_2_0 then 1 else 0
  let w_7_0 : Int := 4
  let w_8_0 : Int := (w_6_0 * w_7_0)
  let w_9_0 : Int := 0
  let w_11_0 : Bool := decide ((w_10_0 / (2 ^ (Int.toNat ((2 * (Int.ofNat i_1)))))) % 2 = 1)
  let w_12_0 : Int := if w_11_0 then 1 else 0
  let w_13_0 : Int := 1
  let w_14_0 : Int := (w_12_0 * w_13_0)
  let w_15_0 : Int := (w_9_0 + w_14_0)
  let w_16_0 : Bool := decide ((w_10_0 / (2 ^ (Int.toNat ((1 + (2 * (Int.ofNat i_1))))))) % 2 = 1)
  let w_17_0 : Int := if w_16_0 then 1 else 0
  let w_18_0 : Int := 2
  let w_19_0 : Int := (w_17_0 * w_18_0)
  let w_20_0 : Int := (w_15_0 + w_19_0)
  let w_21_0 : Int := (w_8_0 + w_20_0)
  ∃ (w_22_0 : mxxMat0),
    ∃ (w_23_0 : mxxMat0),
      ∃ (w_26_0 : mxxMat0),
        ∃ (w_27_0 : mxxMat0),
          ∃ (w_30_0 : mxxMat0),
            ∃ (w_31_0 : mxxMat0),
              ∃ (w_33_0 : mxxMat0),
                ∃ (w_34_0 : mxxMat0),
                  ∃ (w_36_0 : mxxMat0),
                    ∃ (w_37_0 : mxxMat0),
                      ∃ (w_39_0 : mxxMat0),
                        ∃ (w_40_0 : mxxMat0),
                          let w_42_0 : mxxMat1 := (1 : mxxMat1)
                          let w_44_0 : Int := 5
                          let w_45_0 : Int := (w_6_0 * w_44_0)
                          let w_46_0 : Int := (w_45_0 + w_20_0)
                          let w_47_0 : Int := 1
                          let w_48_0 : Int := (w_46_0 + w_47_0)
                          ∃ (w_49_0 : Int),
                            let w_50_0 : Int := 1
                            let w_51_0 : Int := (w_49_0 * w_50_0)
                            let w_52_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_42_0 w_51_0
                            ∃ (w_53_0 : mxxMat1),
                              parallel_parallel_generatedRoot_658_31.constraints_0 hashModel i_1 w_4_0 w_5_0 w_24_0 w_25_0 w_28_0 w_29_0 w_32_0 w_3_0 w_21_0 w_22_0 w_23_0 w_26_0 w_27_0 w_30_0 w_31_0 w_33_0 w_34_0 ∧
    parallel_parallel_generatedRoot_658_31.constraints_1 hashModel w_29_0 w_35_0 w_38_0 w_41_0 w_43_0 w_3_0 w_21_0 w_23_0 w_27_0 w_31_0 w_34_0 w_36_0 w_37_0 w_39_0 w_40_0 w_48_0 w_49_0 w_52_0 w_53_0 outputs

abbrev parallel_parallel_generatedRoot_658_32.constraints_2 (w_25_0 : Fin 4 → mxxMat1) (w_7_0 : Int) (w_14_0 : Int) (w_26_0 : mxxMat1) (w_30_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat0) (w_53_0 : mxxMat0) (w_56_0 : mxxMat0) (w_59_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat1) (w_65_0 : mxxMat1) (w_66_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_56_0, w_62_0] w_63_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_14_0 w_64_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_26_0, w_65_0] w_66_0 ∧
  outputs = (w_30_0, (w_43_0, (w_49_0, (w_53_0, (w_59_0, (w_63_0, (w_66_0, ())))))))

abbrev parallel_parallel_generatedRoot_658_32.constraints_1 (backend : MxxRuntime.BackendContext) (w_12_0 : Fin 4 → mxxMat0) (w_18_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_31_0 : Fin 4 → mxxMat0) (w_35_0 : Fin 4 → mxxMat0) (w_7_0 : Int) (w_10_0 : Int) (w_14_0 : Int) (w_19_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat6) (w_40_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_46_0 : mxxMat0) (w_48_0 : mxxMat0) (w_49_0 : mxxMat0) (w_52_0 : mxxMat0) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_56_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat0) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_36_0 w_37_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_14_0 w_40_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_19_0, w_42_0] w_43_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_10_0 w_44_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_31_0 w_10_0 w_46_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_44_0, w_48_0] w_49_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_46_0, w_52_0] w_53_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_10_0 w_54_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_35_0 w_10_0 w_56_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_54_0, w_58_0] w_59_0

abbrev parallel_parallel_generatedRoot_658_32.constraints_0 (backend : MxxRuntime.BackendContext) (w_8_0 : Fin 4 → mxxMat0) (w_12_0 : Fin 4 → mxxMat0) (w_18_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_25_0 : Fin 4 → mxxMat1) (w_31_0 : Fin 4 → mxxMat0) (w_35_0 : Fin 4 → mxxMat0) (w_7_0 : Int) (w_10_0 : Int) (w_11_0 : mxxMat0) (w_13_0 : Int) (w_14_0 : Int) (w_15_0 : mxxMat0) (w_16_0 : mxxMat6) (w_19_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat6) (w_26_0 : mxxMat1) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat6) (w_36_0 : mxxMat0) : Prop :=
  0 ≤ w_10_0 ∧
  w_10_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_10_0 w_11_0 ∧
  w_13_0 ≠ 0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_14_0 w_15_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_15_0 w_16_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_10_0 w_19_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_14_0 w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_21_0 w_22_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_10_0 w_26_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_14_0 w_27_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_11_0, w_29_0] w_30_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_31_0 w_14_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_32_0 w_33_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_35_0 w_14_0 w_36_0

def parallel_parallel_generatedRoot_658_32 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_5_0 : Int := inputs.1
  let w_8_0 : Fin 4 → mxxMat0 := inputs.2.1
  let w_12_0 : Fin 4 → mxxMat0 := inputs.2.2.1
  let w_18_0 : Fin 4 → mxxMat0 := inputs.2.2.2.1
  let w_20_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.1
  let w_25_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_31_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_35_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_0_0 : Int := (Int.ofNat i_1)
  let w_1_0 : Int := 2
  let w_2_0 : Int := (w_0_0 * w_1_0)
  let w_3_0 : Int := 1
  let w_4_0 : Int := (w_2_0 + w_3_0)
  let w_6_0 : Bool := decide (w_4_0 < w_5_0)
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_9_0 : Int := 2
  let w_10_0 : Int := (w_0_0 * w_9_0)
  ∃ (w_11_0 : mxxMat0),
    let w_13_0 : Int := 4
    let w_14_0 : Int := (w_4_0 % w_13_0)
    ∃ (w_15_0 : mxxMat0),
      ∃ (w_16_0 : mxxMat6),
        let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_11_0 w_16_0
        ∃ (w_19_0 : mxxMat0),
          ∃ (w_21_0 : mxxMat0),
            ∃ (w_22_0 : mxxMat6),
              let w_23_0 : mxxMat0 := MxxRuntime.matrixMul w_19_0 w_22_0
              let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_23_0
              ∃ (w_26_0 : mxxMat1),
                ∃ (w_27_0 : mxxMat0),
                  let w_28_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                  let w_29_0 : mxxMat0 := MxxRuntime.matrixAdd w_24_0 w_28_0
                  ∃ (w_30_0 : mxxMat0),
                    ∃ (w_32_0 : mxxMat0),
                      ∃ (w_33_0 : mxxMat6),
                        let w_34_0 : mxxMat0 := MxxRuntime.matrixMul w_11_0 w_33_0
                        ∃ (w_36_0 : mxxMat0),
                          ∃ (w_37_0 : mxxMat6),
                            let w_38_0 : mxxMat0 := MxxRuntime.matrixMul w_19_0 w_37_0
                            let w_39_0 : mxxMat0 := MxxRuntime.matrixAdd w_34_0 w_38_0
                            ∃ (w_40_0 : mxxMat0),
                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_26_0 w_40_0
                              let w_42_0 : mxxMat0 := MxxRuntime.matrixAdd w_39_0 w_41_0
                              ∃ (w_43_0 : mxxMat0),
                                ∃ (w_44_0 : mxxMat0),
                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixMul w_44_0 w_16_0
                                  ∃ (w_46_0 : mxxMat0),
                                    let w_47_0 : mxxMat0 := MxxRuntime.matrixMul w_46_0 w_22_0
                                    let w_48_0 : mxxMat0 := MxxRuntime.matrixAdd w_45_0 w_47_0
                                    ∃ (w_49_0 : mxxMat0),
                                      let w_50_0 : mxxMat0 := MxxRuntime.matrixMul w_44_0 w_33_0
                                      let w_51_0 : mxxMat0 := MxxRuntime.matrixMul w_46_0 w_37_0
                                      let w_52_0 : mxxMat0 := MxxRuntime.matrixAdd w_50_0 w_51_0
                                      ∃ (w_53_0 : mxxMat0),
                                        ∃ (w_54_0 : mxxMat0),
                                          let w_55_0 : mxxMat0 := MxxRuntime.matrixMul w_54_0 w_16_0
                                          ∃ (w_56_0 : mxxMat0),
                                            let w_57_0 : mxxMat0 := MxxRuntime.matrixMul w_56_0 w_22_0
                                            let w_58_0 : mxxMat0 := MxxRuntime.matrixAdd w_55_0 w_57_0
                                            ∃ (w_59_0 : mxxMat0),
                                              let w_60_0 : mxxMat0 := MxxRuntime.matrixMul w_54_0 w_33_0
                                              let w_61_0 : mxxMat0 := MxxRuntime.matrixMul w_56_0 w_37_0
                                              let w_62_0 : mxxMat0 := MxxRuntime.matrixAdd w_60_0 w_61_0
                                              ∃ (w_63_0 : mxxMat0),
                                                ∃ (w_64_0 : mxxMat1),
                                                  let w_65_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_26_0 w_64_0
                                                  ∃ (w_66_0 : mxxMat1),
                                                    parallel_parallel_generatedRoot_658_32.constraints_0 backend w_8_0 w_12_0 w_18_0 w_20_0 w_25_0 w_31_0 w_35_0 w_7_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_19_0 w_21_0 w_22_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_36_0 ∧
    parallel_parallel_generatedRoot_658_32.constraints_1 backend w_12_0 w_18_0 w_20_0 w_31_0 w_35_0 w_7_0 w_10_0 w_14_0 w_19_0 w_36_0 w_37_0 w_40_0 w_42_0 w_43_0 w_44_0 w_46_0 w_48_0 w_49_0 w_52_0 w_53_0 w_54_0 w_56_0 w_58_0 w_59_0 ∧
    parallel_parallel_generatedRoot_658_32.constraints_2 w_25_0 w_7_0 w_14_0 w_26_0 w_30_0 w_43_0 w_49_0 w_53_0 w_56_0 w_59_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 outputs

abbrev parallel_parallel_generatedRoot_658_33.constraints_2 (w_25_0 : Fin 2 → mxxMat1) (w_7_0 : Int) (w_14_0 : Int) (w_26_0 : mxxMat1) (w_30_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat0) (w_53_0 : mxxMat0) (w_56_0 : mxxMat0) (w_59_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat1) (w_65_0 : mxxMat1) (w_66_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_56_0, w_62_0] w_63_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_14_0 w_64_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_26_0, w_65_0] w_66_0 ∧
  outputs = (w_30_0, (w_43_0, (w_49_0, (w_53_0, (w_59_0, (w_63_0, (w_66_0, ())))))))

abbrev parallel_parallel_generatedRoot_658_33.constraints_1 (backend : MxxRuntime.BackendContext) (w_12_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_20_0 : Fin 2 → mxxMat0) (w_31_0 : Fin 2 → mxxMat0) (w_35_0 : Fin 2 → mxxMat0) (w_7_0 : Int) (w_10_0 : Int) (w_14_0 : Int) (w_19_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat6) (w_40_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_46_0 : mxxMat0) (w_48_0 : mxxMat0) (w_49_0 : mxxMat0) (w_52_0 : mxxMat0) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_56_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat0) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_36_0 w_37_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_14_0 w_40_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_19_0, w_42_0] w_43_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_10_0 w_44_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_31_0 w_10_0 w_46_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_44_0, w_48_0] w_49_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_46_0, w_52_0] w_53_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_10_0 w_54_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_35_0 w_10_0 w_56_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_54_0, w_58_0] w_59_0

abbrev parallel_parallel_generatedRoot_658_33.constraints_0 (backend : MxxRuntime.BackendContext) (w_8_0 : Fin 2 → mxxMat0) (w_12_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_20_0 : Fin 2 → mxxMat0) (w_25_0 : Fin 2 → mxxMat1) (w_31_0 : Fin 2 → mxxMat0) (w_35_0 : Fin 2 → mxxMat0) (w_7_0 : Int) (w_10_0 : Int) (w_11_0 : mxxMat0) (w_13_0 : Int) (w_14_0 : Int) (w_15_0 : mxxMat0) (w_16_0 : mxxMat6) (w_19_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat6) (w_26_0 : mxxMat1) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat6) (w_36_0 : mxxMat0) : Prop :=
  0 ≤ w_10_0 ∧
  w_10_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_10_0 w_11_0 ∧
  w_13_0 ≠ 0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_14_0 w_15_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_15_0 w_16_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_10_0 w_19_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_14_0 w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_21_0 w_22_0 ∧
  0 ≤ w_10_0 ∧
  w_10_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_10_0 w_26_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_14_0 w_27_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_11_0, w_29_0] w_30_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_31_0 w_14_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_32_0 w_33_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_35_0 w_14_0 w_36_0

def parallel_parallel_generatedRoot_658_33 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_5_0 : Int := inputs.1
  let w_8_0 : Fin 2 → mxxMat0 := inputs.2.1
  let w_12_0 : Fin 2 → mxxMat0 := inputs.2.2.1
  let w_18_0 : Fin 2 → mxxMat0 := inputs.2.2.2.1
  let w_20_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.1
  let w_25_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_31_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_35_0 : Fin 2 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_0_0 : Int := (Int.ofNat i_1)
  let w_1_0 : Int := 2
  let w_2_0 : Int := (w_0_0 * w_1_0)
  let w_3_0 : Int := 1
  let w_4_0 : Int := (w_2_0 + w_3_0)
  let w_6_0 : Bool := decide (w_4_0 < w_5_0)
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_9_0 : Int := 2
  let w_10_0 : Int := (w_0_0 * w_9_0)
  ∃ (w_11_0 : mxxMat0),
    let w_13_0 : Int := 2
    let w_14_0 : Int := (w_4_0 % w_13_0)
    ∃ (w_15_0 : mxxMat0),
      ∃ (w_16_0 : mxxMat6),
        let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_11_0 w_16_0
        ∃ (w_19_0 : mxxMat0),
          ∃ (w_21_0 : mxxMat0),
            ∃ (w_22_0 : mxxMat6),
              let w_23_0 : mxxMat0 := MxxRuntime.matrixMul w_19_0 w_22_0
              let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_23_0
              ∃ (w_26_0 : mxxMat1),
                ∃ (w_27_0 : mxxMat0),
                  let w_28_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                  let w_29_0 : mxxMat0 := MxxRuntime.matrixAdd w_24_0 w_28_0
                  ∃ (w_30_0 : mxxMat0),
                    ∃ (w_32_0 : mxxMat0),
                      ∃ (w_33_0 : mxxMat6),
                        let w_34_0 : mxxMat0 := MxxRuntime.matrixMul w_11_0 w_33_0
                        ∃ (w_36_0 : mxxMat0),
                          ∃ (w_37_0 : mxxMat6),
                            let w_38_0 : mxxMat0 := MxxRuntime.matrixMul w_19_0 w_37_0
                            let w_39_0 : mxxMat0 := MxxRuntime.matrixAdd w_34_0 w_38_0
                            ∃ (w_40_0 : mxxMat0),
                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_26_0 w_40_0
                              let w_42_0 : mxxMat0 := MxxRuntime.matrixAdd w_39_0 w_41_0
                              ∃ (w_43_0 : mxxMat0),
                                ∃ (w_44_0 : mxxMat0),
                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixMul w_44_0 w_16_0
                                  ∃ (w_46_0 : mxxMat0),
                                    let w_47_0 : mxxMat0 := MxxRuntime.matrixMul w_46_0 w_22_0
                                    let w_48_0 : mxxMat0 := MxxRuntime.matrixAdd w_45_0 w_47_0
                                    ∃ (w_49_0 : mxxMat0),
                                      let w_50_0 : mxxMat0 := MxxRuntime.matrixMul w_44_0 w_33_0
                                      let w_51_0 : mxxMat0 := MxxRuntime.matrixMul w_46_0 w_37_0
                                      let w_52_0 : mxxMat0 := MxxRuntime.matrixAdd w_50_0 w_51_0
                                      ∃ (w_53_0 : mxxMat0),
                                        ∃ (w_54_0 : mxxMat0),
                                          let w_55_0 : mxxMat0 := MxxRuntime.matrixMul w_54_0 w_16_0
                                          ∃ (w_56_0 : mxxMat0),
                                            let w_57_0 : mxxMat0 := MxxRuntime.matrixMul w_56_0 w_22_0
                                            let w_58_0 : mxxMat0 := MxxRuntime.matrixAdd w_55_0 w_57_0
                                            ∃ (w_59_0 : mxxMat0),
                                              let w_60_0 : mxxMat0 := MxxRuntime.matrixMul w_54_0 w_33_0
                                              let w_61_0 : mxxMat0 := MxxRuntime.matrixMul w_56_0 w_37_0
                                              let w_62_0 : mxxMat0 := MxxRuntime.matrixAdd w_60_0 w_61_0
                                              ∃ (w_63_0 : mxxMat0),
                                                ∃ (w_64_0 : mxxMat1),
                                                  let w_65_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_26_0 w_64_0
                                                  ∃ (w_66_0 : mxxMat1),
                                                    parallel_parallel_generatedRoot_658_33.constraints_0 backend w_8_0 w_12_0 w_18_0 w_20_0 w_25_0 w_31_0 w_35_0 w_7_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_19_0 w_21_0 w_22_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_36_0 ∧
    parallel_parallel_generatedRoot_658_33.constraints_1 backend w_12_0 w_18_0 w_20_0 w_31_0 w_35_0 w_7_0 w_10_0 w_14_0 w_19_0 w_36_0 w_37_0 w_40_0 w_42_0 w_43_0 w_44_0 w_46_0 w_48_0 w_49_0 w_52_0 w_53_0 w_54_0 w_56_0 w_58_0 w_59_0 ∧
    parallel_parallel_generatedRoot_658_33.constraints_2 w_25_0 w_7_0 w_14_0 w_26_0 w_30_0 w_43_0 w_49_0 w_53_0 w_56_0 w_59_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 outputs

abbrev parallel_generatedRoot_658.InputType : Type :=
  (Fin 5 → Int) × mxxMat0 × (Fin 4 → mxxMat0) × (Fin 5 → Int) × mxxMat0 × (Fin 4 → mxxMat0) × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × (Fin 5 → Int) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_658.Witness where
  w_4_0 : Int
  w_11_0 : Int
  w_21_0 : Int
  w_31_0 : Fin 4 → mxxMat0
  w_31_1 : Fin 4 → mxxMat0
  w_31_2 : Fin 4 → mxxMat0
  w_31_3 : Fin 4 → mxxMat0
  w_31_4 : Fin 4 → mxxMat0
  w_31_5 : Fin 4 → mxxMat0
  w_31_6 : Fin 4 → mxxMat1
  w_32_0 : Fin 2 → mxxMat0
  w_32_1 : Fin 2 → mxxMat0
  w_32_2 : Fin 2 → mxxMat0
  w_32_3 : Fin 2 → mxxMat0
  w_32_4 : Fin 2 → mxxMat0
  w_32_5 : Fin 2 → mxxMat0
  w_32_6 : Fin 2 → mxxMat1
  w_33_0 : Fin 1 → mxxMat0
  w_33_1 : Fin 1 → mxxMat0
  w_33_2 : Fin 1 → mxxMat0
  w_33_3 : Fin 1 → mxxMat0
  w_33_4 : Fin 1 → mxxMat0
  w_33_5 : Fin 1 → mxxMat0
  w_33_6 : Fin 1 → mxxMat1
  w_34_0 : mxxMat0
  w_36_0 : mxxMat0
  w_39_0 : mxxMat6
  w_40_0 : mxxMat6
  w_43_0 : mxxMat6
  w_45_0 : mxxMat0
  w_48_0 : mxxMat6
  w_51_0 : mxxMat1
  w_56_0 : mxxMat6
  w_59_0 : mxxMat6
  w_65_0 : mxxMat0
  w_68_0 : mxxMat0
  w_75_0 : mxxMat0
  w_78_0 : mxxMat0

abbrev parallel_generatedRoot_658.constraints_1 (w_33_5 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_64_0 : mxxMat0) (w_71_0 : mxxMat0) (w_74_0 : mxxMat0) (w_78_0 : mxxMat0) (w_81_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.familyGetStatic w_33_5 (0) w_78_0 ∧
  outputs = (w_54_0, (w_64_0, (w_71_0, (w_74_0, (w_81_0, (w_84_0, ()))))))

abbrev parallel_generatedRoot_658.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 5 → Int) (w_16_0 : mxxMat0) (w_17_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 5 → Int) (w_22_0 : mxxMat0) (w_23_0 : Fin 4 → mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : ByteArray) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat1) (w_30_0 : Fin 5 → Int) (w_35_0 : mxxMat0) (w_42_0 : mxxMat0) (w_47_0 : mxxMat0) (w_55_0 : mxxMat0) (w_58_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : Int) (w_4_0 : Int) (w_7_0 : Int) (w_10_0 : Int) (w_11_0 : Int) (w_14_0 : Int) (w_15_0 : Int) (w_18_0 : Int) (w_19_0 : Int) (w_21_0 : Int) (w_31_0 : Fin 4 → mxxMat0) (w_31_1 : Fin 4 → mxxMat0) (w_31_2 : Fin 4 → mxxMat0) (w_31_3 : Fin 4 → mxxMat0) (w_31_4 : Fin 4 → mxxMat0) (w_31_5 : Fin 4 → mxxMat0) (w_31_6 : Fin 4 → mxxMat1) (w_32_0 : Fin 2 → mxxMat0) (w_32_1 : Fin 2 → mxxMat0) (w_32_2 : Fin 2 → mxxMat0) (w_32_3 : Fin 2 → mxxMat0) (w_32_4 : Fin 2 → mxxMat0) (w_32_5 : Fin 2 → mxxMat0) (w_32_6 : Fin 2 → mxxMat1) (w_33_0 : Fin 1 → mxxMat0) (w_33_1 : Fin 1 → mxxMat0) (w_33_2 : Fin 1 → mxxMat0) (w_33_3 : Fin 1 → mxxMat0) (w_33_4 : Fin 1 → mxxMat0) (w_33_5 : Fin 1 → mxxMat0) (w_33_6 : Fin 1 → mxxMat1) (w_34_0 : mxxMat0) (w_36_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat6) (w_40_0 : mxxMat6) (w_43_0 : mxxMat6) (w_45_0 : mxxMat0) (w_48_0 : mxxMat6) (w_51_0 : mxxMat1) (w_56_0 : mxxMat6) (w_59_0 : mxxMat6) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_75_0 : mxxMat0) : Prop :=
  w_2_0 ≠ 0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_3_0 w_4_0 ∧
  0 ≤ w_7_0 ∧
  w_7_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_7_0 [w_4_0, w_10_0] w_11_0 ∧
  w_14_0 ≠ 0 ∧
  w_18_0 ≠ 0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_3_0 w_21_0 ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_parallel_generatedRoot_658_31 backend hashModel tape (path ++ [31, i.val]) params i_0 i (w_4_0, (w_16_0, (w_17_0, (w_19_0, (w_21_0, (w_22_0, (w_23_0, (w_24_0, (w_25_0, (w_26_0, (w_27_0, (w_28_0, (w_29_0, (w_30_0, ())))))))))))))) ((w_31_0 i), ((w_31_1 i), ((w_31_2 i), ((w_31_3 i), ((w_31_4 i), ((w_31_5 i), ((w_31_6 i), ())))))))) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_658_32 backend hashModel tape (path ++ [32, i.val]) params i_0 i (w_11_0, (w_31_0, (w_31_2, (w_31_1, (w_31_4, (w_31_6, (w_31_3, (w_31_5, ())))))))) ((w_32_0 i), ((w_32_1 i), ((w_32_2 i), ((w_32_3 i), ((w_32_4 i), ((w_32_5 i), ((w_32_6 i), ())))))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_parallel_generatedRoot_658_33 backend hashModel tape (path ++ [33, i.val]) params i_0 i (w_15_0, (w_32_0, (w_32_2, (w_32_1, (w_32_4, (w_32_6, (w_32_3, (w_32_5, ())))))))) ((w_33_0 i), ((w_33_1 i), ((w_33_2 i), ((w_33_3 i), ((w_33_4 i), ((w_33_5 i), ((w_33_6 i), ())))))))) ∧
  MxxRuntime.familyGetStatic w_33_0 (0) w_34_0 ∧
  MxxRuntime.centeredRebaseRuns w_35_0 w_36_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_38_0 w_39_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_39_0 w_40_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_42_0 w_43_0 ∧
  MxxRuntime.familyGetStatic w_33_1 (0) w_45_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_47_0 w_48_0 ∧
  MxxRuntime.familyGetStatic w_33_6 (0) w_51_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.familyGetStatic w_33_2 (0) w_65_0 ∧
  MxxRuntime.familyGetStatic w_33_3 (0) w_68_0 ∧
  MxxRuntime.familyGetStatic w_33_4 (0) w_75_0

abbrev parallel_generatedRoot_658.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_658.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_658.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 5 → Int := inputs.1
  let w_16_0 : mxxMat0 := inputs.2.1
  let w_17_0 : Fin 4 → mxxMat0 := inputs.2.2.1
  let w_20_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_23_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_24_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_25_0 : ByteArray := inputs.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_28_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_30_0 : Fin 5 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_35_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_42_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_47_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_52_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_58_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_62_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 : Int := 5
  let w_3_0 : Int := (w_1_0 % w_2_0)
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := 2
  let w_6_0 : Bool := decide (w_4_0 < w_5_0)
  let w_7_0 : Int := if w_6_0 then 1 else 0
  let w_8_0 : Int := 2
  let w_9_0 : Int := 0
  let w_10_0 : Int := (w_8_0 + w_9_0)
  let w_11_0 := witness.w_11_0
  let w_12_0 : Int := 1
  let w_13_0 : Int := (w_11_0 + w_12_0)
  let w_14_0 : Int := 2
  let w_15_0 : Int := (w_13_0 / w_14_0)
  let w_18_0 : Int := 5
  let w_19_0 : Int := (w_1_0 / w_18_0)
  let w_21_0 := witness.w_21_0
  let w_31_0 := witness.w_31_0
  let w_31_1 := witness.w_31_1
  let w_31_2 := witness.w_31_2
  let w_31_3 := witness.w_31_3
  let w_31_4 := witness.w_31_4
  let w_31_5 := witness.w_31_5
  let w_31_6 := witness.w_31_6
  let w_32_0 := witness.w_32_0
  let w_32_1 := witness.w_32_1
  let w_32_2 := witness.w_32_2
  let w_32_3 := witness.w_32_3
  let w_32_4 := witness.w_32_4
  let w_32_5 := witness.w_32_5
  let w_32_6 := witness.w_32_6
  let w_33_0 := witness.w_33_0
  let w_33_1 := witness.w_33_1
  let w_33_2 := witness.w_33_2
  let w_33_3 := witness.w_33_3
  let w_33_4 := witness.w_33_4
  let w_33_5 := witness.w_33_5
  let w_33_6 := witness.w_33_6
  let w_34_0 := witness.w_34_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_38_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_36_0 w_37_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_34_0 w_40_0
  let w_43_0 := witness.w_43_0
  let w_44_0 : mxxMat0 := MxxRuntime.matrixMul w_41_0 w_43_0
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat0 := MxxRuntime.matrixMul w_45_0 w_40_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixMul w_46_0 w_48_0
  let w_50_0 : mxxMat0 := MxxRuntime.matrixAdd w_44_0 w_49_0
  let w_51_0 := witness.w_51_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_51_0 w_52_0
  let w_54_0 : mxxMat0 := MxxRuntime.matrixAdd w_50_0 w_53_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMul w_41_0 w_56_0
  let w_59_0 := witness.w_59_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixMul w_46_0 w_59_0
  let w_61_0 : mxxMat0 := MxxRuntime.matrixAdd w_57_0 w_60_0
  let w_63_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_51_0 w_62_0
  let w_64_0 : mxxMat0 := MxxRuntime.matrixAdd w_61_0 w_63_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : mxxMat0 := MxxRuntime.matrixMul w_65_0 w_40_0
  let w_67_0 : mxxMat0 := MxxRuntime.matrixMul w_66_0 w_43_0
  let w_68_0 := witness.w_68_0
  let w_69_0 : mxxMat0 := MxxRuntime.matrixMul w_68_0 w_40_0
  let w_70_0 : mxxMat0 := MxxRuntime.matrixMul w_69_0 w_48_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixAdd w_67_0 w_70_0
  let w_72_0 : mxxMat0 := MxxRuntime.matrixMul w_66_0 w_56_0
  let w_73_0 : mxxMat0 := MxxRuntime.matrixMul w_69_0 w_59_0
  let w_74_0 : mxxMat0 := MxxRuntime.matrixAdd w_72_0 w_73_0
  let w_75_0 := witness.w_75_0
  let w_76_0 : mxxMat0 := MxxRuntime.matrixMul w_75_0 w_40_0
  let w_77_0 : mxxMat0 := MxxRuntime.matrixMul w_76_0 w_43_0
  let w_78_0 := witness.w_78_0
  let w_79_0 : mxxMat0 := MxxRuntime.matrixMul w_78_0 w_40_0
  let w_80_0 : mxxMat0 := MxxRuntime.matrixMul w_79_0 w_48_0
  let w_81_0 : mxxMat0 := MxxRuntime.matrixAdd w_77_0 w_80_0
  let w_82_0 : mxxMat0 := MxxRuntime.matrixMul w_76_0 w_56_0
  let w_83_0 : mxxMat0 := MxxRuntime.matrixMul w_79_0 w_59_0
  let w_84_0 : mxxMat0 := MxxRuntime.matrixAdd w_82_0 w_83_0
  parallel_generatedRoot_658.constraints_0 backend hashModel tape path params i_0 w_0_0 w_16_0 w_17_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_35_0 w_42_0 w_47_0 w_55_0 w_58_0 w_2_0 w_3_0 w_4_0 w_7_0 w_10_0 w_11_0 w_14_0 w_15_0 w_18_0 w_19_0 w_21_0 w_31_0 w_31_1 w_31_2 w_31_3 w_31_4 w_31_5 w_31_6 w_32_0 w_32_1 w_32_2 w_32_3 w_32_4 w_32_5 w_32_6 w_33_0 w_33_1 w_33_2 w_33_3 w_33_4 w_33_5 w_33_6 w_34_0 w_36_0 w_38_0 w_39_0 w_40_0 w_43_0 w_45_0 w_48_0 w_51_0 w_56_0 w_59_0 w_65_0 w_68_0 w_75_0 ∧
    parallel_generatedRoot_658.constraints_1 w_33_5 w_54_0 w_64_0 w_71_0 w_74_0 w_78_0 w_81_0 w_84_0 outputs

def parallel_generatedRoot_658 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_658.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_658.Witness,
    parallel_generatedRoot_658.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_870.InputType : Type :=
  (Fin 26 → mxxMat0) × (Fin 5 → mxxMat0) × (Fin 5 → mxxMat0) × (Fin 26 → mxxMat0) × (Fin 5 → mxxMat0) × (Fin 5 → mxxMat0) × (Fin 26 → mxxMat0) × (Fin 5 → mxxMat0) × (Fin 5 → mxxMat0) × (Fin 26 → mxxMat3) × ByteArray × Unit

abbrev parallel_generatedRoot_870.constraints_4 (w_12_0 : Int) (w_25_0 : Int) (w_37_0 : mxxMat0) (w_45_0 : mxxMat0) (w_53_0 : mxxMat0) (w_55_0 : mxxMat3) (w_68_0 : mxxMat3) (w_75_0 : mxxMat3) (w_76_0 : mxxMat3) (w_77_0 : mxxMat3) (w_78_0 : mxxMat3) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat3 × Unit) : Prop :=
  MxxRuntime.select w_25_0 [w_68_0, w_75_0] w_76_0 ∧
  MxxRuntime.modulusReduceRuns w_76_0 w_77_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_12_0 [w_55_0, w_77_0] w_78_0 ∧
  outputs = (w_37_0, (w_45_0, (w_53_0, (w_78_0, ()))))

abbrev parallel_generatedRoot_870.constraints_3 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_61_0 : ByteArray) (w_25_0 : Int) (w_60_0 : Int) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat3) : Prop :=
  (5) ≠ 0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (((-1 + (4 * (Int.fdiv ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5)))) + (Int.fmod ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_61_0) w_69_0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_61_0) w_70_0 ∧
  0 ≤ w_60_0 ∧
  w_60_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_60_0 [w_69_0, w_70_0] w_71_0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (((-1 + (4 * (Int.fdiv ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5)))) + (Int.fmod ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_61_0) w_72_0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_61_0) w_73_0 ∧
  0 ≤ w_60_0 ∧
  w_60_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_60_0 [w_72_0, w_73_0] w_74_0 ∧
  MxxRuntime.concatRows w_71_0 w_74_0 w_75_0 ∧
  0 ≤ w_25_0 ∧
  w_25_0 < 2 ∧
  2 = 2

abbrev parallel_generatedRoot_870.constraints_2 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_61_0 : ByteArray) (w_60_0 : Int) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_68_0 : mxxMat3) : Prop :=
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (((-1 + (4 * (Int.fdiv ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5)))) + (Int.fmod ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_61_0) w_62_0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_61_0) w_63_0 ∧
  0 ≤ w_60_0 ∧
  w_60_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_60_0 [w_62_0, w_63_0] w_64_0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (((-1 + (4 * (Int.fdiv ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5)))) + (Int.fmod ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_61_0) w_65_0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0 ∧
  (5) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.fmod ((-3 + (Int.fdiv ((Int.ofNat i_0)) (2)))) (5))) (5))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_61_0) w_66_0 ∧
  0 ≤ w_60_0 ∧
  w_60_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_60_0 [w_65_0, w_66_0] w_67_0 ∧
  MxxRuntime.concatRows w_64_0 w_67_0 w_68_0 ∧
  (2) ≠ 0 ∧
  (5) ≠ 0

abbrev parallel_generatedRoot_870.constraints_1 (w_42_0 : Fin 5 → mxxMat0) (w_46_0 : Fin 26 → mxxMat0) (w_48_0 : Fin 5 → mxxMat0) (w_50_0 : Fin 5 → mxxMat0) (w_54_0 : Fin 26 → mxxMat3) (w_12_0 : Int) (w_25_0 : Int) (w_26_0 : Int) (w_32_0 : Int) (w_39_0 : mxxMat0) (w_41_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_47_0 : mxxMat0) (w_49_0 : mxxMat0) (w_51_0 : mxxMat0) (w_52_0 : mxxMat0) (w_53_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : Int) : Prop :=
  w_32_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_42_0 w_32_0 w_43_0 ∧
  0 ≤ w_25_0 ∧
  w_25_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_25_0 [w_41_0, w_43_0] w_44_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_12_0 [w_39_0, w_44_0] w_45_0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 26 ∧
  MxxRuntime.familyGetDynamic w_46_0 w_26_0 w_47_0 ∧
  0 ≤ w_32_0 ∧
  w_32_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_48_0 w_32_0 w_49_0 ∧
  0 ≤ w_32_0 ∧
  w_32_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_50_0 w_32_0 w_51_0 ∧
  0 ≤ w_25_0 ∧
  w_25_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_25_0 [w_49_0, w_51_0] w_52_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_12_0 [w_47_0, w_52_0] w_53_0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 26 ∧
  MxxRuntime.familyGetDynamic w_54_0 w_26_0 w_55_0 ∧
  w_56_0 ≠ 0 ∧
  (2) ≠ 0

abbrev parallel_generatedRoot_870.constraints_0 (w_13_0 : Fin 26 → mxxMat0) (w_28_0 : Fin 5 → mxxMat0) (w_34_0 : Fin 5 → mxxMat0) (w_38_0 : Fin 26 → mxxMat0) (w_40_0 : Fin 5 → mxxMat0) (w_1_0 : Int) (w_2_0 : Int) (w_12_0 : Int) (w_16_0 : Int) (w_18_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_24_0 : Int) (w_25_0 : Int) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_31_0 : Int) (w_32_0 : Int) (w_33_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_39_0 : mxxMat0) (w_41_0 : mxxMat0) : Prop :=
  w_1_0 ≠ 0 ∧
  0 ≤ w_16_0 ∧
  w_16_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_16_0 [w_18_0, w_2_0] w_19_0 ∧
  w_20_0 ≠ 0 ∧
  w_24_0 ≠ 0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 26 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_26_0 w_27_0 ∧
  w_31_0 ≠ 0 ∧
  0 ≤ w_32_0 ∧
  w_32_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_28_0 w_32_0 w_33_0 ∧
  0 ≤ w_32_0 ∧
  w_32_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_34_0 w_32_0 w_35_0 ∧
  0 ≤ w_25_0 ∧
  w_25_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_25_0 [w_33_0, w_35_0] w_36_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_12_0 [w_27_0, w_36_0] w_37_0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 26 ∧
  MxxRuntime.familyGetDynamic w_38_0 w_26_0 w_39_0 ∧
  0 ≤ w_32_0 ∧
  w_32_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_40_0 w_32_0 w_41_0 ∧
  0 ≤ w_32_0

def parallel_generatedRoot_870 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_870.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat3 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_13_0 : Fin 26 → mxxMat0 := inputs.1
  let w_28_0 : Fin 5 → mxxMat0 := inputs.2.1
  let w_34_0 : Fin 5 → mxxMat0 := inputs.2.2.1
  let w_38_0 : Fin 26 → mxxMat0 := inputs.2.2.2.1
  let w_40_0 : Fin 5 → mxxMat0 := inputs.2.2.2.2.1
  let w_42_0 : Fin 5 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_46_0 : Fin 26 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_48_0 : Fin 5 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_50_0 : Fin 5 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_54_0 : Fin 26 → mxxMat3 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_61_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : Int := (Int.ofNat i_0)
  let w_1_0 : Int := 2
  let w_2_0 : Int := (w_0_0 / w_1_0)
  let w_3_0 : Int := 8
  let w_4_0 : Bool := decide (w_2_0 < w_3_0)
  let w_5_0 : Int := if w_4_0 then 1 else 0
  let w_6_0 : Int := 2
  let w_7_0 : Bool := decide (w_6_0 < w_2_0)
  let w_8_0 : Int := if w_7_0 then 1 else 0
  let w_9_0 : Int := (w_5_0 + w_8_0)
  let w_10_0 : Int := 2
  let w_11_0 : Bool := decide (w_9_0 = w_10_0)
  let w_12_0 : Int := if w_11_0 then 1 else 0
  let w_14_0 : Int := 3
  let w_15_0 : Bool := decide (w_2_0 < w_14_0)
  let w_16_0 : Int := if w_15_0 then 1 else 0
  let w_17_0 : Int := 5
  let w_18_0 : Int := (w_2_0 - w_17_0)
  ∃ (w_19_0 : Int),
    let w_20_0 : Int := 13
    let w_21_0 : Int := (w_19_0 % w_20_0)
    let w_22_0 : Int := 2
    let w_23_0 : Int := (w_21_0 * w_22_0)
    let w_24_0 : Int := 2
    let w_25_0 : Int := (w_0_0 % w_24_0)
    let w_26_0 : Int := (w_23_0 + w_25_0)
    ∃ (w_27_0 : mxxMat0),
      let w_29_0 : Int := 3
      let w_30_0 : Int := (w_2_0 - w_29_0)
      let w_31_0 : Int := 5
      let w_32_0 : Int := (w_30_0 % w_31_0)
      ∃ (w_33_0 : mxxMat0),
        ∃ (w_35_0 : mxxMat0),
          ∃ (w_36_0 : mxxMat0),
            ∃ (w_37_0 : mxxMat0),
              ∃ (w_39_0 : mxxMat0),
                ∃ (w_41_0 : mxxMat0),
                  ∃ (w_43_0 : mxxMat0),
                    ∃ (w_44_0 : mxxMat0),
                      ∃ (w_45_0 : mxxMat0),
                        ∃ (w_47_0 : mxxMat0),
                          ∃ (w_49_0 : mxxMat0),
                            ∃ (w_51_0 : mxxMat0),
                              ∃ (w_52_0 : mxxMat0),
                                ∃ (w_53_0 : mxxMat0),
                                  ∃ (w_55_0 : mxxMat3),
                                    let w_56_0 : Int := 5
                                    let w_57_0 : Int := (w_32_0 % w_56_0)
                                    let w_58_0 : Int := 0
                                    let w_59_0 : Bool := decide (w_57_0 = w_58_0)
                                    let w_60_0 : Int := if w_59_0 then 1 else 0
                                    ∃ (w_62_0 : mxxMat0),
                                      ∃ (w_63_0 : mxxMat0),
                                        ∃ (w_64_0 : mxxMat0),
                                          ∃ (w_65_0 : mxxMat0),
                                            ∃ (w_66_0 : mxxMat0),
                                              ∃ (w_67_0 : mxxMat0),
                                                ∃ (w_68_0 : mxxMat3),
                                                  ∃ (w_69_0 : mxxMat0),
                                                    ∃ (w_70_0 : mxxMat0),
                                                      ∃ (w_71_0 : mxxMat0),
                                                        ∃ (w_72_0 : mxxMat0),
                                                          ∃ (w_73_0 : mxxMat0),
                                                            ∃ (w_74_0 : mxxMat0),
                                                              ∃ (w_75_0 : mxxMat3),
                                                                ∃ (w_76_0 : mxxMat3),
                                                                  ∃ (w_77_0 : mxxMat3),
                                                                    ∃ (w_78_0 : mxxMat3),
                                                                      parallel_generatedRoot_870.constraints_0 w_13_0 w_28_0 w_34_0 w_38_0 w_40_0 w_1_0 w_2_0 w_12_0 w_16_0 w_18_0 w_19_0 w_20_0 w_24_0 w_25_0 w_26_0 w_27_0 w_31_0 w_32_0 w_33_0 w_35_0 w_36_0 w_37_0 w_39_0 w_41_0 ∧
    parallel_generatedRoot_870.constraints_1 w_42_0 w_46_0 w_48_0 w_50_0 w_54_0 w_12_0 w_25_0 w_26_0 w_32_0 w_39_0 w_41_0 w_43_0 w_44_0 w_45_0 w_47_0 w_49_0 w_51_0 w_52_0 w_53_0 w_55_0 w_56_0 ∧
    parallel_generatedRoot_870.constraints_2 hashModel i_0 w_61_0 w_60_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 ∧
    parallel_generatedRoot_870.constraints_3 hashModel i_0 w_61_0 w_25_0 w_60_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 ∧
    parallel_generatedRoot_870.constraints_4 w_12_0 w_25_0 w_37_0 w_45_0 w_53_0 w_55_0 w_68_0 w_75_0 w_76_0 w_77_0 w_78_0 outputs

abbrev parallel_generatedRoot_889.InputType : Type :=
  Int × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit

abbrev parallel_generatedRoot_889.constraints_3 (w_45_0 : mxxMat2) (w_46_0 : mxxMat0) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_889.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_39_0 : ByteArray) (w_41_0 : mxxMat0) (w_48_0 : mxxMat0) (w_24_0 : mxxMat2) (w_30_0 : mxxMat2) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat2) (w_37_concat_1 : mxxMat3) (w_38_0 : mxxMat4) (w_40_0 : mxxMat0) (w_42_0 : mxxMat3) (w_43_0 : mxxMat5) (w_44_0 : mxxMat5) (w_45_0 : mxxMat2) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_30_0 2 3 0 25 w_36_0 ∧
  MxxRuntime.concatRows w_34_0 w_35_0 w_37_concat_1 ∧
  MxxRuntime.concatRows w_37_concat_1 w_36_0 w_37_0 ∧
  MxxRuntime.concatColumns w_24_0 w_37_0 w_38_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 53, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_39_0) w_40_0 ∧
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 53, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_39_0) w_47_0 ∧
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

abbrev parallel_generatedRoot_889.constraints_1 (backend : MxxRuntime.BackendContext) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_16_0 : mxxMat2) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat2) (w_24_concat_1 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat5) (w_29_0 : mxxMat5) (w_30_0 : mxxMat2) (w_31_0 : mxxMat0) (w_35_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_889.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_6_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat2) (w_5_concat_1 : mxxMat3) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat2) (w_9_concat_1 : mxxMat3) (w_10_0 : mxxMat4) (w_13_0 : mxxMat3) (w_14_0 : mxxMat5) (w_15_0 : mxxMat5) (w_16_0 : mxxMat2) (w_17_0 : mxxMat0) : Prop :=
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

def parallel_generatedRoot_889 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_889.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_12_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_18_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_5_0 : mxxMat2),
        ∃ (w_5_concat_1 : mxxMat3),
          ∃ (w_7_0 : mxxMat0),
            ∃ (w_8_0 : mxxMat0),
              ∃ (w_9_0 : mxxMat2),
                ∃ (w_9_concat_1 : mxxMat3),
                  ∃ (w_10_0 : mxxMat4),
                    ∃ (w_13_0 : mxxMat3),
                      ∃ (w_14_0 : mxxMat5),
                        ∃ (w_15_0 : mxxMat5),
                          let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_15_0
                          ∃ (w_17_0 : mxxMat0),
                            let w_20_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_18_0 w_19_0
                            let w_21_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_20_0
                            ∃ (w_22_0 : mxxMat0),
                              ∃ (w_23_0 : mxxMat0),
                                ∃ (w_24_0 : mxxMat2),
                                  ∃ (w_24_concat_1 : mxxMat3),
                                    ∃ (w_27_0 : mxxMat3),
                                      ∃ (w_28_0 : mxxMat5),
                                        ∃ (w_29_0 : mxxMat5),
                                          let w_30_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_29_0
                                          ∃ (w_31_0 : mxxMat0),
                                            let w_33_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_32_0 w_19_0
                                            let w_34_0 : mxxMat0 := MxxRuntime.matrixAdd w_31_0 w_33_0
                                            ∃ (w_35_0 : mxxMat0),
                                              ∃ (w_36_0 : mxxMat0),
                                                ∃ (w_37_0 : mxxMat2),
                                                  ∃ (w_37_concat_1 : mxxMat3),
                                                    ∃ (w_38_0 : mxxMat4),
                                                      ∃ (w_40_0 : mxxMat0),
                                                        ∃ (w_42_0 : mxxMat3),
                                                          ∃ (w_43_0 : mxxMat5),
                                                            ∃ (w_44_0 : mxxMat5),
                                                              let w_45_0 : mxxMat2 := MxxRuntime.matrixMul w_38_0 w_44_0
                                                              ∃ (w_46_0 : mxxMat0),
                                                                ∃ (w_47_0 : mxxMat0),
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_38_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat0),
                                                                              ∃ (w_56_0 : mxxMat0),
                                                                                ∃ (w_57_0 : mxxMat0),
                                                                                  parallel_generatedRoot_889.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_12_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 ∧
    parallel_generatedRoot_889.constraints_1 backend w_25_0 w_26_0 w_16_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_concat_1 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 ∧
    parallel_generatedRoot_889.constraints_2 backend hashModel i_0 w_39_0 w_41_0 w_48_0 w_24_0 w_30_0 w_34_0 w_35_0 w_36_0 w_37_0 w_37_concat_1 w_38_0 w_40_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 ∧
    parallel_generatedRoot_889.constraints_3 w_45_0 w_46_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 outputs

abbrev parallel_parallel_generatedRoot_890_6.constraints_0 (w_0_0 : Fin 14400 → mxxMat0) (w_3_0 : Fin 14400 → mxxMat0) (w_5_0 : Fin 14400 → mxxMat0) (w_7_0 : Fin 14400 → mxxMat0) (w_9_0 : Fin 14400 → mxxMat0) (w_11_0 : Fin 14400 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_1_0 w_12_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, (w_10_0, (w_12_0, ()))))))

def parallel_parallel_generatedRoot_890_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat0 := inputs.1
  let w_3_0 : Fin 14400 → mxxMat0 := inputs.2.1
  let w_5_0 : Fin 14400 → mxxMat0 := inputs.2.2.1
  let w_7_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.1
  let w_9_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.2.1
  let w_11_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_10_0 : mxxMat0),
            ∃ (w_12_0 : mxxMat0),
              parallel_parallel_generatedRoot_890_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_890_25.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_890_25.constraints_0 (w_1_0 : Fin 2 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_10_0 : Fin 2 → mxxMat0) (w_14_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_22_0 : Fin 2 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_890_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_890_25.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              sequential_parallel_generatedRoot_890_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_890.Witness where
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

abbrev parallel_generatedRoot_890.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 14400 → mxxMat0) (w_1_0 : Fin 14400 → mxxMat0) (w_2_0 : Fin 14400 → mxxMat0) (w_3_0 : Fin 14400 → mxxMat0) (w_4_0 : Fin 14400 → mxxMat0) (w_5_0 : Fin 14400 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat0) (w_6_3 : Fin 2 → mxxMat0) (w_6_4 : Fin 2 → mxxMat0) (w_6_5 : Fin 2 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_25_1 : mxxMat0) (w_25_2 : mxxMat0) (w_25_3 : mxxMat0) (w_25_4 : mxxMat0) (w_25_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_890_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_890_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_890.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_890.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat0 := inputs.1
  let w_1_0 : Fin 14400 → mxxMat0 := inputs.2.1
  let w_2_0 : Fin 14400 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.1
  let w_4_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.2.1
  let w_5_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.2.2.1
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
  parallel_generatedRoot_890.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_890 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_890.Witness,
    parallel_generatedRoot_890.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_893.InputType : Type :=
  Int × mxxMat0 × ByteArray × mxxMat0 × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × mxxMat1 × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × ByteArray × mxxMat0 × mxxMat0 × Unit

abbrev parallel_generatedRoot_893.constraints_4 (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_893.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_46_0 : ByteArray) (w_55_0 : mxxMat0) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_56_0 : mxxMat3) (w_57_0 : mxxMat5) (w_58_0 : mxxMat5) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 1 2 0 25 w_53_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 54, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_46_0) w_54_0 ∧
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

abbrev parallel_generatedRoot_893.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_38_0 : Fin 7200 → mxxMat0) (w_46_0 : ByteArray) (w_48_0 : mxxMat0) (w_12_0 : Int) (w_28_0 : mxxMat2) (w_36_0 : mxxMat2) (w_37_0 : mxxMat0) (w_39_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat2) (w_44_concat_1 : mxxMat3) (w_45_0 : mxxMat4) (w_47_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_36_0 0 1 0 25 w_37_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 7200 ∧
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 54, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_46_0) w_47_0 ∧
  MxxRuntime.concatRows w_47_0 w_48_0 w_49_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3

abbrev parallel_generatedRoot_893.constraints_1 (backend : MxxRuntime.BackendContext) (w_21_0 : Fin 7200 → mxxMat0) (w_29_0 : Fin 7200 → mxxMat0) (w_31_0 : Fin 7200 → mxxMat0) (w_12_0 : Int) (w_19_0 : mxxMat2) (w_20_0 : mxxMat0) (w_22_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_30_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat3) (w_34_0 : mxxMat5) (w_35_0 : mxxMat5) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 25 w_20_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 7200 ∧
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
  w_12_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_29_0 w_12_0 w_30_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_31_0 w_12_0 w_32_0 ∧
  MxxRuntime.concatRows w_30_0 w_32_0 w_33_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_33_0 w_34_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_34_0 w_35_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev parallel_generatedRoot_893.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_6_0 : mxxMat0) (w_11_0 : Fin 7200 → mxxMat0) (w_14_0 : Fin 7200 → mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat2) (w_5_concat_1 : mxxMat3) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat2) (w_9_concat_1 : mxxMat3) (w_10_0 : mxxMat4) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat5) (w_18_0 : mxxMat5) : Prop :=
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
  w_12_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_12_0 w_15_0 ∧
  MxxRuntime.concatRows w_13_0 w_15_0 w_16_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_16_0 w_17_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_17_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

def parallel_generatedRoot_893 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_893.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.2.1
  let w_11_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.1
  let w_14_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_21_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_5_0 : mxxMat2),
        ∃ (w_5_concat_1 : mxxMat3),
          ∃ (w_7_0 : mxxMat0),
            ∃ (w_8_0 : mxxMat0),
              ∃ (w_9_0 : mxxMat2),
                ∃ (w_9_concat_1 : mxxMat3),
                  ∃ (w_10_0 : mxxMat4),
                    let w_12_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (2))
                    ∃ (w_13_0 : mxxMat0),
                      ∃ (w_15_0 : mxxMat0),
                        ∃ (w_16_0 : mxxMat3),
                          ∃ (w_17_0 : mxxMat5),
                            ∃ (w_18_0 : mxxMat5),
                              let w_19_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_18_0
                              ∃ (w_20_0 : mxxMat0),
                                ∃ (w_22_0 : mxxMat0),
                                  let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_22_0 w_23_0
                                  let w_25_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_24_0
                                  ∃ (w_26_0 : mxxMat0),
                                    ∃ (w_27_0 : mxxMat0),
                                      ∃ (w_28_0 : mxxMat2),
                                        ∃ (w_28_concat_1 : mxxMat3),
                                          ∃ (w_30_0 : mxxMat0),
                                            ∃ (w_32_0 : mxxMat0),
                                              ∃ (w_33_0 : mxxMat3),
                                                ∃ (w_34_0 : mxxMat5),
                                                  ∃ (w_35_0 : mxxMat5),
                                                    let w_36_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_35_0
                                                    ∃ (w_37_0 : mxxMat0),
                                                      ∃ (w_39_0 : mxxMat0),
                                                        let w_40_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_39_0 w_23_0
                                                        let w_41_0 : mxxMat0 := MxxRuntime.matrixAdd w_37_0 w_40_0
                                                        ∃ (w_42_0 : mxxMat0),
                                                          ∃ (w_43_0 : mxxMat0),
                                                            ∃ (w_44_0 : mxxMat2),
                                                              ∃ (w_44_concat_1 : mxxMat3),
                                                                ∃ (w_45_0 : mxxMat4),
                                                                  ∃ (w_47_0 : mxxMat0),
                                                                    ∃ (w_49_0 : mxxMat3),
                                                                      ∃ (w_50_0 : mxxMat5),
                                                                        ∃ (w_51_0 : mxxMat5),
                                                                          let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_45_0 w_51_0
                                                                          ∃ (w_53_0 : mxxMat0),
                                                                            ∃ (w_54_0 : mxxMat0),
                                                                              ∃ (w_56_0 : mxxMat3),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  ∃ (w_58_0 : mxxMat5),
                                                                                    let w_59_0 : mxxMat2 := MxxRuntime.matrixMul w_45_0 w_58_0
                                                                                    ∃ (w_60_0 : mxxMat0),
                                                                                      ∃ (w_61_0 : mxxMat0),
                                                                                        ∃ (w_62_0 : mxxMat0),
                                                                                          ∃ (w_63_0 : mxxMat0),
                                                                                            ∃ (w_64_0 : mxxMat0),
                                                                                              parallel_generatedRoot_893.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_14_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_893.constraints_1 backend w_21_0 w_29_0 w_31_0 w_12_0 w_19_0 w_20_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_concat_1 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 ∧
    parallel_generatedRoot_893.constraints_2 backend hashModel i_0 w_38_0 w_46_0 w_48_0 w_12_0 w_28_0 w_36_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_concat_1 w_45_0 w_47_0 w_49_0 w_50_0 w_51_0 ∧
    parallel_generatedRoot_893.constraints_3 backend hashModel i_0 w_46_0 w_55_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 ∧
    parallel_generatedRoot_893.constraints_4 w_52_0 w_53_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_parallel_generatedRoot_894_6.constraints_0 (w_0_0 : Fin 14400 → mxxMat0) (w_3_0 : Fin 14400 → mxxMat0) (w_5_0 : Fin 14400 → mxxMat0) (w_7_0 : Fin 14400 → mxxMat0) (w_9_0 : Fin 14400 → mxxMat0) (w_11_0 : Fin 14400 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_1_0 w_12_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, (w_10_0, (w_12_0, ()))))))

def parallel_parallel_generatedRoot_894_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat0 := inputs.1
  let w_3_0 : Fin 14400 → mxxMat0 := inputs.2.1
  let w_5_0 : Fin 14400 → mxxMat0 := inputs.2.2.1
  let w_7_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.1
  let w_9_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.2.1
  let w_11_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_10_0 : mxxMat0),
            ∃ (w_12_0 : mxxMat0),
              parallel_parallel_generatedRoot_894_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_894_25.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_894_25.constraints_0 (w_1_0 : Fin 2 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_10_0 : Fin 2 → mxxMat0) (w_14_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_22_0 : Fin 2 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_894_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_894_25.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              sequential_parallel_generatedRoot_894_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_894.Witness where
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

abbrev parallel_generatedRoot_894.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 14400 → mxxMat0) (w_1_0 : Fin 14400 → mxxMat0) (w_2_0 : Fin 14400 → mxxMat0) (w_3_0 : Fin 14400 → mxxMat0) (w_4_0 : Fin 14400 → mxxMat0) (w_5_0 : Fin 14400 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat0) (w_6_3 : Fin 2 → mxxMat0) (w_6_4 : Fin 2 → mxxMat0) (w_6_5 : Fin 2 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_25_1 : mxxMat0) (w_25_2 : mxxMat0) (w_25_3 : mxxMat0) (w_25_4 : mxxMat0) (w_25_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_894_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_894_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_894.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_894.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat0 := inputs.1
  let w_1_0 : Fin 14400 → mxxMat0 := inputs.2.1
  let w_2_0 : Fin 14400 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.1
  let w_4_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.2.1
  let w_5_0 : Fin 14400 → mxxMat0 := inputs.2.2.2.2.2.1
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
  parallel_generatedRoot_894.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_894 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_894.Witness,
    parallel_generatedRoot_894.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_895_18.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_895_18.constraints_4 (w_75_0 : Fin 8 → mxxMat0) (w_79_0 : Fin 8 → mxxMat0) (w_81_0 : Fin 8 → mxxMat0) (w_83_0 : Fin 8 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_895_18.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_parallel_generatedRoot_895_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_parallel_generatedRoot_895_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
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

abbrev sequential_parallel_generatedRoot_895_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
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

def sequential_parallel_generatedRoot_895_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_895_18.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_parallel_generatedRoot_895_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_895_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_895_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_895_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_895_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev parallel_generatedRoot_895.InputType : Type :=
  (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_895.Witness where
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
  w_23_0 : mxxMat2
  w_23_concat_1 : mxxMat3
  w_24_0 : mxxMat0
  w_25_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat2
  w_27_concat_1 : mxxMat3
  w_28_0 : mxxMat4
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat2
  w_32_concat_1 : mxxMat3
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat2
  w_36_concat_1 : mxxMat3
  w_37_0 : mxxMat4
  w_38_0 : mxxMat3
  w_39_0 : mxxMat5
  w_40_0 : mxxMat5
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat3
  w_45_0 : mxxMat5
  w_46_0 : mxxMat5
  w_48_0 : mxxMat0
  w_50_0 : mxxMat3
  w_51_0 : mxxMat5
  w_52_0 : mxxMat5
  w_54_0 : mxxMat0
  w_55_0 : mxxMat0
  w_56_0 : mxxMat3
  w_57_0 : mxxMat5
  w_58_0 : mxxMat5
  w_60_0 : mxxMat0
  w_62_0 : mxxMat0
  w_64_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0
  w_70_0 : mxxMat0
  w_71_0 : mxxMat0

abbrev parallel_generatedRoot_895.constraints_4 (w_49_0 : mxxMat0) (w_53_0 : mxxMat2) (w_59_0 : mxxMat2) (w_61_0 : mxxMat0) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_895.constraints_3 (w_41_0 : mxxMat2) (w_47_0 : mxxMat2) (w_59_0 : mxxMat2) (w_62_0 : mxxMat0) (w_64_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_895.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat0) (w_18_9 : mxxMat0) (w_47_0 : mxxMat2) (w_48_0 : mxxMat0) (w_50_0 : mxxMat3) (w_51_0 : mxxMat5) (w_52_0 : mxxMat5) (w_53_0 : mxxMat2) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat3) (w_57_0 : mxxMat5) (w_58_0 : mxxMat5) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_895.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_6 : mxxMat0) (w_18_8 : mxxMat0) (w_32_0 : mxxMat2) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat2) (w_36_concat_1 : mxxMat3) (w_37_0 : mxxMat4) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat3) (w_45_0 : mxxMat5) (w_46_0 : mxxMat5) : Prop :=
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

abbrev parallel_generatedRoot_895.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_3_0 : Fin 8 → mxxMat0) (w_5_0 : Fin 8 → mxxMat0) (w_7_0 : Fin 8 → mxxMat0) (w_9_0 : Fin 8 → mxxMat0) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_0_0 : mxxMat0) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_18_2 : mxxMat0) (w_18_3 : mxxMat0) (w_18_4 : mxxMat0) (w_18_5 : mxxMat0) (w_18_6 : mxxMat0) (w_18_7 : mxxMat0) (w_18_8 : mxxMat0) (w_18_9 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : Int) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat2) (w_23_concat_1 : mxxMat3) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat2) (w_27_concat_1 : mxxMat3) (w_28_0 : mxxMat4) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat2) (w_32_concat_1 : mxxMat3) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_895_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
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

abbrev parallel_generatedRoot_895.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_895.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_895.Witness) : Prop :=
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
  let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat2 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat2 := MxxRuntime.matrixMul w_28_0 w_58_0
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
  parallel_generatedRoot_895.constraints_0 backend hashModel tape path params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_895.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_895.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_895.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_895.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_895 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_895.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_895.Witness,
    parallel_generatedRoot_895.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_896.constraints_0 (i_0 : Nat) (w_0_0 : Fin 7200 → mxxMat0) (w_5_0 : Fin 7200 → mxxMat0) (w_9_0 : Fin 7200 → mxxMat0) (w_13_0 : Fin 7200 → mxxMat0) (w_17_0 : Fin 7200 → mxxMat0) (w_21_0 : Fin 7200 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  7200 ≠ 0 ∧
  (Int.ofNat i_0) % 7200 = 0 ∧
  (7200) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_1_0 w_14_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_17_0 w_1_0 w_18_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_1_0 w_22_0 ∧
  (8) ≠ 0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def parallel_generatedRoot_896 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 7200 → mxxMat0 := inputs.1
  let w_5_0 : Fin 7200 → mxxMat0 := inputs.2.1
  let w_9_0 : Fin 7200 → mxxMat0 := inputs.2.2.1
  let w_13_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.1
  let w_17_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := (MxxIR.exactDiv ((Int.ofNat i_0)) (7200) + (Int.fmod ((Int.ofNat i_0)) (7200)))
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
              parallel_generatedRoot_896.constraints_0 i_0 w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 outputs

abbrev sequential_generatedRoot_915.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × Unit

abbrev sequential_generatedRoot_915.constraints_0 (w_1_0 : Fin 7200 → mxxMat0) (w_6_0 : Fin 7200 → mxxMat0) (w_10_0 : Fin 7200 → mxxMat0) (w_14_0 : Fin 7200 → mxxMat0) (w_18_0 : Fin 7200 → mxxMat0) (w_22_0 : Fin 7200 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_2_0 w_19_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_2_0 w_23_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def sequential_generatedRoot_915 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_915.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_5_0 : mxxMat0 := inputs.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
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
              sequential_generatedRoot_915.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev scope_80.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × Unit

abbrev scope_80.constraints_0 (w_1_0 : Fin 7200 → mxxMat0) (w_6_0 : Fin 7200 → mxxMat0) (w_10_0 : Fin 7200 → mxxMat0) (w_14_0 : Fin 7200 → mxxMat0) (w_18_0 : Fin 7200 → mxxMat0) (w_22_0 : Fin 7200 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_2_0 w_19_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_2_0 w_23_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def scope_80 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (scope_80.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_5_0 : mxxMat0 := inputs.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := ((10 + (8 * (Int.ofNat i_0))) + (Int.ofNat i_1))
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
              scope_80.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev sequential_generatedRoot_927.InputType : Type :=
  mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × (Fin 7200 → mxxMat0) × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_927.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_9_0 : mxxMat0
  w_10_0 : mxxMat0
  w_13_0 : mxxMat0
  w_14_0 : mxxMat0
  w_17_0 : mxxMat0
  w_18_0 : mxxMat0
  w_21_0 : mxxMat0
  w_22_0 : mxxMat0
  w_25_0 : mxxMat0
  w_26_0 : mxxMat0
  w_28_0 : mxxMat0
  w_28_1 : mxxMat0
  w_28_2 : mxxMat0
  w_28_3 : mxxMat0
  w_28_4 : mxxMat0
  w_28_5 : mxxMat0
  w_29_0 : mxxMat7
  w_33_0 : mxxMat7
  w_37_0 : mxxMat7
  w_41_0 : mxxMat7
  w_45_0 : mxxMat7
  w_49_0 : mxxMat7

abbrev sequential_generatedRoot_927.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : Fin 7200 → mxxMat0) (w_8_0 : Fin 7200 → mxxMat0) (w_12_0 : Fin 7200 → mxxMat0) (w_16_0 : Fin 7200 → mxxMat0) (w_20_0 : Fin 7200 → mxxMat0) (w_24_0 : Fin 7200 → mxxMat0) (w_1_0 : mxxMat8) (w_3_0 : Int) (w_5_0 : Int) (w_7_0 : mxxMat0) (w_11_0 : mxxMat0) (w_15_0 : mxxMat0) (w_19_0 : mxxMat0) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_28_1 : mxxMat0) (w_28_2 : mxxMat0) (w_28_3 : mxxMat0) (w_28_4 : mxxMat0) (w_28_5 : mxxMat0) (w_29_0 : mxxMat7) (w_30_0 : mxxMat7) (w_32_0 : mxxMat8) (w_33_0 : mxxMat7) (w_34_0 : mxxMat7) (w_36_0 : mxxMat8) (w_37_0 : mxxMat7) (w_38_0 : mxxMat7) (w_40_0 : mxxMat8) (w_41_0 : mxxMat7) (w_42_0 : mxxMat7) (w_44_0 : mxxMat8) (w_45_0 : mxxMat7) (w_46_0 : mxxMat7) (w_48_0 : mxxMat8) (w_49_0 : mxxMat7) (w_50_0 : mxxMat7) (outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_3_0 w_25_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_5_0 w_26_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_80 backend hashModel tape (path ++ [28, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_2_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ())))))))))))) next) (Int.toNat (6)) (w_7_0, (w_11_0, (w_15_0, (w_19_0, (w_23_0, (w_27_0, ())))))) (w_28_0, (w_28_1, (w_28_2, (w_28_3, (w_28_4, (w_28_5, ())))))) ∧
  MxxRuntime.tensorRuns w_1_0 w_28_0 w_29_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  MxxRuntime.tensorRuns w_32_0 w_28_1 w_33_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  MxxRuntime.tensorRuns w_36_0 w_28_2 w_37_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  MxxRuntime.tensorRuns w_40_0 w_28_3 w_41_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  MxxRuntime.tensorRuns w_44_0 w_28_4 w_45_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  MxxRuntime.tensorRuns w_48_0 w_28_5 w_49_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, (w_46_0, (w_50_0, ()))))))

abbrev sequential_generatedRoot_927.constraints_0 (i_0 : Nat) (w_2_0 : Fin 7200 → mxxMat0) (w_8_0 : Fin 7200 → mxxMat0) (w_12_0 : Fin 7200 → mxxMat0) (w_16_0 : Fin 7200 → mxxMat0) (w_20_0 : Fin 7200 → mxxMat0) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_3_0 w_9_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_5_0 w_10_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_3_0 w_13_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_5_0 w_14_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_3_0 w_17_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_5_0 w_18_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_3_0 w_21_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_5_0 w_22_0

abbrev sequential_generatedRoot_927.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_927.InputType)) (outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) (witness : sequential_generatedRoot_927.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat7 := inputs.1
  let w_31_0 : mxxMat7 := inputs.2.1
  let w_35_0 : mxxMat7 := inputs.2.2.1
  let w_39_0 : mxxMat7 := inputs.2.2.2.1
  let w_43_0 : mxxMat7 := inputs.2.2.2.2.1
  let w_47_0 : mxxMat7 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_8_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_12_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : Fin 7200 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat8 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)
  let w_3_0 : Int := (8 + (8 * (Int.ofNat i_0)))
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := (9 + (8 * (Int.ofNat i_0)))
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_4_0 w_6_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_10_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_18_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_21_0 w_22_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat0 := MxxRuntime.matrixAdd w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_1 := witness.w_28_1
  let w_28_2 := witness.w_28_2
  let w_28_3 := witness.w_28_3
  let w_28_4 := witness.w_28_4
  let w_28_5 := witness.w_28_5
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat7 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 : mxxMat8 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat7 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 : mxxMat8 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)
  let w_37_0 := witness.w_37_0
  let w_38_0 : mxxMat7 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 : mxxMat8 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat7 := MxxRuntime.matrixAdd w_39_0 w_41_0
  let w_44_0 : mxxMat8 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat7 := MxxRuntime.matrixAdd w_43_0 w_45_0
  let w_48_0 : mxxMat8 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat8)
  let w_49_0 := witness.w_49_0
  let w_50_0 : mxxMat7 := MxxRuntime.matrixAdd w_47_0 w_49_0
  sequential_generatedRoot_927.constraints_0 i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_20_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_17_0 w_18_0 w_21_0 w_22_0 ∧
    sequential_generatedRoot_927.constraints_1 backend hashModel tape path params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_20_0 w_24_0 w_1_0 w_3_0 w_5_0 w_7_0 w_11_0 w_15_0 w_19_0 w_23_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_1 w_28_2 w_28_3 w_28_4 w_28_5 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_36_0 w_37_0 w_38_0 w_40_0 w_41_0 w_42_0 w_44_0 w_45_0 w_46_0 w_48_0 w_49_0 w_50_0 outputs

def sequential_generatedRoot_927 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_927.InputType)) (outputs : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_927.Witness,
    sequential_generatedRoot_927.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_943.InputType : Type :=
  Int × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit

abbrev parallel_generatedRoot_943.constraints_3 (w_45_0 : mxxMat2) (w_46_0 : mxxMat0) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_943.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_39_0 : ByteArray) (w_41_0 : mxxMat0) (w_48_0 : mxxMat0) (w_24_0 : mxxMat2) (w_30_0 : mxxMat2) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat2) (w_37_concat_1 : mxxMat3) (w_38_0 : mxxMat4) (w_40_0 : mxxMat0) (w_42_0 : mxxMat3) (w_43_0 : mxxMat5) (w_44_0 : mxxMat5) (w_45_0 : mxxMat2) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_943.constraints_1 (backend : MxxRuntime.BackendContext) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_16_0 : mxxMat2) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat2) (w_24_concat_1 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat5) (w_29_0 : mxxMat5) (w_30_0 : mxxMat2) (w_31_0 : mxxMat0) (w_35_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_943.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_6_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat2) (w_5_concat_1 : mxxMat3) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat2) (w_9_concat_1 : mxxMat3) (w_10_0 : mxxMat4) (w_13_0 : mxxMat3) (w_14_0 : mxxMat5) (w_15_0 : mxxMat5) (w_16_0 : mxxMat2) (w_17_0 : mxxMat0) : Prop :=
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

def parallel_generatedRoot_943 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_943.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_12_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_18_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_5_0 : mxxMat2),
        ∃ (w_5_concat_1 : mxxMat3),
          ∃ (w_7_0 : mxxMat0),
            ∃ (w_8_0 : mxxMat0),
              ∃ (w_9_0 : mxxMat2),
                ∃ (w_9_concat_1 : mxxMat3),
                  ∃ (w_10_0 : mxxMat4),
                    ∃ (w_13_0 : mxxMat3),
                      ∃ (w_14_0 : mxxMat5),
                        ∃ (w_15_0 : mxxMat5),
                          let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_15_0
                          ∃ (w_17_0 : mxxMat0),
                            let w_20_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_18_0 w_19_0
                            let w_21_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_20_0
                            ∃ (w_22_0 : mxxMat0),
                              ∃ (w_23_0 : mxxMat0),
                                ∃ (w_24_0 : mxxMat2),
                                  ∃ (w_24_concat_1 : mxxMat3),
                                    ∃ (w_27_0 : mxxMat3),
                                      ∃ (w_28_0 : mxxMat5),
                                        ∃ (w_29_0 : mxxMat5),
                                          let w_30_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_29_0
                                          ∃ (w_31_0 : mxxMat0),
                                            let w_33_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_32_0 w_19_0
                                            let w_34_0 : mxxMat0 := MxxRuntime.matrixAdd w_31_0 w_33_0
                                            ∃ (w_35_0 : mxxMat0),
                                              ∃ (w_36_0 : mxxMat0),
                                                ∃ (w_37_0 : mxxMat2),
                                                  ∃ (w_37_concat_1 : mxxMat3),
                                                    ∃ (w_38_0 : mxxMat4),
                                                      ∃ (w_40_0 : mxxMat0),
                                                        ∃ (w_42_0 : mxxMat3),
                                                          ∃ (w_43_0 : mxxMat5),
                                                            ∃ (w_44_0 : mxxMat5),
                                                              let w_45_0 : mxxMat2 := MxxRuntime.matrixMul w_38_0 w_44_0
                                                              ∃ (w_46_0 : mxxMat0),
                                                                ∃ (w_47_0 : mxxMat0),
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_38_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat0),
                                                                              ∃ (w_56_0 : mxxMat0),
                                                                                ∃ (w_57_0 : mxxMat0),
                                                                                  parallel_generatedRoot_943.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_12_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 ∧
    parallel_generatedRoot_943.constraints_1 backend w_25_0 w_26_0 w_16_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_concat_1 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 ∧
    parallel_generatedRoot_943.constraints_2 backend hashModel i_0 w_39_0 w_41_0 w_48_0 w_24_0 w_30_0 w_34_0 w_35_0 w_36_0 w_37_0 w_37_concat_1 w_38_0 w_40_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 ∧
    parallel_generatedRoot_943.constraints_3 w_45_0 w_46_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 outputs

abbrev parallel_parallel_generatedRoot_944_6.constraints_0 (w_0_0 : Fin 72000 → mxxMat0) (w_3_0 : Fin 72000 → mxxMat0) (w_5_0 : Fin 72000 → mxxMat0) (w_7_0 : Fin 72000 → mxxMat0) (w_9_0 : Fin 72000 → mxxMat0) (w_11_0 : Fin 72000 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_1_0 w_12_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, (w_10_0, (w_12_0, ()))))))

def parallel_parallel_generatedRoot_944_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat0 := inputs.1
  let w_3_0 : Fin 72000 → mxxMat0 := inputs.2.1
  let w_5_0 : Fin 72000 → mxxMat0 := inputs.2.2.1
  let w_7_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.1
  let w_9_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.2.1
  let w_11_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_10_0 : mxxMat0),
            ∃ (w_12_0 : mxxMat0),
              parallel_parallel_generatedRoot_944_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_944_25.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_944_25.constraints_0 (w_1_0 : Fin 2 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_10_0 : Fin 2 → mxxMat0) (w_14_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_22_0 : Fin 2 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_944_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_944_25.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              sequential_parallel_generatedRoot_944_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_944.Witness where
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

abbrev parallel_generatedRoot_944.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 72000 → mxxMat0) (w_1_0 : Fin 72000 → mxxMat0) (w_2_0 : Fin 72000 → mxxMat0) (w_3_0 : Fin 72000 → mxxMat0) (w_4_0 : Fin 72000 → mxxMat0) (w_5_0 : Fin 72000 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat0) (w_6_3 : Fin 2 → mxxMat0) (w_6_4 : Fin 2 → mxxMat0) (w_6_5 : Fin 2 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_25_1 : mxxMat0) (w_25_2 : mxxMat0) (w_25_3 : mxxMat0) (w_25_4 : mxxMat0) (w_25_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_944_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_944_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_944.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_944.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat0 := inputs.1
  let w_1_0 : Fin 72000 → mxxMat0 := inputs.2.1
  let w_2_0 : Fin 72000 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.1
  let w_4_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.2.1
  let w_5_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.2.2.1
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
  parallel_generatedRoot_944.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_944 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_944.Witness,
    parallel_generatedRoot_944.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_947.InputType : Type :=
  Int × mxxMat0 × ByteArray × mxxMat0 × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × mxxMat1 × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × ByteArray × mxxMat0 × mxxMat0 × Unit

abbrev parallel_generatedRoot_947.constraints_4 (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_947.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_46_0 : ByteArray) (w_55_0 : mxxMat0) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_56_0 : mxxMat3) (w_57_0 : mxxMat5) (w_58_0 : mxxMat5) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_947.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_38_0 : Fin 36000 → mxxMat0) (w_46_0 : ByteArray) (w_48_0 : mxxMat0) (w_12_0 : Int) (w_28_0 : mxxMat2) (w_36_0 : mxxMat2) (w_37_0 : mxxMat0) (w_39_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat2) (w_44_concat_1 : mxxMat3) (w_45_0 : mxxMat4) (w_47_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_36_0 0 1 0 25 w_37_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 36000 ∧
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

abbrev parallel_generatedRoot_947.constraints_1 (backend : MxxRuntime.BackendContext) (w_21_0 : Fin 36000 → mxxMat0) (w_29_0 : Fin 36000 → mxxMat0) (w_31_0 : Fin 36000 → mxxMat0) (w_12_0 : Int) (w_19_0 : mxxMat2) (w_20_0 : mxxMat0) (w_22_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_30_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat3) (w_34_0 : mxxMat5) (w_35_0 : mxxMat5) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 25 w_20_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 36000 ∧
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
  w_12_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_29_0 w_12_0 w_30_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_31_0 w_12_0 w_32_0 ∧
  MxxRuntime.concatRows w_30_0 w_32_0 w_33_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_33_0 w_34_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_34_0 w_35_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev parallel_generatedRoot_947.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_6_0 : mxxMat0) (w_11_0 : Fin 36000 → mxxMat0) (w_14_0 : Fin 36000 → mxxMat0) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_5_0 : mxxMat2) (w_5_concat_1 : mxxMat3) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat2) (w_9_concat_1 : mxxMat3) (w_10_0 : mxxMat4) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat5) (w_18_0 : mxxMat5) : Prop :=
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
  w_12_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_12_0 w_15_0 ∧
  MxxRuntime.concatRows w_13_0 w_15_0 w_16_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_16_0 w_17_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_17_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

def parallel_generatedRoot_947 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_947.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.2.1
  let w_11_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.1
  let w_14_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_21_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_5_0 : mxxMat2),
        ∃ (w_5_concat_1 : mxxMat3),
          ∃ (w_7_0 : mxxMat0),
            ∃ (w_8_0 : mxxMat0),
              ∃ (w_9_0 : mxxMat2),
                ∃ (w_9_concat_1 : mxxMat3),
                  ∃ (w_10_0 : mxxMat4),
                    let w_12_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (2))
                    ∃ (w_13_0 : mxxMat0),
                      ∃ (w_15_0 : mxxMat0),
                        ∃ (w_16_0 : mxxMat3),
                          ∃ (w_17_0 : mxxMat5),
                            ∃ (w_18_0 : mxxMat5),
                              let w_19_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_18_0
                              ∃ (w_20_0 : mxxMat0),
                                ∃ (w_22_0 : mxxMat0),
                                  let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_22_0 w_23_0
                                  let w_25_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_24_0
                                  ∃ (w_26_0 : mxxMat0),
                                    ∃ (w_27_0 : mxxMat0),
                                      ∃ (w_28_0 : mxxMat2),
                                        ∃ (w_28_concat_1 : mxxMat3),
                                          ∃ (w_30_0 : mxxMat0),
                                            ∃ (w_32_0 : mxxMat0),
                                              ∃ (w_33_0 : mxxMat3),
                                                ∃ (w_34_0 : mxxMat5),
                                                  ∃ (w_35_0 : mxxMat5),
                                                    let w_36_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_35_0
                                                    ∃ (w_37_0 : mxxMat0),
                                                      ∃ (w_39_0 : mxxMat0),
                                                        let w_40_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_39_0 w_23_0
                                                        let w_41_0 : mxxMat0 := MxxRuntime.matrixAdd w_37_0 w_40_0
                                                        ∃ (w_42_0 : mxxMat0),
                                                          ∃ (w_43_0 : mxxMat0),
                                                            ∃ (w_44_0 : mxxMat2),
                                                              ∃ (w_44_concat_1 : mxxMat3),
                                                                ∃ (w_45_0 : mxxMat4),
                                                                  ∃ (w_47_0 : mxxMat0),
                                                                    ∃ (w_49_0 : mxxMat3),
                                                                      ∃ (w_50_0 : mxxMat5),
                                                                        ∃ (w_51_0 : mxxMat5),
                                                                          let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_45_0 w_51_0
                                                                          ∃ (w_53_0 : mxxMat0),
                                                                            ∃ (w_54_0 : mxxMat0),
                                                                              ∃ (w_56_0 : mxxMat3),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  ∃ (w_58_0 : mxxMat5),
                                                                                    let w_59_0 : mxxMat2 := MxxRuntime.matrixMul w_45_0 w_58_0
                                                                                    ∃ (w_60_0 : mxxMat0),
                                                                                      ∃ (w_61_0 : mxxMat0),
                                                                                        ∃ (w_62_0 : mxxMat0),
                                                                                          ∃ (w_63_0 : mxxMat0),
                                                                                            ∃ (w_64_0 : mxxMat0),
                                                                                              parallel_generatedRoot_947.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_14_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_947.constraints_1 backend w_21_0 w_29_0 w_31_0 w_12_0 w_19_0 w_20_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_concat_1 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 ∧
    parallel_generatedRoot_947.constraints_2 backend hashModel i_0 w_38_0 w_46_0 w_48_0 w_12_0 w_28_0 w_36_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_concat_1 w_45_0 w_47_0 w_49_0 w_50_0 w_51_0 ∧
    parallel_generatedRoot_947.constraints_3 backend hashModel i_0 w_46_0 w_55_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 ∧
    parallel_generatedRoot_947.constraints_4 w_52_0 w_53_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_parallel_generatedRoot_948_6.constraints_0 (w_0_0 : Fin 72000 → mxxMat0) (w_3_0 : Fin 72000 → mxxMat0) (w_5_0 : Fin 72000 → mxxMat0) (w_7_0 : Fin 72000 → mxxMat0) (w_9_0 : Fin 72000 → mxxMat0) (w_11_0 : Fin 72000 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_1_0 w_12_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, (w_10_0, (w_12_0, ()))))))

def parallel_parallel_generatedRoot_948_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat0 := inputs.1
  let w_3_0 : Fin 72000 → mxxMat0 := inputs.2.1
  let w_5_0 : Fin 72000 → mxxMat0 := inputs.2.2.1
  let w_7_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.1
  let w_9_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.2.1
  let w_11_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat0),
    ∃ (w_4_0 : mxxMat0),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_10_0 : mxxMat0),
            ∃ (w_12_0 : mxxMat0),
              parallel_parallel_generatedRoot_948_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_948_25.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_948_25.constraints_0 (w_1_0 : Fin 2 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_10_0 : Fin 2 → mxxMat0) (w_14_0 : Fin 2 → mxxMat0) (w_18_0 : Fin 2 → mxxMat0) (w_22_0 : Fin 2 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_948_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_948_25.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
              sequential_parallel_generatedRoot_948_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_948.Witness where
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

abbrev parallel_generatedRoot_948.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_0_0 : Fin 72000 → mxxMat0) (w_1_0 : Fin 72000 → mxxMat0) (w_2_0 : Fin 72000 → mxxMat0) (w_3_0 : Fin 72000 → mxxMat0) (w_4_0 : Fin 72000 → mxxMat0) (w_5_0 : Fin 72000 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat0) (w_6_3 : Fin 2 → mxxMat0) (w_6_4 : Fin 2 → mxxMat0) (w_6_5 : Fin 2 → mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_25_1 : mxxMat0) (w_25_2 : mxxMat0) (w_25_3 : mxxMat0) (w_25_4 : mxxMat0) (w_25_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_948_6 backend hashModel tape (path ++ [6, i.val]) params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_948_25 backend hashModel tape (path ++ [25, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_948.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_948.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat0 := inputs.1
  let w_1_0 : Fin 72000 → mxxMat0 := inputs.2.1
  let w_2_0 : Fin 72000 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.1
  let w_4_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.2.1
  let w_5_0 : Fin 72000 → mxxMat0 := inputs.2.2.2.2.2.1
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
  parallel_generatedRoot_948.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_948 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_948.Witness,
    parallel_generatedRoot_948.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_949_18.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × Unit

abbrev sequential_parallel_generatedRoot_949_18.constraints_4 (w_75_0 : Fin 8 → mxxMat0) (w_79_0 : Fin 8 → mxxMat0) (w_81_0 : Fin 8 → mxxMat0) (w_83_0 : Fin 8 → mxxMat0) (w_41_0 : mxxMat2) (w_45_0 : mxxMat0) (w_58_0 : mxxMat2) (w_62_0 : mxxMat0) (w_65_0 : mxxMat0) (w_68_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_77_0 : Int) (w_78_0 : mxxMat0) (w_80_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_949_18.constraints_3 (w_41_0 : mxxMat2) (w_52_0 : mxxMat2) (w_58_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_64_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev sequential_parallel_generatedRoot_949_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_35_0 : mxxMat2) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_49_0 : mxxMat3) (w_50_0 : mxxMat5) (w_51_0 : mxxMat5) (w_52_0 : mxxMat2) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_55_0 : mxxMat3) (w_56_0 : mxxMat5) (w_57_0 : mxxMat5) : Prop :=
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

abbrev sequential_parallel_generatedRoot_949_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_21_0 : mxxMat2) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_28_concat_1 : mxxMat3) (w_29_0 : mxxMat4) (w_32_0 : mxxMat3) (w_33_0 : mxxMat5) (w_34_0 : mxxMat5) (w_35_0 : mxxMat2) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) : Prop :=
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

abbrev sequential_parallel_generatedRoot_949_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_22_0 : mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat2) (w_13_concat_1 : mxxMat3) (w_14_0 : mxxMat4) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat2) (w_21_concat_1 : mxxMat3) (w_23_0 : mxxMat0) : Prop :=
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

def sequential_parallel_generatedRoot_949_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : (sequential_parallel_generatedRoot_949_18.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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
        ∃ (w_8_0 : mxxMat2),
          ∃ (w_8_concat_1 : mxxMat3),
            ∃ (w_10_0 : mxxMat0),
              ∃ (w_11_0 : mxxMat0),
                ∃ (w_12_0 : mxxMat0),
                  ∃ (w_13_0 : mxxMat2),
                    ∃ (w_13_concat_1 : mxxMat3),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_16_0 : mxxMat0),
                          ∃ (w_18_0 : mxxMat0),
                            ∃ (w_20_0 : mxxMat0),
                              ∃ (w_21_0 : mxxMat2),
                                ∃ (w_21_concat_1 : mxxMat3),
                                  ∃ (w_23_0 : mxxMat0),
                                    ∃ (w_25_0 : mxxMat0),
                                      ∃ (w_27_0 : mxxMat0),
                                        ∃ (w_28_0 : mxxMat2),
                                          ∃ (w_28_concat_1 : mxxMat3),
                                            ∃ (w_29_0 : mxxMat4),
                                              ∃ (w_32_0 : mxxMat3),
                                                ∃ (w_33_0 : mxxMat5),
                                                  ∃ (w_34_0 : mxxMat5),
                                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat0),
                                                      ∃ (w_37_0 : mxxMat0),
                                                        ∃ (w_38_0 : mxxMat3),
                                                          ∃ (w_39_0 : mxxMat5),
                                                            ∃ (w_40_0 : mxxMat5),
                                                              let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat0),
                                                                ∃ (w_43_0 : mxxMat0),
                                                                  let w_44_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat3),
                                                                    ∃ (w_50_0 : mxxMat5),
                                                                      ∃ (w_51_0 : mxxMat5),
                                                                        let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat0),
                                                                          ∃ (w_54_0 : mxxMat0),
                                                                            ∃ (w_55_0 : mxxMat3),
                                                                              ∃ (w_56_0 : mxxMat5),
                                                                                ∃ (w_57_0 : mxxMat5),
                                                                                  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_14_0 w_57_0
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
                                                                                                      sequential_parallel_generatedRoot_949_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_949_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_949_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_949_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_949_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev parallel_generatedRoot_949.InputType : Type :=
  (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_949.Witness where
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
  w_23_0 : mxxMat2
  w_23_concat_1 : mxxMat3
  w_24_0 : mxxMat0
  w_25_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat2
  w_27_concat_1 : mxxMat3
  w_28_0 : mxxMat4
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat2
  w_32_concat_1 : mxxMat3
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat2
  w_36_concat_1 : mxxMat3
  w_37_0 : mxxMat4
  w_38_0 : mxxMat3
  w_39_0 : mxxMat5
  w_40_0 : mxxMat5
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat3
  w_45_0 : mxxMat5
  w_46_0 : mxxMat5
  w_48_0 : mxxMat0
  w_50_0 : mxxMat3
  w_51_0 : mxxMat5
  w_52_0 : mxxMat5
  w_54_0 : mxxMat0
  w_55_0 : mxxMat0
  w_56_0 : mxxMat3
  w_57_0 : mxxMat5
  w_58_0 : mxxMat5
  w_60_0 : mxxMat0
  w_62_0 : mxxMat0
  w_64_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0
  w_70_0 : mxxMat0
  w_71_0 : mxxMat0

abbrev parallel_generatedRoot_949.constraints_4 (w_49_0 : mxxMat0) (w_53_0 : mxxMat2) (w_59_0 : mxxMat2) (w_61_0 : mxxMat0) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_73_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_949.constraints_3 (w_41_0 : mxxMat2) (w_47_0 : mxxMat2) (w_59_0 : mxxMat2) (w_62_0 : mxxMat0) (w_64_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_949.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat0) (w_18_9 : mxxMat0) (w_47_0 : mxxMat2) (w_48_0 : mxxMat0) (w_50_0 : mxxMat3) (w_51_0 : mxxMat5) (w_52_0 : mxxMat5) (w_53_0 : mxxMat2) (w_54_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat3) (w_57_0 : mxxMat5) (w_58_0 : mxxMat5) (w_59_0 : mxxMat2) (w_60_0 : mxxMat0) : Prop :=
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

abbrev parallel_generatedRoot_949.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_6 : mxxMat0) (w_18_8 : mxxMat0) (w_32_0 : mxxMat2) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat2) (w_36_concat_1 : mxxMat3) (w_37_0 : mxxMat4) (w_38_0 : mxxMat3) (w_39_0 : mxxMat5) (w_40_0 : mxxMat5) (w_41_0 : mxxMat2) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat3) (w_45_0 : mxxMat5) (w_46_0 : mxxMat5) : Prop :=
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

abbrev parallel_generatedRoot_949.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat0) (w_3_0 : Fin 8 → mxxMat0) (w_5_0 : Fin 8 → mxxMat0) (w_7_0 : Fin 8 → mxxMat0) (w_9_0 : Fin 8 → mxxMat0) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (w_0_0 : mxxMat0) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_18_2 : mxxMat0) (w_18_3 : mxxMat0) (w_18_4 : mxxMat0) (w_18_5 : mxxMat0) (w_18_6 : mxxMat0) (w_18_7 : mxxMat0) (w_18_8 : mxxMat0) (w_18_9 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : Int) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat2) (w_23_concat_1 : mxxMat3) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat2) (w_27_concat_1 : mxxMat3) (w_28_0 : mxxMat4) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat2) (w_32_concat_1 : mxxMat3) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_949_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
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

abbrev parallel_generatedRoot_949.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_949.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_949.Witness) : Prop :=
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
  let w_41_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat2 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat2 := MxxRuntime.matrixMul w_28_0 w_58_0
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
  parallel_generatedRoot_949.constraints_0 backend hashModel tape path params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_949.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_949.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_949.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_949.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_949 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_949.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_949.Witness,
    parallel_generatedRoot_949.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_950.constraints_0 (i_0 : Nat) (w_0_0 : Fin 36000 → mxxMat0) (w_5_0 : Fin 36000 → mxxMat0) (w_9_0 : Fin 36000 → mxxMat0) (w_13_0 : Fin 36000 → mxxMat0) (w_17_0 : Fin 36000 → mxxMat0) (w_21_0 : Fin 36000 → mxxMat0) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  36000 ≠ 0 ∧
  (Int.ofNat i_0) % 36000 = 0 ∧
  (36000) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_1_0 w_10_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_1_0 w_14_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_17_0 w_1_0 w_18_0 ∧
  (8) ≠ 0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_1_0 w_22_0 ∧
  (8) ≠ 0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def parallel_generatedRoot_950 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 36000 → mxxMat0 := inputs.1
  let w_5_0 : Fin 36000 → mxxMat0 := inputs.2.1
  let w_9_0 : Fin 36000 → mxxMat0 := inputs.2.2.1
  let w_13_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.1
  let w_17_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := (MxxIR.exactDiv ((Int.ofNat i_0)) (36000) + (Int.fmod ((Int.ofNat i_0)) (36000)))
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
              parallel_generatedRoot_950.constraints_0 i_0 w_0_0 w_5_0 w_9_0 w_13_0 w_17_0 w_21_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 outputs

abbrev sequential_generatedRoot_969.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × Unit

abbrev sequential_generatedRoot_969.constraints_0 (w_1_0 : Fin 36000 → mxxMat0) (w_6_0 : Fin 36000 → mxxMat0) (w_10_0 : Fin 36000 → mxxMat0) (w_14_0 : Fin 36000 → mxxMat0) (w_18_0 : Fin 36000 → mxxMat0) (w_22_0 : Fin 36000 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_2_0 w_19_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_2_0 w_23_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def sequential_generatedRoot_969 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_969.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_5_0 : mxxMat0 := inputs.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
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
              sequential_generatedRoot_969.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev scope_81.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × Unit

abbrev scope_81.constraints_0 (w_1_0 : Fin 36000 → mxxMat0) (w_6_0 : Fin 36000 → mxxMat0) (w_10_0 : Fin 36000 → mxxMat0) (w_14_0 : Fin 36000 → mxxMat0) (w_18_0 : Fin 36000 → mxxMat0) (w_22_0 : Fin 36000 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_2_0 w_19_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_2_0 w_23_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ()))))))

def scope_81 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (scope_81.InputType)) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_5_0 : mxxMat0 := inputs.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.1
  let w_17_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := ((10 + (8 * (Int.ofNat i_0))) + (Int.ofNat i_1))
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
              scope_81.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev sequential_generatedRoot_981.InputType : Type :=
  mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × (Fin 36000 → mxxMat0) × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_981.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_9_0 : mxxMat0
  w_10_0 : mxxMat0
  w_13_0 : mxxMat0
  w_14_0 : mxxMat0
  w_17_0 : mxxMat0
  w_18_0 : mxxMat0
  w_21_0 : mxxMat0
  w_22_0 : mxxMat0
  w_25_0 : mxxMat0
  w_26_0 : mxxMat0
  w_28_0 : mxxMat0
  w_28_1 : mxxMat0
  w_28_2 : mxxMat0
  w_28_3 : mxxMat0
  w_28_4 : mxxMat0
  w_28_5 : mxxMat0
  w_29_0 : mxxMat9
  w_33_0 : mxxMat9
  w_37_0 : mxxMat9
  w_41_0 : mxxMat9
  w_45_0 : mxxMat9
  w_49_0 : mxxMat9

abbrev sequential_generatedRoot_981.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : Fin 36000 → mxxMat0) (w_8_0 : Fin 36000 → mxxMat0) (w_12_0 : Fin 36000 → mxxMat0) (w_16_0 : Fin 36000 → mxxMat0) (w_20_0 : Fin 36000 → mxxMat0) (w_24_0 : Fin 36000 → mxxMat0) (w_1_0 : mxxMat10) (w_3_0 : Int) (w_5_0 : Int) (w_7_0 : mxxMat0) (w_11_0 : mxxMat0) (w_15_0 : mxxMat0) (w_19_0 : mxxMat0) (w_23_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_28_1 : mxxMat0) (w_28_2 : mxxMat0) (w_28_3 : mxxMat0) (w_28_4 : mxxMat0) (w_28_5 : mxxMat0) (w_29_0 : mxxMat9) (w_30_0 : mxxMat9) (w_32_0 : mxxMat10) (w_33_0 : mxxMat9) (w_34_0 : mxxMat9) (w_36_0 : mxxMat10) (w_37_0 : mxxMat9) (w_38_0 : mxxMat9) (w_40_0 : mxxMat10) (w_41_0 : mxxMat9) (w_42_0 : mxxMat9) (w_44_0 : mxxMat10) (w_45_0 : mxxMat9) (w_46_0 : mxxMat9) (w_48_0 : mxxMat10) (w_49_0 : mxxMat9) (w_50_0 : mxxMat9) (outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_3_0 w_25_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_5_0 w_26_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => scope_81 backend hashModel tape (path ++ [28, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_2_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ())))))))))))) next) (Int.toNat (6)) (w_7_0, (w_11_0, (w_15_0, (w_19_0, (w_23_0, (w_27_0, ())))))) (w_28_0, (w_28_1, (w_28_2, (w_28_3, (w_28_4, (w_28_5, ())))))) ∧
  MxxRuntime.tensorRuns w_1_0 w_28_0 w_29_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  MxxRuntime.tensorRuns w_32_0 w_28_1 w_33_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  MxxRuntime.tensorRuns w_36_0 w_28_2 w_37_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  MxxRuntime.tensorRuns w_40_0 w_28_3 w_41_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  MxxRuntime.tensorRuns w_44_0 w_28_4 w_45_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  MxxRuntime.tensorRuns w_48_0 w_28_5 w_49_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, (w_46_0, (w_50_0, ()))))))

abbrev sequential_generatedRoot_981.constraints_0 (i_0 : Nat) (w_2_0 : Fin 36000 → mxxMat0) (w_8_0 : Fin 36000 → mxxMat0) (w_12_0 : Fin 36000 → mxxMat0) (w_16_0 : Fin 36000 → mxxMat0) (w_20_0 : Fin 36000 → mxxMat0) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_5_0 : Int) (w_6_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_3_0 w_9_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_5_0 w_10_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_3_0 w_13_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_5_0 w_14_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_3_0 w_17_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_5_0 w_18_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_3_0 w_21_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_5_0 w_22_0

abbrev sequential_generatedRoot_981.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_981.InputType)) (outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) (witness : sequential_generatedRoot_981.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat9 := inputs.1
  let w_31_0 : mxxMat9 := inputs.2.1
  let w_35_0 : mxxMat9 := inputs.2.2.1
  let w_39_0 : mxxMat9 := inputs.2.2.2.1
  let w_43_0 : mxxMat9 := inputs.2.2.2.2.1
  let w_47_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_8_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_12_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : Fin 36000 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat10 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)
  let w_3_0 : Int := (8 + (8 * (Int.ofNat i_0)))
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := (9 + (8 * (Int.ofNat i_0)))
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_4_0 w_6_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_10_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_18_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_21_0 w_22_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat0 := MxxRuntime.matrixAdd w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_1 := witness.w_28_1
  let w_28_2 := witness.w_28_2
  let w_28_3 := witness.w_28_3
  let w_28_4 := witness.w_28_4
  let w_28_5 := witness.w_28_5
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat9 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 : mxxMat10 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat9 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 : mxxMat10 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)
  let w_37_0 := witness.w_37_0
  let w_38_0 : mxxMat9 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 : mxxMat10 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat9 := MxxRuntime.matrixAdd w_39_0 w_41_0
  let w_44_0 : mxxMat10 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat9 := MxxRuntime.matrixAdd w_43_0 w_45_0
  let w_48_0 : mxxMat10 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat10)
  let w_49_0 := witness.w_49_0
  let w_50_0 : mxxMat9 := MxxRuntime.matrixAdd w_47_0 w_49_0
  sequential_generatedRoot_981.constraints_0 i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_20_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_17_0 w_18_0 w_21_0 w_22_0 ∧
    sequential_generatedRoot_981.constraints_1 backend hashModel tape path params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_20_0 w_24_0 w_1_0 w_3_0 w_5_0 w_7_0 w_11_0 w_15_0 w_19_0 w_23_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_1 w_28_2 w_28_3 w_28_4 w_28_5 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_36_0 w_37_0 w_38_0 w_40_0 w_41_0 w_42_0 w_44_0 w_45_0 w_46_0 w_48_0 w_49_0 w_50_0 outputs

def sequential_generatedRoot_981 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (sequential_generatedRoot_981.InputType)) (outputs : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_981.Witness,
    sequential_generatedRoot_981.body backend hashModel tape path params i_0 inputs outputs witness

abbrev parallel_generatedRoot_983.constraints_0 (i_0 : Nat) (w_2_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) ≤ 4294967295 ∧
  outputs = w_2_0

def parallel_generatedRoot_983 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : mxxMat1 := inputs
  let w_0_0 : mxxMat1 := (MxxRuntime.liftInteger ((8) ^ ((Int.ofNat i_0)).toNat) : mxxMat1)
  let w_2_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_1_0
  parallel_generatedRoot_983.constraints_0 i_0 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_984_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_984_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_984_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_984_48.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) < ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) < ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) < ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_984_48 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_984_48.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_984_89.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_984_89 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_984_89.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_984_92.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) < ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) < ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) < ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((3125 * (Int.ofNat i_0)) + (112500 * (Int.ofNat i_1))) ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_984_92 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_984_92.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev parallel_generatedRoot_984.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat3 × mxxMat0 × mxxMat7 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat7 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_984.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat5
  w_7_0 : mxxMat5
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat15
  w_17_0 : mxxMat0
  w_20_0 : mxxMat6
  w_21_0 : mxxMat6
  w_26_0 : mxxMat11
  w_28_0 : mxxMat16
  w_29_0 : mxxMat16
  w_32_0 : mxxMat17
  w_33_0 : mxxMat17
  w_35_0 : mxxMat0
  w_37_0 : mxxMat12
  w_38_0 : mxxMat12
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_48_0 : mxxMat0
  w_48_1 : mxxMat0
  w_48_2 : mxxMat0
  w_52_0 : mxxMat14
  w_55_0 : mxxMat5
  w_56_0 : mxxMat5
  w_59_0 : mxxMat0
  w_60_0 : mxxMat0
  w_61_0 : mxxMat15
  w_66_0 : mxxMat0
  w_69_0 : mxxMat6
  w_70_0 : mxxMat6
  w_75_0 : mxxMat11
  w_76_0 : mxxMat16
  w_77_0 : mxxMat16
  w_80_0 : mxxMat17
  w_81_0 : mxxMat17
  w_83_0 : mxxMat0
  w_85_0 : mxxMat12
  w_86_0 : mxxMat12
  w_88_0 : mxxMat0
  w_89_0 : mxxMat0
  w_92_0 : mxxMat0
  w_92_1 : mxxMat0
  w_92_2 : mxxMat0
  w_94_0 : mxxMat3

abbrev parallel_generatedRoot_984.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_49_0 : mxxMat0) (w_73_0 : mxxMat0) (w_82_0 : mxxMat11) (w_83_0 : mxxMat0) (w_84_0 : mxxMat1) (w_85_0 : mxxMat12) (w_86_0 : mxxMat12) (w_87_0 : mxxMat1) (w_88_0 : mxxMat0) (w_89_0 : mxxMat0) (w_91_0 : mxxMat0) (w_92_0 : mxxMat0) (w_92_1 : mxxMat0) (w_92_2 : mxxMat0) (w_93_0 : mxxMat0) (w_94_0 : mxxMat3) (outputs : mxxMat3) : Prop :=
  MxxRuntime.sliceMatrix w_82_0 0 1 0 25 w_83_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_84_0 w_85_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_85_0 w_86_0 ∧
  MxxRuntime.tensorRuns w_73_0 w_87_0 w_88_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_984_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, (w_82_0, (w_86_0, ()))) next) (Int.toNat (24)) w_88_0 w_89_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_984_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_91_0, (w_91_0, (w_91_0, ()))) (w_92_0, (w_92_1, (w_92_2, ()))) ∧
  MxxRuntime.concatRows w_49_0 w_93_0 w_94_0 ∧
  outputs = w_94_0

abbrev parallel_generatedRoot_984.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat0) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_50_0 : mxxMat0) (w_51_0 : mxxMat0) (w_74_0 : mxxMat7) (w_34_0 : mxxMat11) (w_38_0 : mxxMat12) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_43_0 : mxxMat0) (w_48_0 : mxxMat0) (w_48_1 : mxxMat0) (w_48_2 : mxxMat0) (w_52_0 : mxxMat14) (w_54_0 : mxxMat3) (w_55_0 : mxxMat5) (w_56_0 : mxxMat5) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat15) (w_65_0 : mxxMat15) (w_66_0 : mxxMat0) (w_68_0 : mxxMat0) (w_69_0 : mxxMat6) (w_70_0 : mxxMat6) (w_75_0 : mxxMat11) (w_76_0 : mxxMat16) (w_77_0 : mxxMat16) (w_79_0 : mxxMat16) (w_80_0 : mxxMat17) (w_81_0 : mxxMat17) : Prop :=
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_984_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_984_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_43_0, (w_43_0, (w_43_0, ()))) (w_48_0, (w_48_1, (w_48_2, ()))) ∧
  MxxRuntime.concatColumns w_50_0 w_51_0 w_52_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_59_0 ∧
  MxxRuntime.centeredRebaseRuns w_59_0 w_60_0 ∧
  MxxRuntime.modulusReduceRuns w_60_0 w_61_0 ∧
  MxxRuntime.centeredRebaseRuns w_65_0 w_66_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_68_0 w_69_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_69_0 w_70_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_74_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_75_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_76_0 ∧
  MxxRuntime.centeredRebaseRuns w_76_0 w_77_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625

abbrev parallel_generatedRoot_984.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_25_0 : mxxMat7) (w_27_0 : mxxMat0) (w_2_0 : mxxMat14) (w_5_0 : mxxMat3) (w_6_0 : mxxMat5) (w_7_0 : mxxMat5) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat15) (w_16_0 : mxxMat15) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat6) (w_24_0 : mxxMat0) (w_26_0 : mxxMat11) (w_28_0 : mxxMat16) (w_29_0 : mxxMat16) (w_31_0 : mxxMat16) (w_32_0 : mxxMat17) (w_33_0 : mxxMat17) (w_34_0 : mxxMat11) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat12) (w_38_0 : mxxMat12) (w_39_0 : mxxMat1) (w_40_0 : mxxMat0) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_5_0 w_6_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_6_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.centeredRebaseRuns w_10_0 w_11_0 ∧
  MxxRuntime.modulusReduceRuns w_11_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_16_0 w_17_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_19_0 w_20_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_20_0 w_21_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_25_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_26_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_28_0 ∧
  MxxRuntime.centeredRebaseRuns w_28_0 w_29_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_31_0 w_32_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_32_0 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_34_0 0 1 0 25 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_36_0 w_37_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_37_0 w_38_0 ∧
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0

abbrev parallel_generatedRoot_984.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_984.InputType)) (outputs : mxxMat3) (witness : parallel_generatedRoot_984.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat3 := inputs.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.1
  let w_25_0 : mxxMat7 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_44_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  let w_45_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.1
  let w_47_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_58_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_74_0 : mxxMat7 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat1)
  let w_5_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat18 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat18)
  let w_14_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat18 := (MxxRuntime.matrixPolynomial [(11560550400)] : mxxMat18)
  let w_16_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat1)
  let w_19_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_31_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat11 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat1)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat1 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_43_0 : mxxMat0 := (0 : mxxMat0)
  let w_48_0 := witness.w_48_0
  let w_48_1 := witness.w_48_1
  let w_48_2 := witness.w_48_2
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_48_1
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat1)
  let w_54_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_53_0 w_4_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMul w_52_0 w_56_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat18 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat18)
  let w_63_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_61_0 w_62_0
  let w_64_0 : mxxMat18 := (MxxRuntime.matrixPolynomial [(11560550400)] : mxxMat18)
  let w_65_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_63_0 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat1)
  let w_68_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_66_0 w_67_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixMul w_58_0 w_70_0
  let w_72_0 : mxxMat0 := MxxRuntime.matrixSub w_57_0 w_71_0
  let w_73_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_79_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_77_0 w_78_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat11 := MxxRuntime.matrixMul w_75_0 w_81_0
  let w_83_0 := witness.w_83_0
  let w_84_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat1)
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 : mxxMat1 := MxxRuntime.matrixMul w_83_0 w_86_0
  let w_88_0 := witness.w_88_0
  let w_89_0 := witness.w_89_0
  let w_90_0 : mxxMat0 := MxxRuntime.matrixAdd w_72_0 w_89_0
  let w_91_0 : mxxMat0 := (0 : mxxMat0)
  let w_92_0 := witness.w_92_0
  let w_92_1 := witness.w_92_1
  let w_92_2 := witness.w_92_2
  let w_93_0 : mxxMat0 := MxxRuntime.matrixAdd w_90_0 w_92_2
  let w_94_0 := witness.w_94_0
  parallel_generatedRoot_984.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_24_0 w_26_0 w_28_0 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    parallel_generatedRoot_984.constraints_1 backend hashModel tape path params i_0 w_27_0 w_44_0 w_45_0 w_46_0 w_47_0 w_50_0 w_51_0 w_74_0 w_34_0 w_38_0 w_40_0 w_41_0 w_43_0 w_48_0 w_48_1 w_48_2 w_52_0 w_54_0 w_55_0 w_56_0 w_59_0 w_60_0 w_61_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_75_0 w_76_0 w_77_0 w_79_0 w_80_0 w_81_0 ∧
    parallel_generatedRoot_984.constraints_2 backend hashModel tape path params i_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_73_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_91_0 w_92_0 w_92_1 w_92_2 w_93_0 w_94_0 outputs

def parallel_generatedRoot_984 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_984.InputType)) (outputs : mxxMat3) : Prop :=
  ∃ witness : parallel_generatedRoot_984.Witness,
    parallel_generatedRoot_984.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_985_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_985_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_985_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_985_48.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_985_48 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_985_48.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_985_89.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_985_89 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_985_89.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_985_92.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((625 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_985_92 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_985_92.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev parallel_generatedRoot_985.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat3 × mxxMat0 × mxxMat7 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat7 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_985.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat5
  w_7_0 : mxxMat5
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat19
  w_17_0 : mxxMat0
  w_20_0 : mxxMat6
  w_21_0 : mxxMat6
  w_26_0 : mxxMat11
  w_28_0 : mxxMat16
  w_29_0 : mxxMat16
  w_32_0 : mxxMat17
  w_33_0 : mxxMat17
  w_35_0 : mxxMat0
  w_37_0 : mxxMat12
  w_38_0 : mxxMat12
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_48_0 : mxxMat0
  w_48_1 : mxxMat0
  w_48_2 : mxxMat0
  w_52_0 : mxxMat14
  w_55_0 : mxxMat5
  w_56_0 : mxxMat5
  w_59_0 : mxxMat0
  w_60_0 : mxxMat0
  w_61_0 : mxxMat19
  w_66_0 : mxxMat0
  w_69_0 : mxxMat6
  w_70_0 : mxxMat6
  w_75_0 : mxxMat11
  w_76_0 : mxxMat16
  w_77_0 : mxxMat16
  w_80_0 : mxxMat17
  w_81_0 : mxxMat17
  w_83_0 : mxxMat0
  w_85_0 : mxxMat12
  w_86_0 : mxxMat12
  w_88_0 : mxxMat0
  w_89_0 : mxxMat0
  w_92_0 : mxxMat0
  w_92_1 : mxxMat0
  w_92_2 : mxxMat0
  w_94_0 : mxxMat3

abbrev parallel_generatedRoot_985.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_49_0 : mxxMat0) (w_73_0 : mxxMat0) (w_82_0 : mxxMat11) (w_83_0 : mxxMat0) (w_84_0 : mxxMat1) (w_85_0 : mxxMat12) (w_86_0 : mxxMat12) (w_87_0 : mxxMat1) (w_88_0 : mxxMat0) (w_89_0 : mxxMat0) (w_91_0 : mxxMat0) (w_92_0 : mxxMat0) (w_92_1 : mxxMat0) (w_92_2 : mxxMat0) (w_93_0 : mxxMat0) (w_94_0 : mxxMat3) (outputs : mxxMat3) : Prop :=
  MxxRuntime.sliceMatrix w_82_0 0 1 0 25 w_83_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_84_0 w_85_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_85_0 w_86_0 ∧
  MxxRuntime.tensorRuns w_73_0 w_87_0 w_88_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_985_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, (w_82_0, (w_86_0, ()))) next) (Int.toNat (24)) w_88_0 w_89_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_985_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_91_0, (w_91_0, (w_91_0, ()))) (w_92_0, (w_92_1, (w_92_2, ()))) ∧
  MxxRuntime.concatRows w_49_0 w_93_0 w_94_0 ∧
  outputs = w_94_0

abbrev parallel_generatedRoot_985.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat0) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_50_0 : mxxMat0) (w_51_0 : mxxMat0) (w_74_0 : mxxMat7) (w_34_0 : mxxMat11) (w_38_0 : mxxMat12) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_43_0 : mxxMat0) (w_48_0 : mxxMat0) (w_48_1 : mxxMat0) (w_48_2 : mxxMat0) (w_52_0 : mxxMat14) (w_54_0 : mxxMat3) (w_55_0 : mxxMat5) (w_56_0 : mxxMat5) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat19) (w_65_0 : mxxMat19) (w_66_0 : mxxMat0) (w_68_0 : mxxMat0) (w_69_0 : mxxMat6) (w_70_0 : mxxMat6) (w_75_0 : mxxMat11) (w_76_0 : mxxMat16) (w_77_0 : mxxMat16) (w_79_0 : mxxMat16) (w_80_0 : mxxMat17) (w_81_0 : mxxMat17) : Prop :=
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_985_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_985_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_43_0, (w_43_0, (w_43_0, ()))) (w_48_0, (w_48_1, (w_48_2, ()))) ∧
  MxxRuntime.concatColumns w_50_0 w_51_0 w_52_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_59_0 ∧
  MxxRuntime.centeredRebaseRuns w_59_0 w_60_0 ∧
  MxxRuntime.modulusReduceRuns w_60_0 w_61_0 ∧
  MxxRuntime.centeredRebaseRuns w_65_0 w_66_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_68_0 w_69_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_69_0 w_70_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_74_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_75_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_76_0 ∧
  MxxRuntime.centeredRebaseRuns w_76_0 w_77_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625

abbrev parallel_generatedRoot_985.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_25_0 : mxxMat7) (w_27_0 : mxxMat0) (w_2_0 : mxxMat14) (w_5_0 : mxxMat3) (w_6_0 : mxxMat5) (w_7_0 : mxxMat5) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat19) (w_16_0 : mxxMat19) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat6) (w_24_0 : mxxMat0) (w_26_0 : mxxMat11) (w_28_0 : mxxMat16) (w_29_0 : mxxMat16) (w_31_0 : mxxMat16) (w_32_0 : mxxMat17) (w_33_0 : mxxMat17) (w_34_0 : mxxMat11) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat12) (w_38_0 : mxxMat12) (w_39_0 : mxxMat1) (w_40_0 : mxxMat0) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_5_0 w_6_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_6_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.centeredRebaseRuns w_10_0 w_11_0 ∧
  MxxRuntime.modulusReduceRuns w_11_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_16_0 w_17_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_19_0 w_20_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_20_0 w_21_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_25_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_26_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_28_0 ∧
  MxxRuntime.centeredRebaseRuns w_28_0 w_29_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_31_0 w_32_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_32_0 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_34_0 0 1 0 25 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_36_0 w_37_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_37_0 w_38_0 ∧
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0

abbrev parallel_generatedRoot_985.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_985.InputType)) (outputs : mxxMat3) (witness : parallel_generatedRoot_985.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat3 := inputs.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.1
  let w_25_0 : mxxMat7 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_44_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  let w_45_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.1
  let w_47_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_58_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_74_0 : mxxMat7 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat1)
  let w_5_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat20 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat20)
  let w_14_0 : mxxMat19 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat20 := (MxxRuntime.matrixPolynomial [(1125894775373377)] : mxxMat20)
  let w_16_0 : mxxMat19 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat4)] : mxxMat1)
  let w_19_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_31_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat11 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat1)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat1 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_43_0 : mxxMat0 := (0 : mxxMat0)
  let w_48_0 := witness.w_48_0
  let w_48_1 := witness.w_48_1
  let w_48_2 := witness.w_48_2
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_48_1
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat1)
  let w_54_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_53_0 w_4_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMul w_52_0 w_56_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat20 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat20)
  let w_63_0 : mxxMat19 := MxxRuntime.matrixMulScalarRight w_61_0 w_62_0
  let w_64_0 : mxxMat20 := (MxxRuntime.matrixPolynomial [(1125894775373377)] : mxxMat20)
  let w_65_0 : mxxMat19 := MxxRuntime.matrixMulScalarRight w_63_0 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat4)] : mxxMat1)
  let w_68_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_66_0 w_67_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixMul w_58_0 w_70_0
  let w_72_0 : mxxMat0 := MxxRuntime.matrixSub w_57_0 w_71_0
  let w_73_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_79_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_77_0 w_78_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat11 := MxxRuntime.matrixMul w_75_0 w_81_0
  let w_83_0 := witness.w_83_0
  let w_84_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat1)
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 : mxxMat1 := MxxRuntime.matrixMul w_83_0 w_86_0
  let w_88_0 := witness.w_88_0
  let w_89_0 := witness.w_89_0
  let w_90_0 : mxxMat0 := MxxRuntime.matrixAdd w_72_0 w_89_0
  let w_91_0 : mxxMat0 := (0 : mxxMat0)
  let w_92_0 := witness.w_92_0
  let w_92_1 := witness.w_92_1
  let w_92_2 := witness.w_92_2
  let w_93_0 : mxxMat0 := MxxRuntime.matrixAdd w_90_0 w_92_2
  let w_94_0 := witness.w_94_0
  parallel_generatedRoot_985.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_24_0 w_26_0 w_28_0 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    parallel_generatedRoot_985.constraints_1 backend hashModel tape path params i_0 w_27_0 w_44_0 w_45_0 w_46_0 w_47_0 w_50_0 w_51_0 w_74_0 w_34_0 w_38_0 w_40_0 w_41_0 w_43_0 w_48_0 w_48_1 w_48_2 w_52_0 w_54_0 w_55_0 w_56_0 w_59_0 w_60_0 w_61_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_75_0 w_76_0 w_77_0 w_79_0 w_80_0 w_81_0 ∧
    parallel_generatedRoot_985.constraints_2 backend hashModel tape path params i_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_73_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_91_0 w_92_0 w_92_1 w_92_2 w_93_0 w_94_0 outputs

def parallel_generatedRoot_985 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_985.InputType)) (outputs : mxxMat3) : Prop :=
  ∃ witness : parallel_generatedRoot_985.Witness,
    parallel_generatedRoot_985.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_986_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_986_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_986_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_986_48.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_986_48 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_986_48.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_986_89.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_986_89 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_986_89.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_986_92.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((1250 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_986_92 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_986_92.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev parallel_generatedRoot_986.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat3 × mxxMat0 × mxxMat7 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat7 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_986.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat5
  w_7_0 : mxxMat5
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat21
  w_17_0 : mxxMat0
  w_20_0 : mxxMat6
  w_21_0 : mxxMat6
  w_26_0 : mxxMat11
  w_28_0 : mxxMat16
  w_29_0 : mxxMat16
  w_32_0 : mxxMat17
  w_33_0 : mxxMat17
  w_35_0 : mxxMat0
  w_37_0 : mxxMat12
  w_38_0 : mxxMat12
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_48_0 : mxxMat0
  w_48_1 : mxxMat0
  w_48_2 : mxxMat0
  w_52_0 : mxxMat14
  w_55_0 : mxxMat5
  w_56_0 : mxxMat5
  w_59_0 : mxxMat0
  w_60_0 : mxxMat0
  w_61_0 : mxxMat21
  w_66_0 : mxxMat0
  w_69_0 : mxxMat6
  w_70_0 : mxxMat6
  w_75_0 : mxxMat11
  w_76_0 : mxxMat16
  w_77_0 : mxxMat16
  w_80_0 : mxxMat17
  w_81_0 : mxxMat17
  w_83_0 : mxxMat0
  w_85_0 : mxxMat12
  w_86_0 : mxxMat12
  w_88_0 : mxxMat0
  w_89_0 : mxxMat0
  w_92_0 : mxxMat0
  w_92_1 : mxxMat0
  w_92_2 : mxxMat0
  w_94_0 : mxxMat3

abbrev parallel_generatedRoot_986.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_49_0 : mxxMat0) (w_73_0 : mxxMat0) (w_82_0 : mxxMat11) (w_83_0 : mxxMat0) (w_84_0 : mxxMat1) (w_85_0 : mxxMat12) (w_86_0 : mxxMat12) (w_87_0 : mxxMat1) (w_88_0 : mxxMat0) (w_89_0 : mxxMat0) (w_91_0 : mxxMat0) (w_92_0 : mxxMat0) (w_92_1 : mxxMat0) (w_92_2 : mxxMat0) (w_93_0 : mxxMat0) (w_94_0 : mxxMat3) (outputs : mxxMat3) : Prop :=
  MxxRuntime.sliceMatrix w_82_0 0 1 0 25 w_83_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_84_0 w_85_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_85_0 w_86_0 ∧
  MxxRuntime.tensorRuns w_73_0 w_87_0 w_88_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_986_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, (w_82_0, (w_86_0, ()))) next) (Int.toNat (24)) w_88_0 w_89_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_986_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_91_0, (w_91_0, (w_91_0, ()))) (w_92_0, (w_92_1, (w_92_2, ()))) ∧
  MxxRuntime.concatRows w_49_0 w_93_0 w_94_0 ∧
  outputs = w_94_0

abbrev parallel_generatedRoot_986.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat0) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_50_0 : mxxMat0) (w_51_0 : mxxMat0) (w_74_0 : mxxMat7) (w_34_0 : mxxMat11) (w_38_0 : mxxMat12) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_43_0 : mxxMat0) (w_48_0 : mxxMat0) (w_48_1 : mxxMat0) (w_48_2 : mxxMat0) (w_52_0 : mxxMat14) (w_54_0 : mxxMat3) (w_55_0 : mxxMat5) (w_56_0 : mxxMat5) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat21) (w_65_0 : mxxMat21) (w_66_0 : mxxMat0) (w_68_0 : mxxMat0) (w_69_0 : mxxMat6) (w_70_0 : mxxMat6) (w_75_0 : mxxMat11) (w_76_0 : mxxMat16) (w_77_0 : mxxMat16) (w_79_0 : mxxMat16) (w_80_0 : mxxMat17) (w_81_0 : mxxMat17) : Prop :=
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_986_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_986_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_43_0, (w_43_0, (w_43_0, ()))) (w_48_0, (w_48_1, (w_48_2, ()))) ∧
  MxxRuntime.concatColumns w_50_0 w_51_0 w_52_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_59_0 ∧
  MxxRuntime.centeredRebaseRuns w_59_0 w_60_0 ∧
  MxxRuntime.modulusReduceRuns w_60_0 w_61_0 ∧
  MxxRuntime.centeredRebaseRuns w_65_0 w_66_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_68_0 w_69_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_69_0 w_70_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_74_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_75_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_76_0 ∧
  MxxRuntime.centeredRebaseRuns w_76_0 w_77_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625

abbrev parallel_generatedRoot_986.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_25_0 : mxxMat7) (w_27_0 : mxxMat0) (w_2_0 : mxxMat14) (w_5_0 : mxxMat3) (w_6_0 : mxxMat5) (w_7_0 : mxxMat5) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat21) (w_16_0 : mxxMat21) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat6) (w_24_0 : mxxMat0) (w_26_0 : mxxMat11) (w_28_0 : mxxMat16) (w_29_0 : mxxMat16) (w_31_0 : mxxMat16) (w_32_0 : mxxMat17) (w_33_0 : mxxMat17) (w_34_0 : mxxMat11) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat12) (w_38_0 : mxxMat12) (w_39_0 : mxxMat1) (w_40_0 : mxxMat0) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_5_0 w_6_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_6_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.centeredRebaseRuns w_10_0 w_11_0 ∧
  MxxRuntime.modulusReduceRuns w_11_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_16_0 w_17_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_19_0 w_20_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_20_0 w_21_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_25_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_26_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_28_0 ∧
  MxxRuntime.centeredRebaseRuns w_28_0 w_29_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_31_0 w_32_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_32_0 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_34_0 0 1 0 25 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_36_0 w_37_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_37_0 w_38_0 ∧
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0

abbrev parallel_generatedRoot_986.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_986.InputType)) (outputs : mxxMat3) (witness : parallel_generatedRoot_986.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat3 := inputs.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.1
  let w_25_0 : mxxMat7 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_44_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  let w_45_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.1
  let w_47_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_58_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_74_0 : mxxMat7 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat1)
  let w_5_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat22 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat22)
  let w_14_0 : mxxMat21 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat22 := (MxxRuntime.matrixPolynomial [(7254835200)] : mxxMat22)
  let w_16_0 : mxxMat21 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat5)] : mxxMat1)
  let w_19_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_31_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat11 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat1)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat1 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_43_0 : mxxMat0 := (0 : mxxMat0)
  let w_48_0 := witness.w_48_0
  let w_48_1 := witness.w_48_1
  let w_48_2 := witness.w_48_2
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_48_1
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat1)
  let w_54_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_53_0 w_4_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMul w_52_0 w_56_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat22 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat22)
  let w_63_0 : mxxMat21 := MxxRuntime.matrixMulScalarRight w_61_0 w_62_0
  let w_64_0 : mxxMat22 := (MxxRuntime.matrixPolynomial [(7254835200)] : mxxMat22)
  let w_65_0 : mxxMat21 := MxxRuntime.matrixMulScalarRight w_63_0 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat5)] : mxxMat1)
  let w_68_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_66_0 w_67_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixMul w_58_0 w_70_0
  let w_72_0 : mxxMat0 := MxxRuntime.matrixSub w_57_0 w_71_0
  let w_73_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_79_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_77_0 w_78_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat11 := MxxRuntime.matrixMul w_75_0 w_81_0
  let w_83_0 := witness.w_83_0
  let w_84_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat1)
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 : mxxMat1 := MxxRuntime.matrixMul w_83_0 w_86_0
  let w_88_0 := witness.w_88_0
  let w_89_0 := witness.w_89_0
  let w_90_0 : mxxMat0 := MxxRuntime.matrixAdd w_72_0 w_89_0
  let w_91_0 : mxxMat0 := (0 : mxxMat0)
  let w_92_0 := witness.w_92_0
  let w_92_1 := witness.w_92_1
  let w_92_2 := witness.w_92_2
  let w_93_0 : mxxMat0 := MxxRuntime.matrixAdd w_90_0 w_92_2
  let w_94_0 := witness.w_94_0
  parallel_generatedRoot_986.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_24_0 w_26_0 w_28_0 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    parallel_generatedRoot_986.constraints_1 backend hashModel tape path params i_0 w_27_0 w_44_0 w_45_0 w_46_0 w_47_0 w_50_0 w_51_0 w_74_0 w_34_0 w_38_0 w_40_0 w_41_0 w_43_0 w_48_0 w_48_1 w_48_2 w_52_0 w_54_0 w_55_0 w_56_0 w_59_0 w_60_0 w_61_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_75_0 w_76_0 w_77_0 w_79_0 w_80_0 w_81_0 ∧
    parallel_generatedRoot_986.constraints_2 backend hashModel tape path params i_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_73_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_91_0 w_92_0 w_92_1 w_92_2 w_93_0 w_94_0 outputs

def parallel_generatedRoot_986 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_986.InputType)) (outputs : mxxMat3) : Prop :=
  ∃ witness : parallel_generatedRoot_986.Witness,
    parallel_generatedRoot_986.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_987_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_987_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_987_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_987_48.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_987_48 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_987_48.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_987_89.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_987_89 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_987_89.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_987_92.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((1875 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_987_92 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_987_92.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev parallel_generatedRoot_987.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat3 × mxxMat0 × mxxMat7 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat7 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_987.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat5
  w_7_0 : mxxMat5
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat23
  w_17_0 : mxxMat0
  w_20_0 : mxxMat6
  w_21_0 : mxxMat6
  w_26_0 : mxxMat11
  w_28_0 : mxxMat16
  w_29_0 : mxxMat16
  w_32_0 : mxxMat17
  w_33_0 : mxxMat17
  w_35_0 : mxxMat0
  w_37_0 : mxxMat12
  w_38_0 : mxxMat12
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_48_0 : mxxMat0
  w_48_1 : mxxMat0
  w_48_2 : mxxMat0
  w_52_0 : mxxMat14
  w_55_0 : mxxMat5
  w_56_0 : mxxMat5
  w_59_0 : mxxMat0
  w_60_0 : mxxMat0
  w_61_0 : mxxMat23
  w_66_0 : mxxMat0
  w_69_0 : mxxMat6
  w_70_0 : mxxMat6
  w_75_0 : mxxMat11
  w_76_0 : mxxMat16
  w_77_0 : mxxMat16
  w_80_0 : mxxMat17
  w_81_0 : mxxMat17
  w_83_0 : mxxMat0
  w_85_0 : mxxMat12
  w_86_0 : mxxMat12
  w_88_0 : mxxMat0
  w_89_0 : mxxMat0
  w_92_0 : mxxMat0
  w_92_1 : mxxMat0
  w_92_2 : mxxMat0
  w_94_0 : mxxMat3

abbrev parallel_generatedRoot_987.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_49_0 : mxxMat0) (w_73_0 : mxxMat0) (w_82_0 : mxxMat11) (w_83_0 : mxxMat0) (w_84_0 : mxxMat1) (w_85_0 : mxxMat12) (w_86_0 : mxxMat12) (w_87_0 : mxxMat1) (w_88_0 : mxxMat0) (w_89_0 : mxxMat0) (w_91_0 : mxxMat0) (w_92_0 : mxxMat0) (w_92_1 : mxxMat0) (w_92_2 : mxxMat0) (w_93_0 : mxxMat0) (w_94_0 : mxxMat3) (outputs : mxxMat3) : Prop :=
  MxxRuntime.sliceMatrix w_82_0 0 1 0 25 w_83_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_84_0 w_85_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_85_0 w_86_0 ∧
  MxxRuntime.tensorRuns w_73_0 w_87_0 w_88_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_987_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, (w_82_0, (w_86_0, ()))) next) (Int.toNat (24)) w_88_0 w_89_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_987_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_91_0, (w_91_0, (w_91_0, ()))) (w_92_0, (w_92_1, (w_92_2, ()))) ∧
  MxxRuntime.concatRows w_49_0 w_93_0 w_94_0 ∧
  outputs = w_94_0

abbrev parallel_generatedRoot_987.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat0) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_50_0 : mxxMat0) (w_51_0 : mxxMat0) (w_74_0 : mxxMat7) (w_34_0 : mxxMat11) (w_38_0 : mxxMat12) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_43_0 : mxxMat0) (w_48_0 : mxxMat0) (w_48_1 : mxxMat0) (w_48_2 : mxxMat0) (w_52_0 : mxxMat14) (w_54_0 : mxxMat3) (w_55_0 : mxxMat5) (w_56_0 : mxxMat5) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat23) (w_65_0 : mxxMat23) (w_66_0 : mxxMat0) (w_68_0 : mxxMat0) (w_69_0 : mxxMat6) (w_70_0 : mxxMat6) (w_75_0 : mxxMat11) (w_76_0 : mxxMat16) (w_77_0 : mxxMat16) (w_79_0 : mxxMat16) (w_80_0 : mxxMat17) (w_81_0 : mxxMat17) : Prop :=
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_987_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_987_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_43_0, (w_43_0, (w_43_0, ()))) (w_48_0, (w_48_1, (w_48_2, ()))) ∧
  MxxRuntime.concatColumns w_50_0 w_51_0 w_52_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_59_0 ∧
  MxxRuntime.centeredRebaseRuns w_59_0 w_60_0 ∧
  MxxRuntime.modulusReduceRuns w_60_0 w_61_0 ∧
  MxxRuntime.centeredRebaseRuns w_65_0 w_66_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_68_0 w_69_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_69_0 w_70_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_74_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_75_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_76_0 ∧
  MxxRuntime.centeredRebaseRuns w_76_0 w_77_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625

abbrev parallel_generatedRoot_987.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_25_0 : mxxMat7) (w_27_0 : mxxMat0) (w_2_0 : mxxMat14) (w_5_0 : mxxMat3) (w_6_0 : mxxMat5) (w_7_0 : mxxMat5) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat23) (w_16_0 : mxxMat23) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat6) (w_24_0 : mxxMat0) (w_26_0 : mxxMat11) (w_28_0 : mxxMat16) (w_29_0 : mxxMat16) (w_31_0 : mxxMat16) (w_32_0 : mxxMat17) (w_33_0 : mxxMat17) (w_34_0 : mxxMat11) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat12) (w_38_0 : mxxMat12) (w_39_0 : mxxMat1) (w_40_0 : mxxMat0) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_5_0 w_6_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_6_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.centeredRebaseRuns w_10_0 w_11_0 ∧
  MxxRuntime.modulusReduceRuns w_11_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_16_0 w_17_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_19_0 w_20_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_20_0 w_21_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_25_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_26_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_28_0 ∧
  MxxRuntime.centeredRebaseRuns w_28_0 w_29_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_31_0 w_32_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_32_0 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_34_0 0 1 0 25 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_36_0 w_37_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_37_0 w_38_0 ∧
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0

abbrev parallel_generatedRoot_987.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_987.InputType)) (outputs : mxxMat3) (witness : parallel_generatedRoot_987.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat3 := inputs.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.1
  let w_25_0 : mxxMat7 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_44_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  let w_45_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.1
  let w_47_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_58_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_74_0 : mxxMat7 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat1)
  let w_5_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat24 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat24)
  let w_14_0 : mxxMat23 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat24 := (MxxRuntime.matrixPolynomial [(1125871968844913)] : mxxMat24)
  let w_16_0 : mxxMat23 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat6)] : mxxMat1)
  let w_19_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_31_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat11 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat1)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat1 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_43_0 : mxxMat0 := (0 : mxxMat0)
  let w_48_0 := witness.w_48_0
  let w_48_1 := witness.w_48_1
  let w_48_2 := witness.w_48_2
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_48_1
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat1)
  let w_54_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_53_0 w_4_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMul w_52_0 w_56_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat24 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat24)
  let w_63_0 : mxxMat23 := MxxRuntime.matrixMulScalarRight w_61_0 w_62_0
  let w_64_0 : mxxMat24 := (MxxRuntime.matrixPolynomial [(1125871968844913)] : mxxMat24)
  let w_65_0 : mxxMat23 := MxxRuntime.matrixMulScalarRight w_63_0 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat6)] : mxxMat1)
  let w_68_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_66_0 w_67_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixMul w_58_0 w_70_0
  let w_72_0 : mxxMat0 := MxxRuntime.matrixSub w_57_0 w_71_0
  let w_73_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_79_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_77_0 w_78_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat11 := MxxRuntime.matrixMul w_75_0 w_81_0
  let w_83_0 := witness.w_83_0
  let w_84_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat1)
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 : mxxMat1 := MxxRuntime.matrixMul w_83_0 w_86_0
  let w_88_0 := witness.w_88_0
  let w_89_0 := witness.w_89_0
  let w_90_0 : mxxMat0 := MxxRuntime.matrixAdd w_72_0 w_89_0
  let w_91_0 : mxxMat0 := (0 : mxxMat0)
  let w_92_0 := witness.w_92_0
  let w_92_1 := witness.w_92_1
  let w_92_2 := witness.w_92_2
  let w_93_0 : mxxMat0 := MxxRuntime.matrixAdd w_90_0 w_92_2
  let w_94_0 := witness.w_94_0
  parallel_generatedRoot_987.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_24_0 w_26_0 w_28_0 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    parallel_generatedRoot_987.constraints_1 backend hashModel tape path params i_0 w_27_0 w_44_0 w_45_0 w_46_0 w_47_0 w_50_0 w_51_0 w_74_0 w_34_0 w_38_0 w_40_0 w_41_0 w_43_0 w_48_0 w_48_1 w_48_2 w_52_0 w_54_0 w_55_0 w_56_0 w_59_0 w_60_0 w_61_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_75_0 w_76_0 w_77_0 w_79_0 w_80_0 w_81_0 ∧
    parallel_generatedRoot_987.constraints_2 backend hashModel tape path params i_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_73_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_91_0 w_92_0 w_92_1 w_92_2 w_93_0 w_94_0 outputs

def parallel_generatedRoot_987 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_987.InputType)) (outputs : mxxMat3) : Prop :=
  ∃ witness : parallel_generatedRoot_987.Witness,
    parallel_generatedRoot_987.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_988_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_988_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_988_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_988_48.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_988_48 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_988_48.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_988_89.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat11) (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat0) (w_7_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_1))) ∧
  ((1 + (Int.ofNat i_1))) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (25 + (25 * (Int.ofNat i_1))) ∧
  (25 + (25 * (Int.ofNat i_1))) < (50 + (25 * (Int.ofNat i_1))) ∧
  (50 + (25 * (Int.ofNat i_1))) ≤ 625 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (25 + (25 * (Int.ofNat i_1))) (50 + (25 * (Int.ofNat i_1))) w_3_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_5_0 w_6_0 ∧
  outputs = w_7_0

def sequential_parallel_generatedRoot_988_89 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat11 × mxxMat12 × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : mxxMat11 := inputs.2.1
  let w_4_0 : mxxMat12 := inputs.2.2.1
  let w_1_0 : mxxMat0 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat0)
  ∃ (w_3_0 : mxxMat0),
    let w_5_0 : mxxMat1 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat0),
      let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_988_89.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_988_92.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat9) (w_3_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat9) (w_20_0 : mxxMat9) (w_2_0 : mxxMat11) (w_4_0 : Int) (w_5_0 : mxxMat1) (w_6_0 : mxxMat6) (w_7_0 : mxxMat6) (w_9_0 : mxxMat6) (w_10_0 : mxxMat13) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat11) (w_18_0 : mxxMat0) (w_21_0 : mxxMat11) (w_23_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_2_0 ∧
  0 ≤ w_4_0 ∧
  w_4_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_4_0 w_5_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_9_0 w_10_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) < ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ∧
  ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 ((2500 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) ((3125 + (3125 * (Int.ofNat i_0))) + (112500 * (Int.ofNat i_1))) w_21_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_988_92 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 : mxxMat9 := inputs.2.2.2.1
  let w_3_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat9 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat11),
    let w_4_0 : Int := (Int.ofNat i_1)
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat6),
        ∃ (w_7_0 : mxxMat6),
          let w_8_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
          let w_9_0 : mxxMat6 := MxxRuntime.matrixMulScalarRight w_7_0 w_8_0
          ∃ (w_10_0 : mxxMat13),
            ∃ (w_11_0 : mxxMat13),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              ∃ (w_16_0 : mxxMat11),
                let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
                let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
                ∃ (w_21_0 : mxxMat11),
                  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_21_0 w_11_0
                  let w_23_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_22_0
                  sequential_parallel_generatedRoot_988_92.constraints_0 backend i_0 i_1 w_1_0 w_3_0 w_15_0 w_20_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 outputs

abbrev parallel_generatedRoot_988.InputType : Type :=
  mxxMat0 × mxxMat0 × mxxMat3 × mxxMat0 × mxxMat7 × mxxMat0 × mxxMat9 × (Fin 1 → mxxMat1) × mxxMat9 × mxxMat9 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat7 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_988.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat5
  w_7_0 : mxxMat5
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat25
  w_17_0 : mxxMat0
  w_20_0 : mxxMat6
  w_21_0 : mxxMat6
  w_26_0 : mxxMat11
  w_28_0 : mxxMat16
  w_29_0 : mxxMat16
  w_32_0 : mxxMat17
  w_33_0 : mxxMat17
  w_35_0 : mxxMat0
  w_37_0 : mxxMat12
  w_38_0 : mxxMat12
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_48_0 : mxxMat0
  w_48_1 : mxxMat0
  w_48_2 : mxxMat0
  w_52_0 : mxxMat14
  w_55_0 : mxxMat5
  w_56_0 : mxxMat5
  w_59_0 : mxxMat0
  w_60_0 : mxxMat0
  w_61_0 : mxxMat25
  w_66_0 : mxxMat0
  w_69_0 : mxxMat6
  w_70_0 : mxxMat6
  w_75_0 : mxxMat11
  w_76_0 : mxxMat16
  w_77_0 : mxxMat16
  w_80_0 : mxxMat17
  w_81_0 : mxxMat17
  w_83_0 : mxxMat0
  w_85_0 : mxxMat12
  w_86_0 : mxxMat12
  w_88_0 : mxxMat0
  w_89_0 : mxxMat0
  w_92_0 : mxxMat0
  w_92_1 : mxxMat0
  w_92_2 : mxxMat0
  w_94_0 : mxxMat3

abbrev parallel_generatedRoot_988.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_49_0 : mxxMat0) (w_73_0 : mxxMat0) (w_82_0 : mxxMat11) (w_83_0 : mxxMat0) (w_84_0 : mxxMat1) (w_85_0 : mxxMat12) (w_86_0 : mxxMat12) (w_87_0 : mxxMat1) (w_88_0 : mxxMat0) (w_89_0 : mxxMat0) (w_91_0 : mxxMat0) (w_92_0 : mxxMat0) (w_92_1 : mxxMat0) (w_92_2 : mxxMat0) (w_93_0 : mxxMat0) (w_94_0 : mxxMat3) (outputs : mxxMat3) : Prop :=
  MxxRuntime.sliceMatrix w_82_0 0 1 0 25 w_83_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_84_0 w_85_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_85_0 w_86_0 ∧
  MxxRuntime.tensorRuns w_73_0 w_87_0 w_88_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_988_89 backend hashModel tape (path ++ [89, i]) params i_0 i (current, (w_82_0, (w_86_0, ()))) next) (Int.toNat (24)) w_88_0 w_89_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_988_92 backend hashModel tape (path ++ [92, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_91_0, (w_91_0, (w_91_0, ()))) (w_92_0, (w_92_1, (w_92_2, ()))) ∧
  MxxRuntime.concatRows w_49_0 w_93_0 w_94_0 ∧
  outputs = w_94_0

abbrev parallel_generatedRoot_988.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat0) (w_44_0 : mxxMat9) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : mxxMat9) (w_47_0 : mxxMat9) (w_50_0 : mxxMat0) (w_51_0 : mxxMat0) (w_74_0 : mxxMat7) (w_34_0 : mxxMat11) (w_38_0 : mxxMat12) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_43_0 : mxxMat0) (w_48_0 : mxxMat0) (w_48_1 : mxxMat0) (w_48_2 : mxxMat0) (w_52_0 : mxxMat14) (w_54_0 : mxxMat3) (w_55_0 : mxxMat5) (w_56_0 : mxxMat5) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat25) (w_65_0 : mxxMat25) (w_66_0 : mxxMat0) (w_68_0 : mxxMat0) (w_69_0 : mxxMat6) (w_70_0 : mxxMat6) (w_75_0 : mxxMat11) (w_76_0 : mxxMat16) (w_77_0 : mxxMat16) (w_79_0 : mxxMat16) (w_80_0 : mxxMat17) (w_81_0 : mxxMat17) : Prop :=
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_988_41 backend hashModel tape (path ++ [41, i]) params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_988_48 backend hashModel tape (path ++ [48, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_44_0, (w_45_0, (w_46_0, (w_47_0, ()))))))) next) (Int.toNat (1)) (w_43_0, (w_43_0, (w_43_0, ()))) (w_48_0, (w_48_1, (w_48_2, ()))) ∧
  MxxRuntime.concatColumns w_50_0 w_51_0 w_52_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_59_0 ∧
  MxxRuntime.centeredRebaseRuns w_59_0 w_60_0 ∧
  MxxRuntime.modulusReduceRuns w_60_0 w_61_0 ∧
  MxxRuntime.centeredRebaseRuns w_65_0 w_66_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_68_0 w_69_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_69_0 w_70_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_74_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_75_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_76_0 ∧
  MxxRuntime.centeredRebaseRuns w_76_0 w_77_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_79_0 w_80_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_80_0 w_81_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625

abbrev parallel_generatedRoot_988.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_25_0 : mxxMat7) (w_27_0 : mxxMat0) (w_2_0 : mxxMat14) (w_5_0 : mxxMat3) (w_6_0 : mxxMat5) (w_7_0 : mxxMat5) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat25) (w_16_0 : mxxMat25) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat6) (w_24_0 : mxxMat0) (w_26_0 : mxxMat11) (w_28_0 : mxxMat16) (w_29_0 : mxxMat16) (w_31_0 : mxxMat16) (w_32_0 : mxxMat17) (w_33_0 : mxxMat17) (w_34_0 : mxxMat11) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat12) (w_38_0 : mxxMat12) (w_39_0 : mxxMat1) (w_40_0 : mxxMat0) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_5_0 w_6_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_6_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.centeredRebaseRuns w_10_0 w_11_0 ∧
  MxxRuntime.modulusReduceRuns w_11_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_16_0 w_17_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_19_0 w_20_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_20_0 w_21_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_25_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_26_0 ∧
  MxxRuntime.blockDiagonalRuns 25 w_27_0 w_28_0 ∧
  MxxRuntime.centeredRebaseRuns w_28_0 w_29_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_31_0 w_32_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_32_0 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_34_0 0 1 0 25 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_36_0 w_37_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_37_0 w_38_0 ∧
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0

abbrev parallel_generatedRoot_988.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_988.InputType)) (outputs : mxxMat3) (witness : parallel_generatedRoot_988.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat3 := inputs.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.1
  let w_25_0 : mxxMat7 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_44_0 : mxxMat9 := inputs.2.2.2.2.2.2.1
  let w_45_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.1
  let w_47_0 : mxxMat9 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_58_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_74_0 : mxxMat7 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat1)
  let w_5_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat0 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat26 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat26)
  let w_14_0 : mxxMat25 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat26 := (MxxRuntime.matrixPolynomial [(157994188800)] : mxxMat26)
  let w_16_0 : mxxMat25 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat7)] : mxxMat1)
  let w_19_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat0 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_31_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat11 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat1)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat1 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_43_0 : mxxMat0 := (0 : mxxMat0)
  let w_48_0 := witness.w_48_0
  let w_48_1 := witness.w_48_1
  let w_48_2 := witness.w_48_2
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_42_0 w_48_1
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat1)
  let w_54_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_53_0 w_4_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMul w_52_0 w_56_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat26 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat26)
  let w_63_0 : mxxMat25 := MxxRuntime.matrixMulScalarRight w_61_0 w_62_0
  let w_64_0 : mxxMat26 := (MxxRuntime.matrixPolynomial [(157994188800)] : mxxMat26)
  let w_65_0 : mxxMat25 := MxxRuntime.matrixMulScalarRight w_63_0 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat7)] : mxxMat1)
  let w_68_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_66_0 w_67_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixMul w_58_0 w_70_0
  let w_72_0 : mxxMat0 := MxxRuntime.matrixSub w_57_0 w_71_0
  let w_73_0 : mxxMat0 := (MxxRuntime.unitRow (0) : mxxMat0)
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_79_0 : mxxMat16 := MxxRuntime.matrixMulScalarRight w_77_0 w_78_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat11 := MxxRuntime.matrixMul w_75_0 w_81_0
  let w_83_0 := witness.w_83_0
  let w_84_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat1)
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 : mxxMat1 := MxxRuntime.matrixMul w_83_0 w_86_0
  let w_88_0 := witness.w_88_0
  let w_89_0 := witness.w_89_0
  let w_90_0 : mxxMat0 := MxxRuntime.matrixAdd w_72_0 w_89_0
  let w_91_0 : mxxMat0 := (0 : mxxMat0)
  let w_92_0 := witness.w_92_0
  let w_92_1 := witness.w_92_1
  let w_92_2 := witness.w_92_2
  let w_93_0 : mxxMat0 := MxxRuntime.matrixAdd w_90_0 w_92_2
  let w_94_0 := witness.w_94_0
  parallel_generatedRoot_988.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_24_0 w_26_0 w_28_0 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    parallel_generatedRoot_988.constraints_1 backend hashModel tape path params i_0 w_27_0 w_44_0 w_45_0 w_46_0 w_47_0 w_50_0 w_51_0 w_74_0 w_34_0 w_38_0 w_40_0 w_41_0 w_43_0 w_48_0 w_48_1 w_48_2 w_52_0 w_54_0 w_55_0 w_56_0 w_59_0 w_60_0 w_61_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_75_0 w_76_0 w_77_0 w_79_0 w_80_0 w_81_0 ∧
    parallel_generatedRoot_988.constraints_2 backend hashModel tape path params i_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_73_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_91_0 w_92_0 w_92_1 w_92_2 w_93_0 w_94_0 outputs

def parallel_generatedRoot_988 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (parallel_generatedRoot_988.InputType)) (outputs : mxxMat3) : Prop :=
  ∃ witness : parallel_generatedRoot_988.Witness,
    parallel_generatedRoot_988.body backend hashModel tape path params i_0 inputs outputs witness

abbrev generatedRoot.InputType : Type :=
  (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × ByteArray × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 16 → Int) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 16 → Int) × mxxMat0 × (Fin 2 → mxxMat1) × mxxMat0 × ByteArray × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 16 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × mxxMat0 × (Fin 2 → mxxMat1) × mxxMat0 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 5 → Int) × ByteArray × mxxMat27 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 14400 → Int) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 14400 → Int) × mxxMat0 × (Fin 2 → mxxMat1) × mxxMat0 × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 14400 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 72000 → Int) × (Fin 72000 → Int) × mxxMat0 × mxxMat0 × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × (Fin 72000 → mxxMat0) × Unit

abbrev generatedRoot.OutputType : Type :=
  (Fin 36 → mxxMat3) × (Fin 36 → mxxMat3) × (Fin 36 → mxxMat3) × (Fin 36 → mxxMat3) × (Fin 36 → mxxMat3) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat1 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_17_0 : mxxMat0
  w_18_0 : mxxMat0
  w_21_0 : mxxMat0
  w_22_0 : mxxMat0
  w_27_0 : Fin 16 → mxxMat0
  w_27_1 : Fin 16 → mxxMat0
  w_27_2 : Fin 16 → mxxMat0
  w_27_3 : Fin 16 → mxxMat0
  w_27_4 : Fin 16 → mxxMat0
  w_27_5 : Fin 16 → mxxMat0
  w_28_0 : Fin 8 → mxxMat0
  w_28_1 : Fin 8 → mxxMat0
  w_28_2 : Fin 8 → mxxMat0
  w_28_3 : Fin 8 → mxxMat0
  w_28_4 : Fin 8 → mxxMat0
  w_28_5 : Fin 8 → mxxMat0
  w_31_0 : Fin 16 → mxxMat0
  w_31_1 : Fin 16 → mxxMat0
  w_31_2 : Fin 16 → mxxMat0
  w_31_3 : Fin 16 → mxxMat0
  w_31_4 : Fin 16 → mxxMat0
  w_31_5 : Fin 16 → mxxMat0
  w_32_0 : Fin 8 → mxxMat0
  w_32_1 : Fin 8 → mxxMat0
  w_32_2 : Fin 8 → mxxMat0
  w_32_3 : Fin 8 → mxxMat0
  w_32_4 : Fin 8 → mxxMat0
  w_32_5 : Fin 8 → mxxMat0
  w_33_0 : Fin 8 → mxxMat0
  w_33_1 : Fin 8 → mxxMat0
  w_33_2 : Fin 8 → mxxMat0
  w_33_3 : Fin 8 → mxxMat0
  w_33_4 : Fin 8 → mxxMat0
  w_33_5 : Fin 8 → mxxMat0
  w_34_0 : Fin 8 → mxxMat0
  w_34_1 : Fin 8 → mxxMat0
  w_34_2 : Fin 8 → mxxMat0
  w_34_3 : Fin 8 → mxxMat0
  w_34_4 : Fin 8 → mxxMat0
  w_34_5 : Fin 8 → mxxMat0
  w_35_0 : Fin 8 → mxxMat0
  w_35_1 : Fin 8 → mxxMat0
  w_35_2 : Fin 8 → mxxMat0
  w_35_3 : Fin 8 → mxxMat0
  w_35_4 : Fin 8 → mxxMat0
  w_35_5 : Fin 8 → mxxMat0
  w_36_0 : mxxMat0
  w_37_0 : mxxMat0
  w_39_0 : mxxMat0
  w_40_0 : mxxMat0
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_45_0 : mxxMat0
  w_46_0 : mxxMat0
  w_48_0 : mxxMat0
  w_49_0 : mxxMat0
  w_51_0 : mxxMat0
  w_52_0 : mxxMat0
  w_54_0 : mxxMat0
  w_54_1 : mxxMat0
  w_54_2 : mxxMat0
  w_54_3 : mxxMat0
  w_54_4 : mxxMat0
  w_54_5 : mxxMat0
  w_55_0 : mxxMat0
  w_56_0 : mxxMat0
  w_59_0 : mxxMat6
  w_60_0 : mxxMat6
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_66_0 : mxxMat6
  w_67_0 : mxxMat6
  w_70_0 : mxxMat0
  w_71_0 : mxxMat0
  w_73_0 : mxxMat0
  w_76_0 : mxxMat6
  w_77_0 : mxxMat6
  w_81_0 : mxxMat0
  w_84_0 : mxxMat6
  w_85_0 : mxxMat6
  w_87_0 : mxxMat0
  w_92_0 : mxxMat0
  w_94_0 : mxxMat0
  w_96_0 : mxxMat0
  w_98_0 : mxxMat0
  w_102_0 : mxxMat0
  w_103_0 : mxxMat0
  w_104_0 : mxxMat0
  w_105_0 : mxxMat2
  w_105_concat_1 : mxxMat3
  w_107_0 : mxxMat0
  w_108_0 : mxxMat0
  w_109_0 : mxxMat0
  w_110_0 : mxxMat2
  w_110_concat_1 : mxxMat3
  w_111_0 : mxxMat4
  w_112_0 : mxxMat0
  w_113_0 : mxxMat0
  w_114_0 : mxxMat3
  w_115_0 : mxxMat5
  w_116_0 : mxxMat5
  w_118_0 : mxxMat0
  w_121_0 : mxxMat1
  w_124_0 : mxxMat0
  w_125_0 : mxxMat0
  w_126_0 : mxxMat2
  w_126_concat_1 : mxxMat3
  w_127_0 : mxxMat0
  w_128_0 : mxxMat0
  w_129_0 : mxxMat3
  w_130_0 : mxxMat5
  w_131_0 : mxxMat5
  w_133_0 : mxxMat0
  w_137_0 : mxxMat0
  w_138_0 : mxxMat0
  w_139_0 : mxxMat2
  w_139_concat_1 : mxxMat3
  w_140_0 : mxxMat4
  w_141_0 : mxxMat0
  w_143_0 : mxxMat0
  w_144_0 : mxxMat3
  w_145_0 : mxxMat5
  w_146_0 : mxxMat5
  w_148_0 : mxxMat0
  w_149_0 : mxxMat0
  w_150_0 : mxxMat0
  w_151_0 : mxxMat0
  w_153_0 : mxxMat0
  w_154_0 : mxxMat3
  w_155_0 : mxxMat5
  w_156_0 : mxxMat5
  w_158_0 : mxxMat0
  w_159_0 : mxxMat0
  w_160_0 : mxxMat0
  w_161_0 : mxxMat0
  w_161_1 : mxxMat0
  w_161_2 : mxxMat0
  w_161_3 : mxxMat0
  w_161_4 : mxxMat0
  w_161_5 : mxxMat0
  w_161_6 : mxxMat0
  w_161_7 : mxxMat0
  w_161_8 : mxxMat0
  w_161_9 : mxxMat0
  w_162_0 : mxxMat0
  w_164_0 : mxxMat0
  w_165_0 : mxxMat0
  w_166_0 : mxxMat2
  w_166_concat_1 : mxxMat3
  w_167_0 : mxxMat0
  w_168_0 : mxxMat0
  w_169_0 : mxxMat0
  w_170_0 : mxxMat2
  w_170_concat_1 : mxxMat3
  w_171_0 : mxxMat4
  w_172_0 : mxxMat0
  w_173_0 : mxxMat0
  w_174_0 : mxxMat0
  w_175_0 : mxxMat2
  w_175_concat_1 : mxxMat3
  w_176_0 : mxxMat0
  w_177_0 : mxxMat0
  w_178_0 : mxxMat0
  w_179_0 : mxxMat2
  w_179_concat_1 : mxxMat3
  w_180_0 : mxxMat4
  w_181_0 : mxxMat3
  w_182_0 : mxxMat5
  w_183_0 : mxxMat5
  w_185_0 : mxxMat0
  w_186_0 : mxxMat0
  w_187_0 : mxxMat3
  w_188_0 : mxxMat5
  w_189_0 : mxxMat5
  w_191_0 : mxxMat0
  w_192_0 : mxxMat0
  w_195_0 : mxxMat0
  w_198_0 : mxxMat6
  w_199_0 : mxxMat6
  w_201_0 : mxxMat3
  w_202_0 : mxxMat5
  w_203_0 : mxxMat5
  w_205_0 : mxxMat0
  w_206_0 : mxxMat0
  w_207_0 : mxxMat3
  w_208_0 : mxxMat5
  w_209_0 : mxxMat5
  w_211_0 : mxxMat0
  w_212_0 : mxxMat0
  w_216_0 : mxxMat0
  w_219_0 : mxxMat6
  w_220_0 : mxxMat6
  w_222_0 : mxxMat0
  w_225_0 : mxxMat6
  w_226_0 : mxxMat6
  w_231_0 : mxxMat0
  w_233_0 : mxxMat0
  w_235_0 : mxxMat0
  w_237_0 : mxxMat0
  w_240_0 : mxxMat0
  w_240_1 : mxxMat0
  w_240_2 : mxxMat0
  w_240_3 : mxxMat0
  w_240_4 : mxxMat0
  w_240_5 : mxxMat0
  w_240_6 : mxxMat0
  w_240_7 : mxxMat0
  w_240_8 : mxxMat0
  w_240_9 : mxxMat0
  w_241_0 : mxxMat0
  w_243_0 : mxxMat0
  w_244_0 : mxxMat0
  w_245_0 : mxxMat2
  w_245_concat_1 : mxxMat3
  w_246_0 : mxxMat0
  w_247_0 : mxxMat0
  w_248_0 : mxxMat0
  w_249_0 : mxxMat2
  w_249_concat_1 : mxxMat3
  w_250_0 : mxxMat4
  w_251_0 : mxxMat0
  w_252_0 : mxxMat0
  w_253_0 : mxxMat0
  w_254_0 : mxxMat2
  w_254_concat_1 : mxxMat3
  w_255_0 : mxxMat0
  w_256_0 : mxxMat0
  w_257_0 : mxxMat0
  w_258_0 : mxxMat2
  w_258_concat_1 : mxxMat3
  w_259_0 : mxxMat4
  w_260_0 : mxxMat3
  w_261_0 : mxxMat5
  w_262_0 : mxxMat5
  w_264_0 : mxxMat0
  w_265_0 : mxxMat0
  w_266_0 : mxxMat3
  w_267_0 : mxxMat5
  w_268_0 : mxxMat5
  w_270_0 : mxxMat0
  w_271_0 : mxxMat0
  w_274_0 : mxxMat0
  w_277_0 : mxxMat6
  w_278_0 : mxxMat6
  w_280_0 : mxxMat0
  w_283_0 : mxxMat6
  w_284_0 : mxxMat6
  w_289_0 : mxxMat0
  w_291_0 : mxxMat0
  w_293_0 : mxxMat0
  w_295_0 : mxxMat0
  w_298_0 : mxxMat0
  w_298_1 : mxxMat0
  w_298_2 : mxxMat0
  w_298_3 : mxxMat0
  w_298_4 : mxxMat0
  w_298_5 : mxxMat0
  w_298_6 : mxxMat0
  w_298_7 : mxxMat0
  w_298_8 : mxxMat0
  w_298_9 : mxxMat0
  w_299_0 : mxxMat0
  w_301_0 : mxxMat0
  w_302_0 : mxxMat0
  w_303_0 : mxxMat2
  w_303_concat_1 : mxxMat3
  w_304_0 : mxxMat0
  w_305_0 : mxxMat0
  w_306_0 : mxxMat0
  w_307_0 : mxxMat2
  w_307_concat_1 : mxxMat3
  w_308_0 : mxxMat4
  w_309_0 : mxxMat0
  w_310_0 : mxxMat0
  w_311_0 : mxxMat0
  w_312_0 : mxxMat2
  w_312_concat_1 : mxxMat3
  w_313_0 : mxxMat0
  w_314_0 : mxxMat0
  w_315_0 : mxxMat0
  w_316_0 : mxxMat2
  w_316_concat_1 : mxxMat3
  w_317_0 : mxxMat4
  w_318_0 : mxxMat3
  w_319_0 : mxxMat5
  w_320_0 : mxxMat5
  w_322_0 : mxxMat0
  w_323_0 : mxxMat0
  w_324_0 : mxxMat3
  w_325_0 : mxxMat5
  w_326_0 : mxxMat5
  w_328_0 : mxxMat0
  w_329_0 : mxxMat0
  w_332_0 : mxxMat0
  w_335_0 : mxxMat6
  w_336_0 : mxxMat6
  w_338_0 : mxxMat0
  w_341_0 : mxxMat6
  w_342_0 : mxxMat6
  w_347_0 : mxxMat0
  w_349_0 : mxxMat0
  w_351_0 : mxxMat0
  w_353_0 : mxxMat0
  w_356_0 : mxxMat0
  w_356_1 : mxxMat0
  w_356_2 : mxxMat0
  w_356_3 : mxxMat0
  w_356_4 : mxxMat0
  w_356_5 : mxxMat0
  w_356_6 : mxxMat0
  w_356_7 : mxxMat0
  w_356_8 : mxxMat0
  w_356_9 : mxxMat0
  w_357_0 : mxxMat0
  w_359_0 : mxxMat0
  w_360_0 : mxxMat0
  w_361_0 : mxxMat2
  w_361_concat_1 : mxxMat3
  w_362_0 : mxxMat0
  w_363_0 : mxxMat0
  w_364_0 : mxxMat0
  w_365_0 : mxxMat2
  w_365_concat_1 : mxxMat3
  w_366_0 : mxxMat4
  w_367_0 : mxxMat0
  w_368_0 : mxxMat0
  w_369_0 : mxxMat0
  w_370_0 : mxxMat2
  w_370_concat_1 : mxxMat3
  w_371_0 : mxxMat0
  w_372_0 : mxxMat0
  w_373_0 : mxxMat0
  w_374_0 : mxxMat2
  w_374_concat_1 : mxxMat3
  w_375_0 : mxxMat4
  w_376_0 : mxxMat3
  w_377_0 : mxxMat5
  w_378_0 : mxxMat5
  w_380_0 : mxxMat0
  w_381_0 : mxxMat0
  w_382_0 : mxxMat3
  w_383_0 : mxxMat5
  w_384_0 : mxxMat5
  w_386_0 : mxxMat0
  w_387_0 : mxxMat0
  w_390_0 : mxxMat0
  w_393_0 : mxxMat6
  w_394_0 : mxxMat6
  w_396_0 : mxxMat0
  w_399_0 : mxxMat6
  w_400_0 : mxxMat6
  w_405_0 : mxxMat0
  w_407_0 : mxxMat0
  w_409_0 : mxxMat0
  w_411_0 : mxxMat0
  w_414_0 : mxxMat0
  w_414_1 : mxxMat0
  w_414_2 : mxxMat0
  w_414_3 : mxxMat0
  w_414_4 : mxxMat0
  w_414_5 : mxxMat0
  w_414_6 : mxxMat0
  w_414_7 : mxxMat0
  w_414_8 : mxxMat0
  w_414_9 : mxxMat0
  w_415_0 : mxxMat0
  w_417_0 : mxxMat0
  w_418_0 : mxxMat0
  w_419_0 : mxxMat2
  w_419_concat_1 : mxxMat3
  w_420_0 : mxxMat0
  w_421_0 : mxxMat0
  w_422_0 : mxxMat0
  w_423_0 : mxxMat2
  w_423_concat_1 : mxxMat3
  w_424_0 : mxxMat4
  w_425_0 : mxxMat0
  w_426_0 : mxxMat0
  w_427_0 : mxxMat0
  w_428_0 : mxxMat2
  w_428_concat_1 : mxxMat3
  w_429_0 : mxxMat0
  w_430_0 : mxxMat0
  w_431_0 : mxxMat0
  w_432_0 : mxxMat2
  w_432_concat_1 : mxxMat3
  w_433_0 : mxxMat4
  w_434_0 : mxxMat3
  w_435_0 : mxxMat5
  w_436_0 : mxxMat5
  w_438_0 : mxxMat0
  w_439_0 : mxxMat0
  w_440_0 : mxxMat3
  w_441_0 : mxxMat5
  w_442_0 : mxxMat5
  w_444_0 : mxxMat0
  w_445_0 : mxxMat0
  w_448_0 : mxxMat0
  w_451_0 : mxxMat6
  w_452_0 : mxxMat6
  w_454_0 : mxxMat0
  w_457_0 : mxxMat6
  w_458_0 : mxxMat6
  w_463_0 : mxxMat0
  w_465_0 : mxxMat0
  w_467_0 : mxxMat0
  w_469_0 : mxxMat0
  w_472_0 : mxxMat0
  w_472_1 : mxxMat0
  w_472_2 : mxxMat0
  w_472_3 : mxxMat0
  w_472_4 : mxxMat0
  w_472_5 : mxxMat0
  w_472_6 : mxxMat0
  w_472_7 : mxxMat0
  w_472_8 : mxxMat0
  w_472_9 : mxxMat0
  w_473_0 : mxxMat0
  w_475_0 : mxxMat0
  w_476_0 : mxxMat0
  w_477_0 : mxxMat2
  w_477_concat_1 : mxxMat3
  w_478_0 : mxxMat0
  w_479_0 : mxxMat0
  w_480_0 : mxxMat0
  w_481_0 : mxxMat2
  w_481_concat_1 : mxxMat3
  w_482_0 : mxxMat4
  w_483_0 : mxxMat0
  w_484_0 : mxxMat0
  w_485_0 : mxxMat0
  w_486_0 : mxxMat2
  w_486_concat_1 : mxxMat3
  w_487_0 : mxxMat0
  w_488_0 : mxxMat0
  w_489_0 : mxxMat0
  w_490_0 : mxxMat2
  w_490_concat_1 : mxxMat3
  w_491_0 : mxxMat4
  w_492_0 : mxxMat3
  w_493_0 : mxxMat5
  w_494_0 : mxxMat5
  w_496_0 : mxxMat0
  w_497_0 : mxxMat0
  w_498_0 : mxxMat3
  w_499_0 : mxxMat5
  w_500_0 : mxxMat5
  w_502_0 : mxxMat0
  w_503_0 : mxxMat0
  w_506_0 : mxxMat0
  w_509_0 : mxxMat6
  w_510_0 : mxxMat6
  w_512_0 : mxxMat0
  w_515_0 : mxxMat6
  w_516_0 : mxxMat6
  w_521_0 : mxxMat0
  w_523_0 : mxxMat0
  w_525_0 : mxxMat0
  w_527_0 : mxxMat0
  w_530_0 : mxxMat0
  w_530_1 : mxxMat0
  w_530_2 : mxxMat0
  w_530_3 : mxxMat0
  w_530_4 : mxxMat0
  w_530_5 : mxxMat0
  w_530_6 : mxxMat0
  w_530_7 : mxxMat0
  w_530_8 : mxxMat0
  w_530_9 : mxxMat0
  w_531_0 : mxxMat0
  w_533_0 : mxxMat0
  w_534_0 : mxxMat0
  w_535_0 : mxxMat2
  w_535_concat_1 : mxxMat3
  w_536_0 : mxxMat0
  w_537_0 : mxxMat0
  w_538_0 : mxxMat0
  w_539_0 : mxxMat2
  w_539_concat_1 : mxxMat3
  w_540_0 : mxxMat4
  w_541_0 : mxxMat0
  w_542_0 : mxxMat0
  w_543_0 : mxxMat0
  w_544_0 : mxxMat2
  w_544_concat_1 : mxxMat3
  w_545_0 : mxxMat0
  w_546_0 : mxxMat0
  w_547_0 : mxxMat0
  w_548_0 : mxxMat2
  w_548_concat_1 : mxxMat3
  w_549_0 : mxxMat4
  w_550_0 : mxxMat3
  w_551_0 : mxxMat5
  w_552_0 : mxxMat5
  w_554_0 : mxxMat0
  w_555_0 : mxxMat0
  w_556_0 : mxxMat3
  w_557_0 : mxxMat5
  w_558_0 : mxxMat5
  w_560_0 : mxxMat0
  w_561_0 : mxxMat0
  w_564_0 : mxxMat0
  w_567_0 : mxxMat6
  w_568_0 : mxxMat6
  w_570_0 : mxxMat0
  w_573_0 : mxxMat6
  w_574_0 : mxxMat6
  w_579_0 : mxxMat0
  w_581_0 : mxxMat0
  w_583_0 : mxxMat0
  w_585_0 : mxxMat0
  w_588_0 : mxxMat0
  w_588_1 : mxxMat0
  w_588_2 : mxxMat0
  w_588_3 : mxxMat0
  w_588_4 : mxxMat0
  w_588_5 : mxxMat0
  w_588_6 : mxxMat0
  w_588_7 : mxxMat0
  w_588_8 : mxxMat0
  w_588_9 : mxxMat0
  w_589_0 : mxxMat0
  w_591_0 : mxxMat0
  w_592_0 : mxxMat0
  w_593_0 : mxxMat2
  w_593_concat_1 : mxxMat3
  w_594_0 : mxxMat0
  w_595_0 : mxxMat0
  w_596_0 : mxxMat0
  w_597_0 : mxxMat2
  w_597_concat_1 : mxxMat3
  w_598_0 : mxxMat4
  w_599_0 : mxxMat0
  w_600_0 : mxxMat0
  w_601_0 : mxxMat0
  w_602_0 : mxxMat2
  w_602_concat_1 : mxxMat3
  w_603_0 : mxxMat0
  w_604_0 : mxxMat0
  w_605_0 : mxxMat0
  w_606_0 : mxxMat2
  w_606_concat_1 : mxxMat3
  w_607_0 : mxxMat4
  w_608_0 : mxxMat3
  w_609_0 : mxxMat5
  w_610_0 : mxxMat5
  w_612_0 : mxxMat0
  w_613_0 : mxxMat0
  w_614_0 : mxxMat3
  w_615_0 : mxxMat5
  w_616_0 : mxxMat5
  w_618_0 : mxxMat0
  w_619_0 : mxxMat0
  w_622_0 : mxxMat0
  w_625_0 : mxxMat6
  w_626_0 : mxxMat6
  w_628_0 : mxxMat0
  w_631_0 : mxxMat6
  w_632_0 : mxxMat6
  w_636_0 : Fin 26 → mxxMat0
  w_637_0 : Fin 5 → Int
  w_637_1 : Fin 5 → Int
  w_639_0 : mxxMat0
  w_642_0 : mxxMat0
  w_644_0 : mxxMat0
  w_645_0 : mxxMat0
  w_646_0 : mxxMat0
  w_647_0 : mxxMat0
  w_650_0 : mxxMat0
  w_655_0 : mxxMat0
  w_658_0 : Fin 5 → mxxMat0
  w_658_1 : Fin 5 → mxxMat0
  w_658_2 : Fin 5 → mxxMat0
  w_658_3 : Fin 5 → mxxMat0
  w_658_4 : Fin 5 → mxxMat0
  w_658_5 : Fin 5 → mxxMat0
  w_661_0 : mxxMat0
  w_664_0 : mxxMat0
  w_670_0 : mxxMat0
  w_675_0 : mxxMat0
  w_680_0 : mxxMat0
  w_685_0 : mxxMat0
  w_690_0 : mxxMat0
  w_695_0 : mxxMat0
  w_700_0 : mxxMat0
  w_705_0 : Fin 26 → mxxMat0
  w_708_0 : mxxMat0
  w_711_0 : mxxMat0
  w_717_0 : mxxMat0
  w_722_0 : mxxMat0
  w_727_0 : mxxMat0
  w_732_0 : mxxMat0
  w_737_0 : mxxMat0
  w_742_0 : mxxMat0
  w_747_0 : mxxMat0
  w_752_0 : Fin 26 → mxxMat0
  w_754_0 : mxxMat0
  w_755_0 : mxxMat0
  w_756_0 : mxxMat3
  w_757_0 : mxxMat3
  w_758_0 : mxxMat0
  w_759_0 : mxxMat0
  w_760_0 : mxxMat3
  w_761_0 : mxxMat3
  w_762_0 : mxxMat0
  w_763_0 : mxxMat0
  w_764_0 : mxxMat3
  w_765_0 : mxxMat3
  w_766_0 : mxxMat0
  w_767_0 : mxxMat0
  w_768_0 : mxxMat3
  w_769_0 : mxxMat3
  w_770_0 : mxxMat0
  w_771_0 : mxxMat0
  w_772_0 : mxxMat3
  w_773_0 : mxxMat3
  w_774_0 : mxxMat0
  w_775_0 : mxxMat0
  w_776_0 : mxxMat3
  w_777_0 : mxxMat3
  w_779_0 : mxxMat0
  w_780_0 : mxxMat0
  w_781_0 : mxxMat3
  w_782_0 : mxxMat3
  w_784_0 : mxxMat0
  w_785_0 : mxxMat0
  w_786_0 : mxxMat3
  w_787_0 : mxxMat3
  w_789_0 : mxxMat0
  w_790_0 : mxxMat0
  w_791_0 : mxxMat3
  w_792_0 : mxxMat3
  w_794_0 : mxxMat0
  w_795_0 : mxxMat0
  w_796_0 : mxxMat3
  w_797_0 : mxxMat3
  w_799_0 : mxxMat0
  w_800_0 : mxxMat0
  w_801_0 : mxxMat3
  w_802_0 : mxxMat3
  w_804_0 : mxxMat0
  w_805_0 : mxxMat0
  w_806_0 : mxxMat3
  w_807_0 : mxxMat3
  w_809_0 : mxxMat0
  w_810_0 : mxxMat0
  w_811_0 : mxxMat3
  w_812_0 : mxxMat3
  w_814_0 : mxxMat0
  w_815_0 : mxxMat0
  w_816_0 : mxxMat3
  w_817_0 : mxxMat3
  w_819_0 : mxxMat0
  w_820_0 : mxxMat0
  w_821_0 : mxxMat3
  w_822_0 : mxxMat3
  w_824_0 : mxxMat0
  w_825_0 : mxxMat0
  w_826_0 : mxxMat3
  w_827_0 : mxxMat3
  w_829_0 : mxxMat0
  w_830_0 : mxxMat0
  w_831_0 : mxxMat3
  w_832_0 : mxxMat3
  w_834_0 : mxxMat0
  w_835_0 : mxxMat0
  w_836_0 : mxxMat3
  w_837_0 : mxxMat3
  w_839_0 : mxxMat0
  w_840_0 : mxxMat0
  w_841_0 : mxxMat3
  w_842_0 : mxxMat3
  w_844_0 : mxxMat0
  w_845_0 : mxxMat0
  w_846_0 : mxxMat3
  w_847_0 : mxxMat3
  w_849_0 : mxxMat0
  w_850_0 : mxxMat0
  w_851_0 : mxxMat3
  w_852_0 : mxxMat3
  w_854_0 : mxxMat0
  w_855_0 : mxxMat0
  w_856_0 : mxxMat3
  w_857_0 : mxxMat3
  w_859_0 : mxxMat28
  w_860_0 : mxxMat29
  w_861_0 : mxxMat3
  w_862_0 : mxxMat3
  w_863_0 : mxxMat3
  w_864_0 : mxxMat3
  w_865_0 : mxxMat3
  w_866_0 : mxxMat3
  w_867_0 : mxxMat3
  w_868_0 : mxxMat3
  w_869_0 : Fin 26 → mxxMat3
  w_870_0 : Fin 36 → mxxMat0
  w_870_1 : Fin 36 → mxxMat0
  w_870_2 : Fin 36 → mxxMat0
  w_870_3 : Fin 36 → mxxMat3
  w_880_0 : mxxMat0
  w_881_0 : mxxMat0
  w_884_0 : mxxMat0
  w_885_0 : mxxMat0
  w_889_0 : Fin 14400 → mxxMat0
  w_889_1 : Fin 14400 → mxxMat0
  w_889_2 : Fin 14400 → mxxMat0
  w_889_3 : Fin 14400 → mxxMat0
  w_889_4 : Fin 14400 → mxxMat0
  w_889_5 : Fin 14400 → mxxMat0
  w_890_0 : Fin 7200 → mxxMat0
  w_890_1 : Fin 7200 → mxxMat0
  w_890_2 : Fin 7200 → mxxMat0
  w_890_3 : Fin 7200 → mxxMat0
  w_890_4 : Fin 7200 → mxxMat0
  w_890_5 : Fin 7200 → mxxMat0
  w_893_0 : Fin 14400 → mxxMat0
  w_893_1 : Fin 14400 → mxxMat0
  w_893_2 : Fin 14400 → mxxMat0
  w_893_3 : Fin 14400 → mxxMat0
  w_893_4 : Fin 14400 → mxxMat0
  w_893_5 : Fin 14400 → mxxMat0
  w_894_0 : Fin 7200 → mxxMat0
  w_894_1 : Fin 7200 → mxxMat0
  w_894_2 : Fin 7200 → mxxMat0
  w_894_3 : Fin 7200 → mxxMat0
  w_894_4 : Fin 7200 → mxxMat0
  w_894_5 : Fin 7200 → mxxMat0
  w_895_0 : Fin 7200 → mxxMat0
  w_895_1 : Fin 7200 → mxxMat0
  w_895_2 : Fin 7200 → mxxMat0
  w_895_3 : Fin 7200 → mxxMat0
  w_895_4 : Fin 7200 → mxxMat0
  w_895_5 : Fin 7200 → mxxMat0
  w_896_0 : Fin 7200 → mxxMat0
  w_896_1 : Fin 7200 → mxxMat0
  w_896_2 : Fin 7200 → mxxMat0
  w_896_3 : Fin 7200 → mxxMat0
  w_896_4 : Fin 7200 → mxxMat0
  w_896_5 : Fin 7200 → mxxMat0
  w_897_0 : mxxMat0
  w_898_0 : mxxMat0
  w_900_0 : mxxMat0
  w_901_0 : mxxMat0
  w_903_0 : mxxMat0
  w_904_0 : mxxMat0
  w_906_0 : mxxMat0
  w_907_0 : mxxMat0
  w_909_0 : mxxMat0
  w_910_0 : mxxMat0
  w_912_0 : mxxMat0
  w_913_0 : mxxMat0
  w_915_0 : mxxMat0
  w_915_1 : mxxMat0
  w_915_2 : mxxMat0
  w_915_3 : mxxMat0
  w_915_4 : mxxMat0
  w_915_5 : mxxMat0
  w_916_0 : mxxMat7
  w_918_0 : mxxMat7
  w_920_0 : mxxMat7
  w_922_0 : mxxMat7
  w_924_0 : mxxMat7
  w_926_0 : mxxMat7
  w_927_0 : mxxMat7
  w_927_1 : mxxMat7
  w_927_2 : mxxMat7
  w_927_3 : mxxMat7
  w_927_4 : mxxMat7
  w_927_5 : mxxMat7
  w_935_0 : mxxMat0
  w_936_0 : mxxMat0
  w_938_0 : mxxMat0
  w_939_0 : mxxMat0
  w_943_0 : Fin 72000 → mxxMat0
  w_943_1 : Fin 72000 → mxxMat0
  w_943_2 : Fin 72000 → mxxMat0
  w_943_3 : Fin 72000 → mxxMat0
  w_943_4 : Fin 72000 → mxxMat0
  w_943_5 : Fin 72000 → mxxMat0
  w_944_0 : Fin 36000 → mxxMat0
  w_944_1 : Fin 36000 → mxxMat0
  w_944_2 : Fin 36000 → mxxMat0
  w_944_3 : Fin 36000 → mxxMat0
  w_944_4 : Fin 36000 → mxxMat0
  w_944_5 : Fin 36000 → mxxMat0
  w_947_0 : Fin 72000 → mxxMat0
  w_947_1 : Fin 72000 → mxxMat0
  w_947_2 : Fin 72000 → mxxMat0
  w_947_3 : Fin 72000 → mxxMat0
  w_947_4 : Fin 72000 → mxxMat0
  w_947_5 : Fin 72000 → mxxMat0
  w_948_0 : Fin 36000 → mxxMat0
  w_948_1 : Fin 36000 → mxxMat0
  w_948_2 : Fin 36000 → mxxMat0
  w_948_3 : Fin 36000 → mxxMat0
  w_948_4 : Fin 36000 → mxxMat0
  w_948_5 : Fin 36000 → mxxMat0
  w_949_0 : Fin 36000 → mxxMat0
  w_949_1 : Fin 36000 → mxxMat0
  w_949_2 : Fin 36000 → mxxMat0
  w_949_3 : Fin 36000 → mxxMat0
  w_949_4 : Fin 36000 → mxxMat0
  w_949_5 : Fin 36000 → mxxMat0
  w_950_0 : Fin 36000 → mxxMat0
  w_950_1 : Fin 36000 → mxxMat0
  w_950_2 : Fin 36000 → mxxMat0
  w_950_3 : Fin 36000 → mxxMat0
  w_950_4 : Fin 36000 → mxxMat0
  w_950_5 : Fin 36000 → mxxMat0
  w_951_0 : mxxMat0
  w_952_0 : mxxMat0
  w_954_0 : mxxMat0
  w_955_0 : mxxMat0
  w_957_0 : mxxMat0
  w_958_0 : mxxMat0
  w_960_0 : mxxMat0
  w_961_0 : mxxMat0
  w_963_0 : mxxMat0
  w_964_0 : mxxMat0
  w_966_0 : mxxMat0
  w_967_0 : mxxMat0
  w_969_0 : mxxMat0
  w_969_1 : mxxMat0
  w_969_2 : mxxMat0
  w_969_3 : mxxMat0
  w_969_4 : mxxMat0
  w_969_5 : mxxMat0
  w_970_0 : mxxMat9
  w_972_0 : mxxMat9
  w_974_0 : mxxMat9
  w_976_0 : mxxMat9
  w_978_0 : mxxMat9
  w_980_0 : mxxMat9
  w_981_0 : mxxMat9
  w_981_1 : mxxMat9
  w_981_2 : mxxMat9
  w_981_3 : mxxMat9
  w_981_4 : mxxMat9
  w_981_5 : mxxMat9
  w_983_0 : Fin 1 → mxxMat1
  w_984_0 : Fin 36 → mxxMat3
  w_985_0 : Fin 36 → mxxMat3
  w_986_0 : Fin 36 → mxxMat3
  w_987_0 : Fin 36 → mxxMat3
  w_988_0 : Fin 36 → mxxMat3
  w_989_0 : mxxMat3
  w_990_0 : mxxMat5
  w_991_0 : mxxMat5
  w_993_0 : mxxMat0
  w_994_0 : mxxMat0
  w_995_0 : mxxMat3
  w_996_0 : mxxMat5
  w_997_0 : mxxMat5
  w_999_0 : mxxMat0
  w_1001_0 : mxxMat0
  w_1003_0 : mxxMat0
  w_1004_0 : mxxMat0
  w_1007_0 : mxxMat3
  w_1008_0 : mxxMat5
  w_1009_0 : mxxMat5
  w_1011_0 : mxxMat0
  w_1012_0 : mxxMat0
  w_1013_0 : mxxMat3
  w_1014_0 : mxxMat5
  w_1015_0 : mxxMat5
  w_1017_0 : mxxMat0
  w_1019_0 : mxxMat0
  w_1021_0 : mxxMat0
  w_1022_0 : mxxMat0
  w_1025_0 : mxxMat0
  w_1026_0 : mxxMat0
  w_1027_0 : mxxMat0
  w_1028_0 : mxxMat1
  w_1029_0 : mxxMat3
  w_1030_0 : mxxMat5
  w_1031_0 : mxxMat5
  w_1033_0 : mxxMat0
  w_1034_0 : mxxMat0
  w_1035_0 : mxxMat3
  w_1036_0 : mxxMat5
  w_1037_0 : mxxMat5
  w_1039_0 : mxxMat0
  w_1041_0 : mxxMat0
  w_1043_0 : mxxMat0
  w_1044_0 : mxxMat0
  w_1047_0 : mxxMat3
  w_1048_0 : mxxMat5
  w_1049_0 : mxxMat5
  w_1051_0 : mxxMat0
  w_1052_0 : mxxMat0
  w_1053_0 : mxxMat3
  w_1054_0 : mxxMat5
  w_1055_0 : mxxMat5
  w_1057_0 : mxxMat0
  w_1059_0 : mxxMat0
  w_1061_0 : mxxMat0
  w_1062_0 : mxxMat0
  w_1065_0 : mxxMat3
  w_1066_0 : mxxMat5
  w_1067_0 : mxxMat5
  w_1069_0 : mxxMat0
  w_1070_0 : mxxMat0
  w_1071_0 : mxxMat3
  w_1072_0 : mxxMat5
  w_1073_0 : mxxMat5
  w_1075_0 : mxxMat0
  w_1077_0 : mxxMat0
  w_1079_0 : mxxMat0
  w_1080_0 : mxxMat0
  w_1083_0 : mxxMat3
  w_1084_0 : mxxMat5
  w_1085_0 : mxxMat5
  w_1087_0 : mxxMat0
  w_1088_0 : mxxMat0
  w_1089_0 : mxxMat3
  w_1090_0 : mxxMat5
  w_1091_0 : mxxMat5
  w_1093_0 : mxxMat0
  w_1095_0 : mxxMat0
  w_1097_0 : mxxMat0
  w_1098_0 : mxxMat0
  w_1101_0 : mxxMat3
  w_1102_0 : mxxMat5
  w_1103_0 : mxxMat5
  w_1105_0 : mxxMat0
  w_1106_0 : mxxMat0
  w_1107_0 : mxxMat3
  w_1108_0 : mxxMat5
  w_1109_0 : mxxMat5
  w_1111_0 : mxxMat0
  w_1113_0 : mxxMat0
  w_1115_0 : mxxMat0
  w_1116_0 : mxxMat0

abbrev generatedRoot.constraints_49 (w_19_0 : mxxMat0) (w_23_0 : mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_62_0 : mxxMat0) (w_87_0 : mxxMat0) (w_194_0 : mxxMat0) (w_214_0 : mxxMat0) (w_273_0 : mxxMat0) (w_331_0 : mxxMat0) (w_389_0 : mxxMat0) (w_447_0 : mxxMat0) (w_505_0 : mxxMat0) (w_563_0 : mxxMat0) (w_621_0 : mxxMat0) (w_655_0 : mxxMat0) (w_662_0 : mxxMat0) (w_665_0 : mxxMat0) (w_671_0 : mxxMat0) (w_676_0 : mxxMat0) (w_681_0 : mxxMat0) (w_686_0 : mxxMat0) (w_691_0 : mxxMat0) (w_696_0 : mxxMat0) (w_701_0 : mxxMat0) (w_709_0 : mxxMat0) (w_712_0 : mxxMat0) (w_718_0 : mxxMat0) (w_723_0 : mxxMat0) (w_728_0 : mxxMat0) (w_733_0 : mxxMat0) (w_738_0 : mxxMat0) (w_743_0 : mxxMat0) (w_748_0 : mxxMat0) (w_927_0 : mxxMat7) (w_927_1 : mxxMat7) (w_927_2 : mxxMat7) (w_927_3 : mxxMat7) (w_927_4 : mxxMat7) (w_927_5 : mxxMat7) (w_981_0 : mxxMat9) (w_981_1 : mxxMat9) (w_981_2 : mxxMat9) (w_981_3 : mxxMat9) (w_981_4 : mxxMat9) (w_981_5 : mxxMat9) (w_984_0 : Fin 36 → mxxMat3) (w_985_0 : Fin 36 → mxxMat3) (w_986_0 : Fin 36 → mxxMat3) (w_987_0 : Fin 36 → mxxMat3) (w_988_0 : Fin 36 → mxxMat3) (w_1000_0 : mxxMat0) (w_1002_0 : mxxMat0) (w_1006_0 : mxxMat0) (w_1018_0 : mxxMat0) (w_1020_0 : mxxMat0) (w_1024_0 : mxxMat0) (w_1025_0 : mxxMat0) (w_1026_0 : mxxMat0) (w_1027_0 : mxxMat0) (w_1028_0 : mxxMat1) (w_1040_0 : mxxMat0) (w_1042_0 : mxxMat0) (w_1046_0 : mxxMat0) (w_1058_0 : mxxMat0) (w_1060_0 : mxxMat0) (w_1064_0 : mxxMat0) (w_1076_0 : mxxMat0) (w_1078_0 : mxxMat0) (w_1082_0 : mxxMat0) (w_1094_0 : mxxMat0) (w_1096_0 : mxxMat0) (w_1100_0 : mxxMat0) (w_1104_0 : mxxMat2) (w_1110_0 : mxxMat2) (w_1112_0 : mxxMat0) (w_1114_0 : mxxMat0) (w_1115_0 : mxxMat0) (w_1116_0 : mxxMat0) (w_1118_0 : mxxMat0) (outputs : (generatedRoot.OutputType)) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1110_0 0 1 0 25 w_1115_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1104_0 0 1 0 25 w_1116_0 ∧
  outputs = (w_984_0, (w_985_0, (w_986_0, (w_987_0, (w_988_0, (w_662_0, (w_665_0, (w_709_0, (w_712_0, (w_194_0, (w_214_0, (w_981_0, (w_981_1, (w_981_2, (w_981_3, (w_981_4, (w_981_5, (w_1000_0, (w_696_0, (w_1002_0, (w_743_0, (w_1006_0, (w_563_0, (w_1018_0, (w_701_0, (w_1020_0, (w_748_0, (w_1024_0, (w_621_0, (w_927_0, (w_927_1, (w_927_2, (w_927_3, (w_927_4, (w_927_5, (w_17_0, (w_21_0, (w_18_0, (w_22_0, (w_19_0, (w_23_0, (w_1025_0, (w_62_0, (w_1026_0, (w_655_0, (w_1027_0, (w_87_0, (w_1028_0, (w_54_0, (w_54_1, (w_54_2, (w_54_3, (w_54_4, (w_54_5, (w_1040_0, (w_671_0, (w_1042_0, (w_718_0, (w_1046_0, (w_273_0, (w_1058_0, (w_676_0, (w_1060_0, (w_723_0, (w_1064_0, (w_331_0, (w_1076_0, (w_681_0, (w_1078_0, (w_728_0, (w_1082_0, (w_389_0, (w_1094_0, (w_686_0, (w_1096_0, (w_733_0, (w_1100_0, (w_447_0, (w_1112_0, (w_691_0, (w_1114_0, (w_738_0, (w_1118_0, (w_505_0, ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

abbrev generatedRoot.constraints_48 (backend : MxxRuntime.BackendContext) (w_1104_0 : mxxMat2) (w_1105_0 : mxxMat0) (w_1106_0 : mxxMat0) (w_1107_0 : mxxMat3) (w_1108_0 : mxxMat5) (w_1109_0 : mxxMat5) (w_1110_0 : mxxMat2) (w_1111_0 : mxxMat0) (w_1113_0 : mxxMat0) : Prop :=
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1104_0 1 2 0 25 w_1105_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1104_0 2 3 0 25 w_1106_0 ∧
  MxxRuntime.concatRows w_1105_0 w_1106_0 w_1107_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1107_0 w_1108_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1108_0 w_1109_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1110_0 1 2 0 25 w_1111_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1110_0 2 3 0 25 w_1113_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev generatedRoot.constraints_47 (backend : MxxRuntime.BackendContext) (w_472_6 : mxxMat0) (w_472_8 : mxxMat0) (w_1086_0 : mxxMat2) (w_1092_0 : mxxMat2) (w_1093_0 : mxxMat0) (w_1095_0 : mxxMat0) (w_1097_0 : mxxMat0) (w_1098_0 : mxxMat0) (w_1101_0 : mxxMat3) (w_1102_0 : mxxMat5) (w_1103_0 : mxxMat5) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1092_0 1 2 0 25 w_1093_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1092_0 2 3 0 25 w_1095_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1092_0 0 1 0 25 w_1097_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1086_0 0 1 0 25 w_1098_0 ∧
  MxxRuntime.concatRows w_472_6 w_472_8 w_1101_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1101_0 w_1102_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1102_0 w_1103_0 ∧
  0 ≤ 1

abbrev generatedRoot.constraints_46 (backend : MxxRuntime.BackendContext) (w_414_6 : mxxMat0) (w_414_8 : mxxMat0) (w_1068_0 : mxxMat2) (w_1074_0 : mxxMat2) (w_1079_0 : mxxMat0) (w_1080_0 : mxxMat0) (w_1083_0 : mxxMat3) (w_1084_0 : mxxMat5) (w_1085_0 : mxxMat5) (w_1086_0 : mxxMat2) (w_1087_0 : mxxMat0) (w_1088_0 : mxxMat0) (w_1089_0 : mxxMat3) (w_1090_0 : mxxMat5) (w_1091_0 : mxxMat5) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1074_0 0 1 0 25 w_1079_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1068_0 0 1 0 25 w_1080_0 ∧
  MxxRuntime.concatRows w_414_6 w_414_8 w_1083_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1083_0 w_1084_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1084_0 w_1085_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1086_0 1 2 0 25 w_1087_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1086_0 2 3 0 25 w_1088_0 ∧
  MxxRuntime.concatRows w_1087_0 w_1088_0 w_1089_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1089_0 w_1090_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1090_0 w_1091_0

abbrev generatedRoot.constraints_45 (backend : MxxRuntime.BackendContext) (w_1068_0 : mxxMat2) (w_1069_0 : mxxMat0) (w_1070_0 : mxxMat0) (w_1071_0 : mxxMat3) (w_1072_0 : mxxMat5) (w_1073_0 : mxxMat5) (w_1074_0 : mxxMat2) (w_1075_0 : mxxMat0) (w_1077_0 : mxxMat0) : Prop :=
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1068_0 1 2 0 25 w_1069_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1068_0 2 3 0 25 w_1070_0 ∧
  MxxRuntime.concatRows w_1069_0 w_1070_0 w_1071_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1071_0 w_1072_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1072_0 w_1073_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1074_0 1 2 0 25 w_1075_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1074_0 2 3 0 25 w_1077_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev generatedRoot.constraints_44 (backend : MxxRuntime.BackendContext) (w_356_6 : mxxMat0) (w_356_8 : mxxMat0) (w_1050_0 : mxxMat2) (w_1056_0 : mxxMat2) (w_1057_0 : mxxMat0) (w_1059_0 : mxxMat0) (w_1061_0 : mxxMat0) (w_1062_0 : mxxMat0) (w_1065_0 : mxxMat3) (w_1066_0 : mxxMat5) (w_1067_0 : mxxMat5) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1056_0 1 2 0 25 w_1057_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1056_0 2 3 0 25 w_1059_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1056_0 0 1 0 25 w_1061_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1050_0 0 1 0 25 w_1062_0 ∧
  MxxRuntime.concatRows w_356_6 w_356_8 w_1065_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1065_0 w_1066_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1066_0 w_1067_0 ∧
  0 ≤ 1

abbrev generatedRoot.constraints_43 (backend : MxxRuntime.BackendContext) (w_298_6 : mxxMat0) (w_298_8 : mxxMat0) (w_1032_0 : mxxMat2) (w_1038_0 : mxxMat2) (w_1043_0 : mxxMat0) (w_1044_0 : mxxMat0) (w_1047_0 : mxxMat3) (w_1048_0 : mxxMat5) (w_1049_0 : mxxMat5) (w_1050_0 : mxxMat2) (w_1051_0 : mxxMat0) (w_1052_0 : mxxMat0) (w_1053_0 : mxxMat3) (w_1054_0 : mxxMat5) (w_1055_0 : mxxMat5) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1038_0 0 1 0 25 w_1043_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1032_0 0 1 0 25 w_1044_0 ∧
  MxxRuntime.concatRows w_298_6 w_298_8 w_1047_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1047_0 w_1048_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1048_0 w_1049_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1050_0 1 2 0 25 w_1051_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1050_0 2 3 0 25 w_1052_0 ∧
  MxxRuntime.concatRows w_1051_0 w_1052_0 w_1053_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1053_0 w_1054_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1054_0 w_1055_0

abbrev generatedRoot.constraints_42 (backend : MxxRuntime.BackendContext) (w_1032_0 : mxxMat2) (w_1033_0 : mxxMat0) (w_1034_0 : mxxMat0) (w_1035_0 : mxxMat3) (w_1036_0 : mxxMat5) (w_1037_0 : mxxMat5) (w_1038_0 : mxxMat2) (w_1039_0 : mxxMat0) (w_1041_0 : mxxMat0) : Prop :=
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1032_0 1 2 0 25 w_1033_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1032_0 2 3 0 25 w_1034_0 ∧
  MxxRuntime.concatRows w_1033_0 w_1034_0 w_1035_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1035_0 w_1036_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1036_0 w_1037_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1038_0 1 2 0 25 w_1039_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1038_0 2 3 0 25 w_1041_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev generatedRoot.constraints_41 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_14_0 : Fin 2 → mxxMat0) (w_20_0 : Fin 2 → mxxMat1) (w_240_6 : mxxMat0) (w_240_8 : mxxMat0) (w_1010_0 : mxxMat2) (w_1016_0 : mxxMat2) (w_1017_0 : mxxMat0) (w_1019_0 : mxxMat0) (w_1021_0 : mxxMat0) (w_1022_0 : mxxMat0) (w_1025_0 : mxxMat0) (w_1026_0 : mxxMat0) (w_1027_0 : mxxMat0) (w_1028_0 : mxxMat1) (w_1029_0 : mxxMat3) (w_1030_0 : mxxMat5) (w_1031_0 : mxxMat5) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1016_0 1 2 0 25 w_1017_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1016_0 2 3 0 25 w_1019_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1016_0 0 1 0 25 w_1021_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1010_0 0 1 0 25 w_1022_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_1025_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_1026_0 ∧
  MxxRuntime.familyGetStatic w_14_0 (1) w_1027_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (1) w_1028_0 ∧
  MxxRuntime.concatRows w_240_6 w_240_8 w_1029_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1029_0 w_1030_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1030_0 w_1031_0 ∧
  0 ≤ 1

abbrev generatedRoot.constraints_40 (backend : MxxRuntime.BackendContext) (w_588_6 : mxxMat0) (w_588_8 : mxxMat0) (w_992_0 : mxxMat2) (w_998_0 : mxxMat2) (w_1003_0 : mxxMat0) (w_1004_0 : mxxMat0) (w_1007_0 : mxxMat3) (w_1008_0 : mxxMat5) (w_1009_0 : mxxMat5) (w_1010_0 : mxxMat2) (w_1011_0 : mxxMat0) (w_1012_0 : mxxMat0) (w_1013_0 : mxxMat3) (w_1014_0 : mxxMat5) (w_1015_0 : mxxMat5) : Prop :=
  MxxRuntime.sliceMatrix w_998_0 0 1 0 25 w_1003_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_992_0 0 1 0 25 w_1004_0 ∧
  MxxRuntime.concatRows w_588_6 w_588_8 w_1007_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1007_0 w_1008_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1008_0 w_1009_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1010_0 1 2 0 25 w_1011_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1010_0 2 3 0 25 w_1012_0 ∧
  MxxRuntime.concatRows w_1011_0 w_1012_0 w_1013_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1013_0 w_1014_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1014_0 w_1015_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_39 (backend : MxxRuntime.BackendContext) (w_992_0 : mxxMat2) (w_993_0 : mxxMat0) (w_994_0 : mxxMat0) (w_995_0 : mxxMat3) (w_996_0 : mxxMat5) (w_997_0 : mxxMat5) (w_998_0 : mxxMat2) (w_999_0 : mxxMat0) (w_1001_0 : mxxMat0) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_992_0 1 2 0 25 w_993_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_992_0 2 3 0 25 w_994_0 ∧
  MxxRuntime.concatRows w_993_0 w_994_0 w_995_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_995_0 w_996_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_996_0 w_997_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_998_0 1 2 0 25 w_999_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_998_0 2 3 0 25 w_1001_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev generatedRoot.constraints_38 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_55_0 : mxxMat0) (w_61_0 : mxxMat0) (w_68_0 : mxxMat0) (w_530_6 : mxxMat0) (w_530_8 : mxxMat0) (w_652_0 : mxxMat0) (w_656_0 : mxxMat0) (w_870_1 : Fin 36 → mxxMat0) (w_870_2 : Fin 36 → mxxMat0) (w_870_3 : Fin 36 → mxxMat3) (w_927_1 : mxxMat7) (w_927_3 : mxxMat7) (w_950_0 : Fin 36000 → mxxMat0) (w_950_1 : Fin 36000 → mxxMat0) (w_950_2 : Fin 36000 → mxxMat0) (w_950_3 : Fin 36000 → mxxMat0) (w_950_4 : Fin 36000 → mxxMat0) (w_950_5 : Fin 36000 → mxxMat0) (w_969_2 : mxxMat0) (w_969_3 : mxxMat0) (w_969_4 : mxxMat0) (w_969_5 : mxxMat0) (w_970_0 : mxxMat9) (w_972_0 : mxxMat9) (w_973_0 : mxxMat10) (w_974_0 : mxxMat9) (w_975_0 : mxxMat10) (w_976_0 : mxxMat9) (w_977_0 : mxxMat10) (w_978_0 : mxxMat9) (w_979_0 : mxxMat10) (w_980_0 : mxxMat9) (w_981_0 : mxxMat9) (w_981_1 : mxxMat9) (w_981_2 : mxxMat9) (w_981_3 : mxxMat9) (w_981_4 : mxxMat9) (w_981_5 : mxxMat9) (w_982_0 : mxxMat1) (w_983_0 : Fin 1 → mxxMat1) (w_984_0 : Fin 36 → mxxMat3) (w_985_0 : Fin 36 → mxxMat3) (w_986_0 : Fin 36 → mxxMat3) (w_987_0 : Fin 36 → mxxMat3) (w_988_0 : Fin 36 → mxxMat3) (w_989_0 : mxxMat3) (w_990_0 : mxxMat5) (w_991_0 : mxxMat5) : Prop :=
  MxxRuntime.tensorRuns w_973_0 w_969_2 w_974_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_975_0 w_969_3 w_976_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_977_0 w_969_4 w_978_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_979_0 w_969_5 w_980_0 ∧
  0 ≤ (4499) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × mxxMat9 × Unit) => sequential_generatedRoot_981 backend hashModel tape (path ++ [981, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_950_0, (w_950_1, (w_950_2, (w_950_3, (w_950_4, (w_950_5, ())))))))))))) next) (Int.toNat (4499)) (w_970_0, (w_972_0, (w_974_0, (w_976_0, (w_978_0, (w_980_0, ())))))) (w_981_0, (w_981_1, (w_981_2, (w_981_3, (w_981_4, (w_981_5, ())))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_983 backend hashModel tape (path ++ [983, i.val]) params i w_982_0 (w_983_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_984 backend hashModel tape (path ++ [984, i.val]) params i (w_61_0, (w_68_0, ((w_870_3 i), ((w_870_1 i), (w_927_1, (w_55_0, (w_981_5, (w_983_0, (w_981_1, (w_981_3, (w_652_0, (w_656_0, ((w_870_2 i), (w_927_3, ())))))))))))))) (w_984_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_985 backend hashModel tape (path ++ [985, i.val]) params i (w_61_0, (w_68_0, ((w_870_3 i), ((w_870_1 i), (w_927_1, (w_55_0, (w_981_5, (w_983_0, (w_981_1, (w_981_3, (w_652_0, (w_656_0, ((w_870_2 i), (w_927_3, ())))))))))))))) (w_985_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_986 backend hashModel tape (path ++ [986, i.val]) params i (w_61_0, (w_68_0, ((w_870_3 i), ((w_870_1 i), (w_927_1, (w_55_0, (w_981_5, (w_983_0, (w_981_1, (w_981_3, (w_652_0, (w_656_0, ((w_870_2 i), (w_927_3, ())))))))))))))) (w_986_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_987 backend hashModel tape (path ++ [987, i.val]) params i (w_61_0, (w_68_0, ((w_870_3 i), ((w_870_1 i), (w_927_1, (w_55_0, (w_981_5, (w_983_0, (w_981_1, (w_981_3, (w_652_0, (w_656_0, ((w_870_2 i), (w_927_3, ())))))))))))))) (w_987_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_988 backend hashModel tape (path ++ [988, i.val]) params i (w_61_0, (w_68_0, ((w_870_3 i), ((w_870_1 i), (w_927_1, (w_55_0, (w_981_5, (w_983_0, (w_981_1, (w_981_3, (w_652_0, (w_656_0, ((w_870_2 i), (w_927_3, ())))))))))))))) (w_988_0 i)) ∧
  MxxRuntime.concatRows w_530_6 w_530_8 w_989_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_989_0 w_990_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_990_0 w_991_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_37 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat0) (w_12_0 : Fin 8 → mxxMat0) (w_24_0 : ByteArray) (w_101_0 : Fin 2 → mxxMat0) (w_106_0 : Fin 2 → mxxMat0) (w_120_0 : Fin 2 → mxxMat1) (w_929_0 : Fin 8 → mxxMat0) (w_930_0 : Fin 8 → mxxMat0) (w_931_0 : Fin 8 → mxxMat0) (w_932_0 : Fin 8 → mxxMat0) (w_933_0 : Fin 72000 → Int) (w_934_0 : Fin 72000 → Int) (w_937_0 : mxxMat0) (w_940_0 : mxxMat0) (w_941_0 : Fin 72000 → mxxMat0) (w_942_0 : Fin 72000 → mxxMat0) (w_945_0 : Fin 72000 → mxxMat0) (w_946_0 : Fin 72000 → mxxMat0) (w_928_0 : mxxMat10) (w_935_0 : mxxMat0) (w_936_0 : mxxMat0) (w_938_0 : mxxMat0) (w_939_0 : mxxMat0) (w_943_0 : Fin 72000 → mxxMat0) (w_943_1 : Fin 72000 → mxxMat0) (w_943_2 : Fin 72000 → mxxMat0) (w_943_3 : Fin 72000 → mxxMat0) (w_943_4 : Fin 72000 → mxxMat0) (w_943_5 : Fin 72000 → mxxMat0) (w_944_0 : Fin 36000 → mxxMat0) (w_944_1 : Fin 36000 → mxxMat0) (w_944_2 : Fin 36000 → mxxMat0) (w_944_3 : Fin 36000 → mxxMat0) (w_944_4 : Fin 36000 → mxxMat0) (w_944_5 : Fin 36000 → mxxMat0) (w_947_0 : Fin 72000 → mxxMat0) (w_947_1 : Fin 72000 → mxxMat0) (w_947_2 : Fin 72000 → mxxMat0) (w_947_3 : Fin 72000 → mxxMat0) (w_947_4 : Fin 72000 → mxxMat0) (w_947_5 : Fin 72000 → mxxMat0) (w_948_0 : Fin 36000 → mxxMat0) (w_948_1 : Fin 36000 → mxxMat0) (w_948_2 : Fin 36000 → mxxMat0) (w_948_3 : Fin 36000 → mxxMat0) (w_948_4 : Fin 36000 → mxxMat0) (w_948_5 : Fin 36000 → mxxMat0) (w_949_0 : Fin 36000 → mxxMat0) (w_949_1 : Fin 36000 → mxxMat0) (w_949_2 : Fin 36000 → mxxMat0) (w_949_3 : Fin 36000 → mxxMat0) (w_949_4 : Fin 36000 → mxxMat0) (w_949_5 : Fin 36000 → mxxMat0) (w_950_0 : Fin 36000 → mxxMat0) (w_950_1 : Fin 36000 → mxxMat0) (w_950_2 : Fin 36000 → mxxMat0) (w_950_3 : Fin 36000 → mxxMat0) (w_950_4 : Fin 36000 → mxxMat0) (w_950_5 : Fin 36000 → mxxMat0) (w_951_0 : mxxMat0) (w_952_0 : mxxMat0) (w_953_0 : mxxMat0) (w_954_0 : mxxMat0) (w_955_0 : mxxMat0) (w_956_0 : mxxMat0) (w_957_0 : mxxMat0) (w_958_0 : mxxMat0) (w_959_0 : mxxMat0) (w_960_0 : mxxMat0) (w_961_0 : mxxMat0) (w_962_0 : mxxMat0) (w_963_0 : mxxMat0) (w_964_0 : mxxMat0) (w_965_0 : mxxMat0) (w_966_0 : mxxMat0) (w_967_0 : mxxMat0) (w_968_0 : mxxMat0) (w_969_0 : mxxMat0) (w_969_1 : mxxMat0) (w_969_2 : mxxMat0) (w_969_3 : mxxMat0) (w_969_4 : mxxMat0) (w_969_5 : mxxMat0) (w_970_0 : mxxMat9) (w_971_0 : mxxMat10) (w_972_0 : mxxMat9) : Prop :=
  (72000) = 72000 ∧
  (∀ i : Fin 72000, ∃ w_101_0_gathered : mxxMat0, ∃ w_106_0_gathered : mxxMat0, ∃ w_120_0_gathered : mxxMat1, 0 ≤ (w_934_0 i) ∧
  (w_934_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_101_0 (w_934_0 i) w_101_0_gathered ∧
  0 ≤ (w_934_0 i) ∧
  (w_934_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_106_0 (w_934_0 i) w_106_0_gathered ∧
  0 ≤ (w_934_0 i) ∧
  (w_934_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_120_0 (w_934_0 i) w_120_0_gathered ∧
  parallel_generatedRoot_943 backend hashModel tape (path ++ [943, i.val]) params i ((w_934_0 i), (w_101_0_gathered, (w_5_0, (w_106_0_gathered, (w_935_0, (w_936_0, (w_937_0, (w_120_0_gathered, (w_938_0, (w_939_0, (w_940_0, (w_24_0, ((w_941_0 i), ((w_942_0 i), ())))))))))))))) ((w_943_0 i), ((w_943_1 i), ((w_943_2 i), ((w_943_3 i), ((w_943_4 i), ((w_943_5 i), ()))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_944 backend hashModel tape (path ++ [944, i.val]) params i (w_943_0, (w_943_1, (w_943_2, (w_943_3, (w_943_4, (w_943_5, ())))))) ((w_944_0 i), ((w_944_1 i), ((w_944_2 i), ((w_944_3 i), ((w_944_4 i), ((w_944_5 i), ()))))))) ∧
  (72000) = 72000 ∧
  (∀ i : Fin 72000, ∃ w_101_0_gathered : mxxMat0, ∃ w_106_0_gathered : mxxMat0, ∃ w_120_0_gathered : mxxMat1, 0 ≤ (w_933_0 i) ∧
  (w_933_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_101_0 (w_933_0 i) w_101_0_gathered ∧
  0 ≤ (w_933_0 i) ∧
  (w_933_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_106_0 (w_933_0 i) w_106_0_gathered ∧
  0 ≤ (w_933_0 i) ∧
  (w_933_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_120_0 (w_933_0 i) w_120_0_gathered ∧
  parallel_generatedRoot_947 backend hashModel tape (path ++ [947, i.val]) params i ((w_933_0 i), (w_101_0_gathered, (w_5_0, (w_106_0_gathered, (w_944_0, (w_944_2, (w_944_4, (w_120_0_gathered, (w_944_1, (w_944_3, (w_944_5, (w_24_0, ((w_945_0 i), ((w_946_0 i), ())))))))))))))) ((w_947_0 i), ((w_947_1 i), ((w_947_2 i), ((w_947_3 i), ((w_947_4 i), ((w_947_5 i), ()))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_948 backend hashModel tape (path ++ [948, i.val]) params i (w_947_0, (w_947_1, (w_947_2, (w_947_3, (w_947_4, (w_947_5, ())))))) ((w_948_0 i), ((w_948_1 i), ((w_948_2 i), ((w_948_3 i), ((w_948_4 i), ((w_948_5 i), ()))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_949 backend hashModel tape (path ++ [949, i.val]) params i (w_929_0, (w_930_0, (w_931_0, (w_932_0, (w_11_0, (w_5_0, (w_12_0, ((w_948_4 i), ((w_948_0 i), ((w_948_2 i), ((w_948_5 i), ((w_948_1 i), ((w_948_3 i), ()))))))))))))) ((w_949_0 i), ((w_949_1 i), ((w_949_2 i), ((w_949_3 i), ((w_949_4 i), ((w_949_5 i), ()))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_950 backend hashModel tape (path ++ [950, i.val]) params i (w_949_0, (w_949_1, (w_949_2, (w_949_3, (w_949_4, (w_949_5, ())))))) ((w_950_0 i), ((w_950_1 i), ((w_950_2 i), ((w_950_3 i), ((w_950_4 i), ((w_950_5 i), ()))))))) ∧
  MxxRuntime.familyGetStatic w_950_0 (0) w_951_0 ∧
  MxxRuntime.familyGetStatic w_950_0 (1) w_952_0 ∧
  MxxRuntime.familyGetStatic w_950_1 (0) w_954_0 ∧
  MxxRuntime.familyGetStatic w_950_1 (1) w_955_0 ∧
  MxxRuntime.familyGetStatic w_950_2 (0) w_957_0 ∧
  MxxRuntime.familyGetStatic w_950_2 (1) w_958_0 ∧
  MxxRuntime.familyGetStatic w_950_3 (0) w_960_0 ∧
  MxxRuntime.familyGetStatic w_950_3 (1) w_961_0 ∧
  MxxRuntime.familyGetStatic w_950_4 (0) w_963_0 ∧
  MxxRuntime.familyGetStatic w_950_4 (1) w_964_0 ∧
  MxxRuntime.familyGetStatic w_950_5 (0) w_966_0 ∧
  MxxRuntime.familyGetStatic w_950_5 (1) w_967_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_969 backend hashModel tape (path ++ [969, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_950_0, (w_950_1, (w_950_2, (w_950_3, (w_950_4, (w_950_5, ())))))))))))) next) (Int.toNat (6)) (w_953_0, (w_956_0, (w_959_0, (w_962_0, (w_965_0, (w_968_0, ())))))) (w_969_0, (w_969_1, (w_969_2, (w_969_3, (w_969_4, (w_969_5, ())))))) ∧
  MxxRuntime.tensorRuns w_928_0 w_969_0 w_970_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_971_0 w_969_1 w_972_0 ∧
  0 ≤ (0) ∧
  (0) < 4500

abbrev generatedRoot.constraints_36 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_871_0 : mxxMat8) (w_896_0 : Fin 7200 → mxxMat0) (w_896_1 : Fin 7200 → mxxMat0) (w_896_2 : Fin 7200 → mxxMat0) (w_896_3 : Fin 7200 → mxxMat0) (w_896_4 : Fin 7200 → mxxMat0) (w_896_5 : Fin 7200 → mxxMat0) (w_899_0 : mxxMat0) (w_902_0 : mxxMat0) (w_905_0 : mxxMat0) (w_906_0 : mxxMat0) (w_907_0 : mxxMat0) (w_908_0 : mxxMat0) (w_909_0 : mxxMat0) (w_910_0 : mxxMat0) (w_911_0 : mxxMat0) (w_912_0 : mxxMat0) (w_913_0 : mxxMat0) (w_914_0 : mxxMat0) (w_915_0 : mxxMat0) (w_915_1 : mxxMat0) (w_915_2 : mxxMat0) (w_915_3 : mxxMat0) (w_915_4 : mxxMat0) (w_915_5 : mxxMat0) (w_916_0 : mxxMat7) (w_917_0 : mxxMat8) (w_918_0 : mxxMat7) (w_919_0 : mxxMat8) (w_920_0 : mxxMat7) (w_921_0 : mxxMat8) (w_922_0 : mxxMat7) (w_923_0 : mxxMat8) (w_924_0 : mxxMat7) (w_925_0 : mxxMat8) (w_926_0 : mxxMat7) (w_927_0 : mxxMat7) (w_927_1 : mxxMat7) (w_927_2 : mxxMat7) (w_927_3 : mxxMat7) (w_927_4 : mxxMat7) (w_927_5 : mxxMat7) (w_935_0 : mxxMat0) (w_936_0 : mxxMat0) (w_938_0 : mxxMat0) (w_939_0 : mxxMat0) : Prop :=
  MxxRuntime.familyGetStatic w_896_3 (0) w_906_0 ∧
  MxxRuntime.familyGetStatic w_896_3 (1) w_907_0 ∧
  MxxRuntime.familyGetStatic w_896_4 (0) w_909_0 ∧
  MxxRuntime.familyGetStatic w_896_4 (1) w_910_0 ∧
  MxxRuntime.familyGetStatic w_896_5 (0) w_912_0 ∧
  MxxRuntime.familyGetStatic w_896_5 (1) w_913_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_915 backend hashModel tape (path ++ [915, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_896_0, (w_896_1, (w_896_2, (w_896_3, (w_896_4, (w_896_5, ())))))))))))) next) (Int.toNat (6)) (w_899_0, (w_902_0, (w_905_0, (w_908_0, (w_911_0, (w_914_0, ())))))) (w_915_0, (w_915_1, (w_915_2, (w_915_3, (w_915_4, (w_915_5, ())))))) ∧
  MxxRuntime.tensorRuns w_871_0 w_915_0 w_916_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_917_0 w_915_1 w_918_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_919_0 w_915_2 w_920_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_921_0 w_915_3 w_922_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_923_0 w_915_4 w_924_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_925_0 w_915_5 w_926_0 ∧
  0 ≤ (899) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × mxxMat7 × Unit) => sequential_generatedRoot_927 backend hashModel tape (path ++ [927, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_896_0, (w_896_1, (w_896_2, (w_896_3, (w_896_4, (w_896_5, ())))))))))))) next) (Int.toNat (899)) (w_916_0, (w_918_0, (w_920_0, (w_922_0, (w_924_0, (w_926_0, ())))))) (w_927_0, (w_927_1, (w_927_2, (w_927_3, (w_927_4, (w_927_5, ())))))) ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_935_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_936_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_938_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_939_0

abbrev generatedRoot.constraints_35 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat0) (w_12_0 : Fin 8 → mxxMat0) (w_24_0 : ByteArray) (w_753_0 : ByteArray) (w_872_0 : Fin 8 → mxxMat0) (w_873_0 : Fin 8 → mxxMat0) (w_874_0 : Fin 8 → mxxMat0) (w_875_0 : Fin 8 → mxxMat0) (w_876_0 : Fin 14400 → Int) (w_877_0 : Fin 2 → mxxMat0) (w_878_0 : Fin 2 → mxxMat0) (w_879_0 : Fin 14400 → Int) (w_882_0 : mxxMat0) (w_883_0 : Fin 2 → mxxMat1) (w_886_0 : mxxMat0) (w_887_0 : Fin 14400 → mxxMat0) (w_888_0 : Fin 14400 → mxxMat0) (w_891_0 : Fin 14400 → mxxMat0) (w_892_0 : Fin 14400 → mxxMat0) (w_636_0 : Fin 26 → mxxMat0) (w_658_0 : Fin 5 → mxxMat0) (w_658_1 : Fin 5 → mxxMat0) (w_658_2 : Fin 5 → mxxMat0) (w_658_3 : Fin 5 → mxxMat0) (w_658_4 : Fin 5 → mxxMat0) (w_658_5 : Fin 5 → mxxMat0) (w_705_0 : Fin 26 → mxxMat0) (w_752_0 : Fin 26 → mxxMat0) (w_757_0 : mxxMat3) (w_761_0 : mxxMat3) (w_765_0 : mxxMat3) (w_769_0 : mxxMat3) (w_773_0 : mxxMat3) (w_777_0 : mxxMat3) (w_782_0 : mxxMat3) (w_787_0 : mxxMat3) (w_792_0 : mxxMat3) (w_797_0 : mxxMat3) (w_802_0 : mxxMat3) (w_807_0 : mxxMat3) (w_812_0 : mxxMat3) (w_817_0 : mxxMat3) (w_822_0 : mxxMat3) (w_827_0 : mxxMat3) (w_832_0 : mxxMat3) (w_837_0 : mxxMat3) (w_842_0 : mxxMat3) (w_847_0 : mxxMat3) (w_852_0 : mxxMat3) (w_857_0 : mxxMat3) (w_860_0 : mxxMat29) (w_862_0 : mxxMat3) (w_864_0 : mxxMat3) (w_866_0 : mxxMat3) (w_867_0 : mxxMat3) (w_868_0 : mxxMat3) (w_869_0 : Fin 26 → mxxMat3) (w_870_0 : Fin 36 → mxxMat0) (w_870_1 : Fin 36 → mxxMat0) (w_870_2 : Fin 36 → mxxMat0) (w_870_3 : Fin 36 → mxxMat3) (w_880_0 : mxxMat0) (w_881_0 : mxxMat0) (w_884_0 : mxxMat0) (w_885_0 : mxxMat0) (w_889_0 : Fin 14400 → mxxMat0) (w_889_1 : Fin 14400 → mxxMat0) (w_889_2 : Fin 14400 → mxxMat0) (w_889_3 : Fin 14400 → mxxMat0) (w_889_4 : Fin 14400 → mxxMat0) (w_889_5 : Fin 14400 → mxxMat0) (w_890_0 : Fin 7200 → mxxMat0) (w_890_1 : Fin 7200 → mxxMat0) (w_890_2 : Fin 7200 → mxxMat0) (w_890_3 : Fin 7200 → mxxMat0) (w_890_4 : Fin 7200 → mxxMat0) (w_890_5 : Fin 7200 → mxxMat0) (w_893_0 : Fin 14400 → mxxMat0) (w_893_1 : Fin 14400 → mxxMat0) (w_893_2 : Fin 14400 → mxxMat0) (w_893_3 : Fin 14400 → mxxMat0) (w_893_4 : Fin 14400 → mxxMat0) (w_893_5 : Fin 14400 → mxxMat0) (w_894_0 : Fin 7200 → mxxMat0) (w_894_1 : Fin 7200 → mxxMat0) (w_894_2 : Fin 7200 → mxxMat0) (w_894_3 : Fin 7200 → mxxMat0) (w_894_4 : Fin 7200 → mxxMat0) (w_894_5 : Fin 7200 → mxxMat0) (w_895_0 : Fin 7200 → mxxMat0) (w_895_1 : Fin 7200 → mxxMat0) (w_895_2 : Fin 7200 → mxxMat0) (w_895_3 : Fin 7200 → mxxMat0) (w_895_4 : Fin 7200 → mxxMat0) (w_895_5 : Fin 7200 → mxxMat0) (w_896_0 : Fin 7200 → mxxMat0) (w_896_1 : Fin 7200 → mxxMat0) (w_896_2 : Fin 7200 → mxxMat0) (w_896_3 : Fin 7200 → mxxMat0) (w_896_4 : Fin 7200 → mxxMat0) (w_896_5 : Fin 7200 → mxxMat0) (w_897_0 : mxxMat0) (w_898_0 : mxxMat0) (w_900_0 : mxxMat0) (w_901_0 : mxxMat0) (w_903_0 : mxxMat0) (w_904_0 : mxxMat0) : Prop :=
  0 ≤ 75 ∧
  75 < 100 ∧
  100 ≤ 100 ∧
  MxxRuntime.sliceMatrix w_860_0 0 2 75 100 w_867_0 ∧
  MxxRuntime.modulusReduceRuns w_867_0 w_868_0 ∧
  MxxRuntime.familyPack (26) [w_757_0, w_761_0, w_765_0, w_769_0, w_773_0, w_777_0, w_782_0, w_787_0, w_792_0, w_797_0, w_802_0, w_807_0, w_812_0, w_817_0, w_822_0, w_827_0, w_832_0, w_837_0, w_842_0, w_847_0, w_852_0, w_857_0, w_862_0, w_864_0, w_866_0, w_868_0] w_869_0 ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_870 backend hashModel tape (path ++ [870, i.val]) params i (w_636_0, (w_658_0, (w_658_1, (w_705_0, (w_658_2, (w_658_3, (w_752_0, (w_658_4, (w_658_5, (w_869_0, (w_753_0, ()))))))))))) ((w_870_0 i), ((w_870_1 i), ((w_870_2 i), ((w_870_3 i), ()))))) ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_880_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_881_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_884_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_885_0 ∧
  (14400) = 14400 ∧
  (∀ i : Fin 14400, ∃ w_877_0_gathered : mxxMat0, ∃ w_878_0_gathered : mxxMat0, ∃ w_883_0_gathered : mxxMat1, 0 ≤ (w_879_0 i) ∧
  (w_879_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_877_0 (w_879_0 i) w_877_0_gathered ∧
  0 ≤ (w_879_0 i) ∧
  (w_879_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_878_0 (w_879_0 i) w_878_0_gathered ∧
  0 ≤ (w_879_0 i) ∧
  (w_879_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_883_0 (w_879_0 i) w_883_0_gathered ∧
  parallel_generatedRoot_889 backend hashModel tape (path ++ [889, i.val]) params i ((w_879_0 i), (w_877_0_gathered, (w_5_0, (w_878_0_gathered, (w_880_0, (w_881_0, (w_882_0, (w_883_0_gathered, (w_884_0, (w_885_0, (w_886_0, (w_24_0, ((w_887_0 i), ((w_888_0 i), ())))))))))))))) ((w_889_0 i), ((w_889_1 i), ((w_889_2 i), ((w_889_3 i), ((w_889_4 i), ((w_889_5 i), ()))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_890 backend hashModel tape (path ++ [890, i.val]) params i (w_889_0, (w_889_1, (w_889_2, (w_889_3, (w_889_4, (w_889_5, ())))))) ((w_890_0 i), ((w_890_1 i), ((w_890_2 i), ((w_890_3 i), ((w_890_4 i), ((w_890_5 i), ()))))))) ∧
  (14400) = 14400 ∧
  (∀ i : Fin 14400, ∃ w_877_0_gathered : mxxMat0, ∃ w_878_0_gathered : mxxMat0, ∃ w_883_0_gathered : mxxMat1, 0 ≤ (w_876_0 i) ∧
  (w_876_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_877_0 (w_876_0 i) w_877_0_gathered ∧
  0 ≤ (w_876_0 i) ∧
  (w_876_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_878_0 (w_876_0 i) w_878_0_gathered ∧
  0 ≤ (w_876_0 i) ∧
  (w_876_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_883_0 (w_876_0 i) w_883_0_gathered ∧
  parallel_generatedRoot_893 backend hashModel tape (path ++ [893, i.val]) params i ((w_876_0 i), (w_877_0_gathered, (w_5_0, (w_878_0_gathered, (w_890_0, (w_890_2, (w_890_4, (w_883_0_gathered, (w_890_1, (w_890_3, (w_890_5, (w_24_0, ((w_891_0 i), ((w_892_0 i), ())))))))))))))) ((w_893_0 i), ((w_893_1 i), ((w_893_2 i), ((w_893_3 i), ((w_893_4 i), ((w_893_5 i), ()))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_894 backend hashModel tape (path ++ [894, i.val]) params i (w_893_0, (w_893_1, (w_893_2, (w_893_3, (w_893_4, (w_893_5, ())))))) ((w_894_0 i), ((w_894_1 i), ((w_894_2 i), ((w_894_3 i), ((w_894_4 i), ((w_894_5 i), ()))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_895 backend hashModel tape (path ++ [895, i.val]) params i (w_872_0, (w_873_0, (w_874_0, (w_875_0, (w_11_0, (w_5_0, (w_12_0, ((w_894_4 i), ((w_894_0 i), ((w_894_2 i), ((w_894_5 i), ((w_894_1 i), ((w_894_3 i), ()))))))))))))) ((w_895_0 i), ((w_895_1 i), ((w_895_2 i), ((w_895_3 i), ((w_895_4 i), ((w_895_5 i), ()))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_896 backend hashModel tape (path ++ [896, i.val]) params i (w_895_0, (w_895_1, (w_895_2, (w_895_3, (w_895_4, (w_895_5, ())))))) ((w_896_0 i), ((w_896_1 i), ((w_896_2 i), ((w_896_3 i), ((w_896_4 i), ((w_896_5 i), ()))))))) ∧
  MxxRuntime.familyGetStatic w_896_0 (0) w_897_0 ∧
  MxxRuntime.familyGetStatic w_896_0 (1) w_898_0 ∧
  MxxRuntime.familyGetStatic w_896_1 (0) w_900_0 ∧
  MxxRuntime.familyGetStatic w_896_1 (1) w_901_0 ∧
  MxxRuntime.familyGetStatic w_896_2 (0) w_903_0 ∧
  MxxRuntime.familyGetStatic w_896_2 (1) w_904_0

abbrev generatedRoot.constraints_34 (hashModel : MxxRuntime.HashModel) (w_753_0 : ByteArray) (w_858_0 : mxxMat27) (w_853_0 : Int) (w_854_0 : mxxMat0) (w_855_0 : mxxMat0) (w_856_0 : mxxMat3) (w_857_0 : mxxMat3) (w_859_0 : mxxMat28) (w_860_0 : mxxMat29) (w_861_0 : mxxMat3) (w_862_0 : mxxMat3) (w_863_0 : mxxMat3) (w_864_0 : mxxMat3) (w_865_0 : mxxMat3) (w_866_0 : mxxMat3) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_853_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_855_0 ∧
  MxxRuntime.concatRows w_854_0 w_855_0 w_856_0 ∧
  MxxRuntime.modulusReduceRuns w_856_0 w_857_0 ∧
  MxxRuntime.hashSample (hashModel) ([100, 105, 111, 50, 47, 97, 110, 99, 104, 111, 114, 45, 112, 97, 100, 100, 105, 110, 103]) ([]) (w_753_0) w_859_0 ∧
  MxxRuntime.concatColumns w_858_0 w_859_0 w_860_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 100 ∧
  MxxRuntime.sliceMatrix w_860_0 0 2 0 25 w_861_0 ∧
  MxxRuntime.modulusReduceRuns w_861_0 w_862_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 25 ∧
  25 < 50 ∧
  50 ≤ 100 ∧
  MxxRuntime.sliceMatrix w_860_0 0 2 25 50 w_863_0 ∧
  MxxRuntime.modulusReduceRuns w_863_0 w_864_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 50 ∧
  50 < 75 ∧
  75 ≤ 100 ∧
  MxxRuntime.sliceMatrix w_860_0 0 2 50 75 w_865_0 ∧
  MxxRuntime.modulusReduceRuns w_865_0 w_866_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2

abbrev generatedRoot.constraints_33 (hashModel : MxxRuntime.HashModel) (w_753_0 : ByteArray) (w_813_0 : Int) (w_814_0 : mxxMat0) (w_815_0 : mxxMat0) (w_816_0 : mxxMat3) (w_817_0 : mxxMat3) (w_818_0 : Int) (w_819_0 : mxxMat0) (w_820_0 : mxxMat0) (w_821_0 : mxxMat3) (w_822_0 : mxxMat3) (w_823_0 : Int) (w_824_0 : mxxMat0) (w_825_0 : mxxMat0) (w_826_0 : mxxMat3) (w_827_0 : mxxMat3) (w_828_0 : Int) (w_829_0 : mxxMat0) (w_830_0 : mxxMat0) (w_831_0 : mxxMat3) (w_832_0 : mxxMat3) (w_833_0 : Int) (w_834_0 : mxxMat0) (w_835_0 : mxxMat0) (w_836_0 : mxxMat3) (w_837_0 : mxxMat3) (w_838_0 : Int) (w_839_0 : mxxMat0) (w_840_0 : mxxMat0) (w_841_0 : mxxMat3) (w_842_0 : mxxMat3) (w_843_0 : Int) (w_844_0 : mxxMat0) (w_845_0 : mxxMat0) (w_846_0 : mxxMat3) (w_847_0 : mxxMat3) (w_848_0 : Int) (w_849_0 : mxxMat0) (w_850_0 : mxxMat0) (w_851_0 : mxxMat3) (w_852_0 : mxxMat3) (w_853_0 : Int) (w_854_0 : mxxMat0) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_813_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_815_0 ∧
  MxxRuntime.concatRows w_814_0 w_815_0 w_816_0 ∧
  MxxRuntime.modulusReduceRuns w_816_0 w_817_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_818_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_819_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_818_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_820_0 ∧
  MxxRuntime.concatRows w_819_0 w_820_0 w_821_0 ∧
  MxxRuntime.modulusReduceRuns w_821_0 w_822_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_823_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_824_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_823_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_825_0 ∧
  MxxRuntime.concatRows w_824_0 w_825_0 w_826_0 ∧
  MxxRuntime.modulusReduceRuns w_826_0 w_827_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_828_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_829_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_828_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_830_0 ∧
  MxxRuntime.concatRows w_829_0 w_830_0 w_831_0 ∧
  MxxRuntime.modulusReduceRuns w_831_0 w_832_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_833_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_834_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_833_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_835_0 ∧
  MxxRuntime.concatRows w_834_0 w_835_0 w_836_0 ∧
  MxxRuntime.modulusReduceRuns w_836_0 w_837_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_838_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_839_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_838_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_840_0 ∧
  MxxRuntime.concatRows w_839_0 w_840_0 w_841_0 ∧
  MxxRuntime.modulusReduceRuns w_841_0 w_842_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_843_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_844_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_843_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_845_0 ∧
  MxxRuntime.concatRows w_844_0 w_845_0 w_846_0 ∧
  MxxRuntime.modulusReduceRuns w_846_0 w_847_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_848_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_849_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_848_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_850_0 ∧
  MxxRuntime.concatRows w_849_0 w_850_0 w_851_0 ∧
  MxxRuntime.modulusReduceRuns w_851_0 w_852_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_853_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_854_0

abbrev generatedRoot.constraints_32 (hashModel : MxxRuntime.HashModel) (w_753_0 : ByteArray) (w_774_0 : mxxMat0) (w_775_0 : mxxMat0) (w_776_0 : mxxMat3) (w_777_0 : mxxMat3) (w_778_0 : Int) (w_779_0 : mxxMat0) (w_780_0 : mxxMat0) (w_781_0 : mxxMat3) (w_782_0 : mxxMat3) (w_783_0 : Int) (w_784_0 : mxxMat0) (w_785_0 : mxxMat0) (w_786_0 : mxxMat3) (w_787_0 : mxxMat3) (w_788_0 : Int) (w_789_0 : mxxMat0) (w_790_0 : mxxMat0) (w_791_0 : mxxMat3) (w_792_0 : mxxMat3) (w_793_0 : Int) (w_794_0 : mxxMat0) (w_795_0 : mxxMat0) (w_796_0 : mxxMat3) (w_797_0 : mxxMat3) (w_798_0 : Int) (w_799_0 : mxxMat0) (w_800_0 : mxxMat0) (w_801_0 : mxxMat3) (w_802_0 : mxxMat3) (w_803_0 : Int) (w_804_0 : mxxMat0) (w_805_0 : mxxMat0) (w_806_0 : mxxMat3) (w_807_0 : mxxMat3) (w_808_0 : Int) (w_809_0 : mxxMat0) (w_810_0 : mxxMat0) (w_811_0 : mxxMat3) (w_812_0 : mxxMat3) (w_813_0 : Int) (w_814_0 : mxxMat0) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_775_0 ∧
  MxxRuntime.concatRows w_774_0 w_775_0 w_776_0 ∧
  MxxRuntime.modulusReduceRuns w_776_0 w_777_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_778_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_779_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_778_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_780_0 ∧
  MxxRuntime.concatRows w_779_0 w_780_0 w_781_0 ∧
  MxxRuntime.modulusReduceRuns w_781_0 w_782_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_783_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_784_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_783_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_785_0 ∧
  MxxRuntime.concatRows w_784_0 w_785_0 w_786_0 ∧
  MxxRuntime.modulusReduceRuns w_786_0 w_787_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_788_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_789_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_788_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_790_0 ∧
  MxxRuntime.concatRows w_789_0 w_790_0 w_791_0 ∧
  MxxRuntime.modulusReduceRuns w_791_0 w_792_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_793_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_794_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_793_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_795_0 ∧
  MxxRuntime.concatRows w_794_0 w_795_0 w_796_0 ∧
  MxxRuntime.modulusReduceRuns w_796_0 w_797_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_798_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_799_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_798_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_800_0 ∧
  MxxRuntime.concatRows w_799_0 w_800_0 w_801_0 ∧
  MxxRuntime.modulusReduceRuns w_801_0 w_802_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_803_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_804_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_803_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_805_0 ∧
  MxxRuntime.concatRows w_804_0 w_805_0 w_806_0 ∧
  MxxRuntime.modulusReduceRuns w_806_0 w_807_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_808_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_809_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_808_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_810_0 ∧
  MxxRuntime.concatRows w_809_0 w_810_0 w_811_0 ∧
  MxxRuntime.modulusReduceRuns w_811_0 w_812_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_813_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_814_0

abbrev generatedRoot.constraints_31 (hashModel : MxxRuntime.HashModel) (w_753_0 : ByteArray) (w_559_0 : mxxMat2) (w_617_0 : mxxMat2) (w_635_0 : mxxMat0) (w_653_0 : mxxMat0) (w_657_0 : mxxMat0) (w_706_0 : mxxMat0) (w_707_0 : mxxMat0) (w_710_0 : mxxMat0) (w_713_0 : mxxMat0) (w_715_0 : mxxMat0) (w_716_0 : mxxMat0) (w_720_0 : mxxMat0) (w_721_0 : mxxMat0) (w_725_0 : mxxMat0) (w_726_0 : mxxMat0) (w_730_0 : mxxMat0) (w_731_0 : mxxMat0) (w_735_0 : mxxMat0) (w_736_0 : mxxMat0) (w_740_0 : mxxMat0) (w_741_0 : mxxMat0) (w_742_0 : mxxMat0) (w_745_0 : mxxMat0) (w_746_0 : mxxMat0) (w_747_0 : mxxMat0) (w_750_0 : mxxMat0) (w_751_0 : mxxMat0) (w_752_0 : Fin 26 → mxxMat0) (w_754_0 : mxxMat0) (w_755_0 : mxxMat0) (w_756_0 : mxxMat3) (w_757_0 : mxxMat3) (w_758_0 : mxxMat0) (w_759_0 : mxxMat0) (w_760_0 : mxxMat3) (w_761_0 : mxxMat3) (w_762_0 : mxxMat0) (w_763_0 : mxxMat0) (w_764_0 : mxxMat3) (w_765_0 : mxxMat3) (w_766_0 : mxxMat0) (w_767_0 : mxxMat0) (w_768_0 : mxxMat3) (w_769_0 : mxxMat3) (w_770_0 : mxxMat0) (w_771_0 : mxxMat0) (w_772_0 : mxxMat3) (w_773_0 : mxxMat3) (w_774_0 : mxxMat0) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_559_0 2 3 0 25 w_742_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_617_0 2 3 0 25 w_747_0 ∧
  MxxRuntime.familyPack (26) [w_706_0, w_707_0, w_653_0, w_657_0, w_710_0, w_713_0, w_715_0, w_716_0, w_720_0, w_721_0, w_725_0, w_726_0, w_730_0, w_731_0, w_735_0, w_736_0, w_740_0, w_741_0, w_745_0, w_746_0, w_750_0, w_751_0, w_635_0, w_635_0, w_635_0, w_635_0] w_752_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_754_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_755_0 ∧
  MxxRuntime.concatRows w_754_0 w_755_0 w_756_0 ∧
  MxxRuntime.modulusReduceRuns w_756_0 w_757_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_758_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_759_0 ∧
  MxxRuntime.concatRows w_758_0 w_759_0 w_760_0 ∧
  MxxRuntime.modulusReduceRuns w_760_0 w_761_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_762_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_763_0 ∧
  MxxRuntime.concatRows w_762_0 w_763_0 w_764_0 ∧
  MxxRuntime.modulusReduceRuns w_764_0 w_765_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_766_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_753_0) w_767_0 ∧
  MxxRuntime.concatRows w_766_0 w_767_0 w_768_0 ∧
  MxxRuntime.modulusReduceRuns w_768_0 w_769_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_753_0) w_770_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_753_0) w_771_0 ∧
  MxxRuntime.concatRows w_770_0 w_771_0 w_772_0 ∧
  MxxRuntime.modulusReduceRuns w_772_0 w_773_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_753_0) w_774_0

abbrev generatedRoot.constraints_30 (w_327_0 : mxxMat2) (w_385_0 : mxxMat2) (w_443_0 : mxxMat2) (w_501_0 : mxxMat2) (w_722_0 : mxxMat0) (w_727_0 : mxxMat0) (w_732_0 : mxxMat0) (w_737_0 : mxxMat0) : Prop :=
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_327_0 2 3 0 25 w_722_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_385_0 2 3 0 25 w_727_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_443_0 2 3 0 25 w_732_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_501_0 2 3 0 25 w_737_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_29 (w_190_0 : mxxMat2) (w_210_0 : mxxMat2) (w_269_0 : mxxMat2) (w_559_0 : mxxMat2) (w_617_0 : mxxMat2) (w_635_0 : mxxMat0) (w_651_0 : mxxMat0) (w_654_0 : mxxMat0) (w_659_0 : mxxMat0) (w_660_0 : mxxMat0) (w_663_0 : mxxMat0) (w_666_0 : mxxMat0) (w_668_0 : mxxMat0) (w_669_0 : mxxMat0) (w_673_0 : mxxMat0) (w_674_0 : mxxMat0) (w_678_0 : mxxMat0) (w_679_0 : mxxMat0) (w_683_0 : mxxMat0) (w_684_0 : mxxMat0) (w_688_0 : mxxMat0) (w_689_0 : mxxMat0) (w_693_0 : mxxMat0) (w_694_0 : mxxMat0) (w_695_0 : mxxMat0) (w_698_0 : mxxMat0) (w_699_0 : mxxMat0) (w_700_0 : mxxMat0) (w_703_0 : mxxMat0) (w_704_0 : mxxMat0) (w_705_0 : Fin 26 → mxxMat0) (w_708_0 : mxxMat0) (w_711_0 : mxxMat0) (w_717_0 : mxxMat0) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_559_0 1 2 0 25 w_695_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_617_0 1 2 0 25 w_700_0 ∧
  MxxRuntime.familyPack (26) [w_659_0, w_660_0, w_651_0, w_654_0, w_663_0, w_666_0, w_668_0, w_669_0, w_673_0, w_674_0, w_678_0, w_679_0, w_683_0, w_684_0, w_688_0, w_689_0, w_693_0, w_694_0, w_698_0, w_699_0, w_703_0, w_704_0, w_635_0, w_635_0, w_635_0, w_635_0] w_705_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_190_0 2 3 0 25 w_708_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_210_0 2 3 0 25 w_711_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_269_0 2 3 0 25 w_717_0

abbrev generatedRoot.constraints_28 (w_327_0 : mxxMat2) (w_385_0 : mxxMat2) (w_443_0 : mxxMat2) (w_501_0 : mxxMat2) (w_675_0 : mxxMat0) (w_680_0 : mxxMat0) (w_685_0 : mxxMat0) (w_690_0 : mxxMat0) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_327_0 1 2 0 25 w_675_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_385_0 1 2 0 25 w_680_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_443_0 1 2 0 25 w_685_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_501_0 1 2 0 25 w_690_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_27 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_638_0 : Fin 2 → mxxMat0) (w_640_0 : Fin 4 → mxxMat0) (w_641_0 : Fin 2 → mxxMat0) (w_643_0 : Fin 4 → mxxMat0) (w_649_0 : Fin 5 → Int) (w_86_0 : mxxMat0) (w_89_0 : mxxMat0) (w_190_0 : mxxMat2) (w_210_0 : mxxMat2) (w_269_0 : mxxMat2) (w_637_0 : Fin 5 → Int) (w_637_1 : Fin 5 → Int) (w_639_0 : mxxMat0) (w_642_0 : mxxMat0) (w_644_0 : mxxMat0) (w_645_0 : mxxMat0) (w_646_0 : mxxMat0) (w_647_0 : mxxMat0) (w_648_0 : mxxMat1) (w_650_0 : mxxMat0) (w_651_0 : mxxMat0) (w_653_0 : mxxMat0) (w_654_0 : mxxMat0) (w_655_0 : mxxMat0) (w_657_0 : mxxMat0) (w_658_0 : Fin 5 → mxxMat0) (w_658_1 : Fin 5 → mxxMat0) (w_658_2 : Fin 5 → mxxMat0) (w_658_3 : Fin 5 → mxxMat0) (w_658_4 : Fin 5 → mxxMat0) (w_658_5 : Fin 5 → mxxMat0) (w_661_0 : mxxMat0) (w_664_0 : mxxMat0) (w_670_0 : mxxMat0) : Prop :=
  MxxRuntime.sha256BitColumns [17, 229, 242, 240, 20, 113, 12, 254, 126, 237, 25, 125, 17, 124, 200, 38, 220, 135, 58, 4, 215, 86, 146, 138, 135, 103, 188, 97, 153, 28, 159, 14] 4 5 2 w_637_0 w_637_1 ∧
  MxxRuntime.familyGetStatic w_638_0 (1) w_639_0 ∧
  MxxRuntime.familyGetStatic w_641_0 (1) w_642_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_644_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_645_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_646_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_647_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_650_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_655_0 ∧
  (5) = 5 ∧
  (∀ i : Fin 5, parallel_generatedRoot_658 backend hashModel tape (path ++ [658, i.val]) params i (w_637_1, (w_639_0, (w_640_0, (w_637_0, (w_642_0, (w_643_0, (w_644_0, (w_5_0, (w_645_0, (w_646_0, (w_647_0, (w_648_0, (w_649_0, (w_650_0, (w_651_0, (w_653_0, (w_86_0, (w_654_0, (w_657_0, (w_89_0, ())))))))))))))))))))) ((w_658_0 i), ((w_658_1 i), ((w_658_2 i), ((w_658_3 i), ((w_658_4 i), ((w_658_5 i), ()))))))) ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_190_0 1 2 0 25 w_661_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_210_0 1 2 0 25 w_664_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_269_0 1 2 0 25 w_670_0

abbrev generatedRoot.constraints_26 (backend : MxxRuntime.BackendContext) (w_55_0 : mxxMat0) (w_72_0 : mxxMat0) (w_78_0 : mxxMat0) (w_79_0 : mxxMat0) (w_86_0 : mxxMat0) (w_89_0 : mxxMat0) (w_200_0 : mxxMat0) (w_215_0 : mxxMat0) (w_227_0 : mxxMat0) (w_228_0 : mxxMat0) (w_285_0 : mxxMat0) (w_286_0 : mxxMat0) (w_343_0 : mxxMat0) (w_344_0 : mxxMat0) (w_401_0 : mxxMat0) (w_402_0 : mxxMat0) (w_459_0 : mxxMat0) (w_460_0 : mxxMat0) (w_517_0 : mxxMat0) (w_518_0 : mxxMat0) (w_575_0 : mxxMat0) (w_576_0 : mxxMat0) (w_611_0 : mxxMat2) (w_612_0 : mxxMat0) (w_613_0 : mxxMat0) (w_614_0 : mxxMat3) (w_615_0 : mxxMat5) (w_616_0 : mxxMat5) (w_617_0 : mxxMat2) (w_618_0 : mxxMat0) (w_619_0 : mxxMat0) (w_622_0 : mxxMat0) (w_624_0 : mxxMat0) (w_625_0 : mxxMat6) (w_626_0 : mxxMat6) (w_628_0 : mxxMat0) (w_630_0 : mxxMat0) (w_631_0 : mxxMat6) (w_632_0 : mxxMat6) (w_633_0 : mxxMat0) (w_634_0 : mxxMat0) (w_635_0 : mxxMat0) (w_636_0 : Fin 26 → mxxMat0) : Prop :=
  MxxRuntime.sliceMatrix w_611_0 1 2 0 25 w_612_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_611_0 2 3 0 25 w_613_0 ∧
  MxxRuntime.concatRows w_612_0 w_613_0 w_614_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_614_0 w_615_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_615_0 w_616_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_617_0 0 1 0 25 w_618_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_611_0 0 1 0 25 w_619_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_622_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_624_0 w_625_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_625_0 w_626_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_628_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_630_0 w_631_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_631_0 w_632_0 ∧
  MxxRuntime.familyPack (26) [w_78_0, w_79_0, w_86_0, w_89_0, w_200_0, w_215_0, w_227_0, w_228_0, w_285_0, w_286_0, w_343_0, w_344_0, w_401_0, w_402_0, w_459_0, w_460_0, w_517_0, w_518_0, w_575_0, w_576_0, w_633_0, w_634_0, w_635_0, w_635_0, w_635_0, w_635_0] w_636_0

abbrev generatedRoot.constraints_25 (backend : MxxRuntime.BackendContext) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_588_7 : mxxMat0) (w_588_9 : mxxMat0) (w_599_0 : mxxMat0) (w_600_0 : mxxMat0) (w_601_0 : mxxMat0) (w_602_0 : mxxMat2) (w_602_concat_1 : mxxMat3) (w_603_0 : mxxMat0) (w_604_0 : mxxMat0) (w_605_0 : mxxMat0) (w_606_0 : mxxMat2) (w_606_concat_1 : mxxMat3) (w_607_0 : mxxMat4) (w_608_0 : mxxMat3) (w_609_0 : mxxMat5) (w_610_0 : mxxMat5) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_4 w_599_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_0 w_600_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_2 w_601_0 ∧
  MxxRuntime.concatRows w_599_0 w_600_0 w_602_concat_1 ∧
  MxxRuntime.concatRows w_602_concat_1 w_601_0 w_602_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_5 w_603_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_1 w_604_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_3 w_605_0 ∧
  MxxRuntime.concatRows w_603_0 w_604_0 w_606_concat_1 ∧
  MxxRuntime.concatRows w_606_concat_1 w_605_0 w_606_0 ∧
  MxxRuntime.concatColumns w_602_0 w_606_0 w_607_0 ∧
  MxxRuntime.concatRows w_588_7 w_588_9 w_608_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_608_0 w_609_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_609_0 w_610_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev generatedRoot.constraints_24 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_578_0 : Fin 1 → mxxMat0) (w_580_0 : Fin 1 → mxxMat0) (w_582_0 : Fin 1 → mxxMat0) (w_584_0 : Fin 1 → mxxMat0) (w_586_0 : Fin 1 → mxxMat0) (w_587_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_55_0 : mxxMat0) (w_72_0 : mxxMat0) (w_553_0 : mxxMat2) (w_561_0 : mxxMat0) (w_564_0 : mxxMat0) (w_566_0 : mxxMat0) (w_567_0 : mxxMat6) (w_568_0 : mxxMat6) (w_570_0 : mxxMat0) (w_572_0 : mxxMat0) (w_573_0 : mxxMat6) (w_574_0 : mxxMat6) (w_577_0 : mxxMat0) (w_579_0 : mxxMat0) (w_581_0 : mxxMat0) (w_583_0 : mxxMat0) (w_585_0 : mxxMat0) (w_588_0 : mxxMat0) (w_588_1 : mxxMat0) (w_588_2 : mxxMat0) (w_588_3 : mxxMat0) (w_588_4 : mxxMat0) (w_588_5 : mxxMat0) (w_588_6 : mxxMat0) (w_588_7 : mxxMat0) (w_588_8 : mxxMat0) (w_588_9 : mxxMat0) (w_589_0 : mxxMat0) (w_590_0 : Int) (w_591_0 : mxxMat0) (w_592_0 : mxxMat0) (w_593_0 : mxxMat2) (w_593_concat_1 : mxxMat3) (w_594_0 : mxxMat0) (w_595_0 : mxxMat0) (w_596_0 : mxxMat0) (w_597_0 : mxxMat2) (w_597_concat_1 : mxxMat3) (w_598_0 : mxxMat4) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_553_0 0 1 0 25 w_561_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_564_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_566_0 w_567_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_567_0 w_568_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_570_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_572_0 w_573_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_573_0 w_574_0 ∧
  MxxRuntime.familyGetStatic w_578_0 (0) w_579_0 ∧
  MxxRuntime.familyGetStatic w_580_0 (0) w_581_0 ∧
  MxxRuntime.familyGetStatic w_582_0 (0) w_583_0 ∧
  MxxRuntime.familyGetStatic w_584_0 (0) w_585_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_588 backend hashModel tape (path ++ [588, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_586_0, (w_5_0, (w_587_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_578_0, (w_580_0, (w_582_0, (w_584_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_577_0, (w_577_0, (w_577_0, (w_577_0, (w_577_0, (w_577_0, (w_579_0, (w_581_0, (w_583_0, (w_585_0, ())))))))))) (w_588_0, (w_588_1, (w_588_2, (w_588_3, (w_588_4, (w_588_5, (w_588_6, (w_588_7, (w_588_8, (w_588_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_586_0 (0) w_589_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_590_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_591_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_590_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_592_0 ∧
  MxxRuntime.concatRows w_589_0 w_591_0 w_593_concat_1 ∧
  MxxRuntime.concatRows w_593_concat_1 w_592_0 w_593_0 ∧
  MxxRuntime.familyGetStatic w_587_0 (0) w_594_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_590_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_595_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_590_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_596_0 ∧
  MxxRuntime.concatRows w_594_0 w_595_0 w_597_concat_1 ∧
  MxxRuntime.concatRows w_597_concat_1 w_596_0 w_597_0 ∧
  MxxRuntime.concatColumns w_593_0 w_597_0 w_598_0

abbrev generatedRoot.constraints_23 (backend : MxxRuntime.BackendContext) (w_54_3 : mxxMat0) (w_530_7 : mxxMat0) (w_530_9 : mxxMat0) (w_544_0 : mxxMat2) (w_545_0 : mxxMat0) (w_546_0 : mxxMat0) (w_547_0 : mxxMat0) (w_548_0 : mxxMat2) (w_548_concat_1 : mxxMat3) (w_549_0 : mxxMat4) (w_550_0 : mxxMat3) (w_551_0 : mxxMat5) (w_552_0 : mxxMat5) (w_553_0 : mxxMat2) (w_554_0 : mxxMat0) (w_555_0 : mxxMat0) (w_556_0 : mxxMat3) (w_557_0 : mxxMat5) (w_558_0 : mxxMat5) (w_559_0 : mxxMat2) (w_560_0 : mxxMat0) : Prop :=
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_3 w_547_0 ∧
  MxxRuntime.concatRows w_545_0 w_546_0 w_548_concat_1 ∧
  MxxRuntime.concatRows w_548_concat_1 w_547_0 w_548_0 ∧
  MxxRuntime.concatColumns w_544_0 w_548_0 w_549_0 ∧
  MxxRuntime.concatRows w_530_7 w_530_9 w_550_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_550_0 w_551_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_551_0 w_552_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_553_0 1 2 0 25 w_554_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_553_0 2 3 0 25 w_555_0 ∧
  MxxRuntime.concatRows w_554_0 w_555_0 w_556_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_556_0 w_557_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_557_0 w_558_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_559_0 0 1 0 25 w_560_0

abbrev generatedRoot.constraints_22 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_520_0 : Fin 1 → mxxMat0) (w_522_0 : Fin 1 → mxxMat0) (w_524_0 : Fin 1 → mxxMat0) (w_526_0 : Fin 1 → mxxMat0) (w_528_0 : Fin 1 → mxxMat0) (w_529_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_519_0 : mxxMat0) (w_521_0 : mxxMat0) (w_523_0 : mxxMat0) (w_525_0 : mxxMat0) (w_527_0 : mxxMat0) (w_530_0 : mxxMat0) (w_530_1 : mxxMat0) (w_530_2 : mxxMat0) (w_530_3 : mxxMat0) (w_530_4 : mxxMat0) (w_530_5 : mxxMat0) (w_530_6 : mxxMat0) (w_530_7 : mxxMat0) (w_530_8 : mxxMat0) (w_530_9 : mxxMat0) (w_531_0 : mxxMat0) (w_532_0 : Int) (w_533_0 : mxxMat0) (w_534_0 : mxxMat0) (w_535_0 : mxxMat2) (w_535_concat_1 : mxxMat3) (w_536_0 : mxxMat0) (w_537_0 : mxxMat0) (w_538_0 : mxxMat0) (w_539_0 : mxxMat2) (w_539_concat_1 : mxxMat3) (w_540_0 : mxxMat4) (w_541_0 : mxxMat0) (w_542_0 : mxxMat0) (w_543_0 : mxxMat0) (w_544_0 : mxxMat2) (w_544_concat_1 : mxxMat3) (w_545_0 : mxxMat0) (w_546_0 : mxxMat0) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_530 backend hashModel tape (path ++ [530, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_528_0, (w_5_0, (w_529_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_520_0, (w_522_0, (w_524_0, (w_526_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_519_0, (w_519_0, (w_519_0, (w_519_0, (w_519_0, (w_519_0, (w_521_0, (w_523_0, (w_525_0, (w_527_0, ())))))))))) (w_530_0, (w_530_1, (w_530_2, (w_530_3, (w_530_4, (w_530_5, (w_530_6, (w_530_7, (w_530_8, (w_530_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_528_0 (0) w_531_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_532_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_533_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_532_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_534_0 ∧
  MxxRuntime.concatRows w_531_0 w_533_0 w_535_concat_1 ∧
  MxxRuntime.concatRows w_535_concat_1 w_534_0 w_535_0 ∧
  MxxRuntime.familyGetStatic w_529_0 (0) w_536_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_532_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_537_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_532_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_538_0 ∧
  MxxRuntime.concatRows w_536_0 w_537_0 w_539_concat_1 ∧
  MxxRuntime.concatRows w_539_concat_1 w_538_0 w_539_0 ∧
  MxxRuntime.concatColumns w_535_0 w_539_0 w_540_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_4 w_541_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_0 w_542_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_2 w_543_0 ∧
  MxxRuntime.concatRows w_541_0 w_542_0 w_544_concat_1 ∧
  MxxRuntime.concatRows w_544_concat_1 w_543_0 w_544_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_5 w_545_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_1 w_546_0 ∧
  0 ≤ (0)

abbrev generatedRoot.constraints_21 (backend : MxxRuntime.BackendContext) (w_520_0 : Fin 1 → mxxMat0) (w_522_0 : Fin 1 → mxxMat0) (w_524_0 : Fin 1 → mxxMat0) (w_526_0 : Fin 1 → mxxMat0) (w_55_0 : mxxMat0) (w_72_0 : mxxMat0) (w_495_0 : mxxMat2) (w_496_0 : mxxMat0) (w_497_0 : mxxMat0) (w_498_0 : mxxMat3) (w_499_0 : mxxMat5) (w_500_0 : mxxMat5) (w_501_0 : mxxMat2) (w_502_0 : mxxMat0) (w_503_0 : mxxMat0) (w_506_0 : mxxMat0) (w_508_0 : mxxMat0) (w_509_0 : mxxMat6) (w_510_0 : mxxMat6) (w_512_0 : mxxMat0) (w_514_0 : mxxMat0) (w_515_0 : mxxMat6) (w_516_0 : mxxMat6) (w_521_0 : mxxMat0) (w_523_0 : mxxMat0) (w_525_0 : mxxMat0) (w_527_0 : mxxMat0) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_495_0 2 3 0 25 w_497_0 ∧
  MxxRuntime.concatRows w_496_0 w_497_0 w_498_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_498_0 w_499_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_499_0 w_500_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_501_0 0 1 0 25 w_502_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_495_0 0 1 0 25 w_503_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_506_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_508_0 w_509_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_509_0 w_510_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_512_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_514_0 w_515_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_515_0 w_516_0 ∧
  MxxRuntime.familyGetStatic w_520_0 (0) w_521_0 ∧
  MxxRuntime.familyGetStatic w_522_0 (0) w_523_0 ∧
  MxxRuntime.familyGetStatic w_524_0 (0) w_525_0 ∧
  MxxRuntime.familyGetStatic w_526_0 (0) w_527_0 ∧
  0 ≤ (0)

abbrev generatedRoot.constraints_20 (backend : MxxRuntime.BackendContext) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_5 : mxxMat0) (w_472_7 : mxxMat0) (w_472_9 : mxxMat0) (w_483_0 : mxxMat0) (w_484_0 : mxxMat0) (w_485_0 : mxxMat0) (w_486_0 : mxxMat2) (w_486_concat_1 : mxxMat3) (w_487_0 : mxxMat0) (w_488_0 : mxxMat0) (w_489_0 : mxxMat0) (w_490_0 : mxxMat2) (w_490_concat_1 : mxxMat3) (w_491_0 : mxxMat4) (w_492_0 : mxxMat3) (w_493_0 : mxxMat5) (w_494_0 : mxxMat5) (w_495_0 : mxxMat2) (w_496_0 : mxxMat0) : Prop :=
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_0 w_484_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_2 w_485_0 ∧
  MxxRuntime.concatRows w_483_0 w_484_0 w_486_concat_1 ∧
  MxxRuntime.concatRows w_486_concat_1 w_485_0 w_486_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_5 w_487_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_1 w_488_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_3 w_489_0 ∧
  MxxRuntime.concatRows w_487_0 w_488_0 w_490_concat_1 ∧
  MxxRuntime.concatRows w_490_concat_1 w_489_0 w_490_0 ∧
  MxxRuntime.concatColumns w_486_0 w_490_0 w_491_0 ∧
  MxxRuntime.concatRows w_472_7 w_472_9 w_492_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_492_0 w_493_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_493_0 w_494_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_495_0 1 2 0 25 w_496_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3

abbrev generatedRoot.constraints_19 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_462_0 : Fin 1 → mxxMat0) (w_464_0 : Fin 1 → mxxMat0) (w_466_0 : Fin 1 → mxxMat0) (w_468_0 : Fin 1 → mxxMat0) (w_470_0 : Fin 1 → mxxMat0) (w_471_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_55_0 : mxxMat0) (w_72_0 : mxxMat0) (w_437_0 : mxxMat2) (w_445_0 : mxxMat0) (w_448_0 : mxxMat0) (w_450_0 : mxxMat0) (w_451_0 : mxxMat6) (w_452_0 : mxxMat6) (w_454_0 : mxxMat0) (w_456_0 : mxxMat0) (w_457_0 : mxxMat6) (w_458_0 : mxxMat6) (w_461_0 : mxxMat0) (w_463_0 : mxxMat0) (w_465_0 : mxxMat0) (w_467_0 : mxxMat0) (w_469_0 : mxxMat0) (w_472_0 : mxxMat0) (w_472_1 : mxxMat0) (w_472_2 : mxxMat0) (w_472_3 : mxxMat0) (w_472_4 : mxxMat0) (w_472_5 : mxxMat0) (w_472_6 : mxxMat0) (w_472_7 : mxxMat0) (w_472_8 : mxxMat0) (w_472_9 : mxxMat0) (w_473_0 : mxxMat0) (w_474_0 : Int) (w_475_0 : mxxMat0) (w_476_0 : mxxMat0) (w_477_0 : mxxMat2) (w_477_concat_1 : mxxMat3) (w_478_0 : mxxMat0) (w_479_0 : mxxMat0) (w_480_0 : mxxMat0) (w_481_0 : mxxMat2) (w_481_concat_1 : mxxMat3) (w_482_0 : mxxMat4) (w_483_0 : mxxMat0) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_437_0 0 1 0 25 w_445_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_448_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_450_0 w_451_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_451_0 w_452_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_454_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_456_0 w_457_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_457_0 w_458_0 ∧
  MxxRuntime.familyGetStatic w_462_0 (0) w_463_0 ∧
  MxxRuntime.familyGetStatic w_464_0 (0) w_465_0 ∧
  MxxRuntime.familyGetStatic w_466_0 (0) w_467_0 ∧
  MxxRuntime.familyGetStatic w_468_0 (0) w_469_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_472 backend hashModel tape (path ++ [472, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_470_0, (w_5_0, (w_471_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_462_0, (w_464_0, (w_466_0, (w_468_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_461_0, (w_461_0, (w_461_0, (w_461_0, (w_461_0, (w_461_0, (w_463_0, (w_465_0, (w_467_0, (w_469_0, ())))))))))) (w_472_0, (w_472_1, (w_472_2, (w_472_3, (w_472_4, (w_472_5, (w_472_6, (w_472_7, (w_472_8, (w_472_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_470_0 (0) w_473_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_474_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_475_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_474_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_476_0 ∧
  MxxRuntime.concatRows w_473_0 w_475_0 w_477_concat_1 ∧
  MxxRuntime.concatRows w_477_concat_1 w_476_0 w_477_0 ∧
  MxxRuntime.familyGetStatic w_471_0 (0) w_478_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_474_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_479_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_474_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_480_0 ∧
  MxxRuntime.concatRows w_478_0 w_479_0 w_481_concat_1 ∧
  MxxRuntime.concatRows w_481_concat_1 w_480_0 w_481_0 ∧
  MxxRuntime.concatColumns w_477_0 w_481_0 w_482_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_4 w_483_0 ∧
  0 ≤ (0)

abbrev generatedRoot.constraints_18 (backend : MxxRuntime.BackendContext) (w_414_7 : mxxMat0) (w_414_9 : mxxMat0) (w_428_0 : mxxMat2) (w_432_0 : mxxMat2) (w_433_0 : mxxMat4) (w_434_0 : mxxMat3) (w_435_0 : mxxMat5) (w_436_0 : mxxMat5) (w_437_0 : mxxMat2) (w_438_0 : mxxMat0) (w_439_0 : mxxMat0) (w_440_0 : mxxMat3) (w_441_0 : mxxMat5) (w_442_0 : mxxMat5) (w_443_0 : mxxMat2) (w_444_0 : mxxMat0) : Prop :=
  MxxRuntime.concatColumns w_428_0 w_432_0 w_433_0 ∧
  MxxRuntime.concatRows w_414_7 w_414_9 w_434_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_434_0 w_435_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_435_0 w_436_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_437_0 1 2 0 25 w_438_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_437_0 2 3 0 25 w_439_0 ∧
  MxxRuntime.concatRows w_438_0 w_439_0 w_440_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_440_0 w_441_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_441_0 w_442_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_443_0 0 1 0 25 w_444_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_17 (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_413_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_415_0 : mxxMat0) (w_416_0 : Int) (w_417_0 : mxxMat0) (w_418_0 : mxxMat0) (w_419_0 : mxxMat2) (w_419_concat_1 : mxxMat3) (w_420_0 : mxxMat0) (w_421_0 : mxxMat0) (w_422_0 : mxxMat0) (w_423_0 : mxxMat2) (w_423_concat_1 : mxxMat3) (w_424_0 : mxxMat4) (w_425_0 : mxxMat0) (w_426_0 : mxxMat0) (w_427_0 : mxxMat0) (w_428_0 : mxxMat2) (w_428_concat_1 : mxxMat3) (w_429_0 : mxxMat0) (w_430_0 : mxxMat0) (w_431_0 : mxxMat0) (w_432_0 : mxxMat2) (w_432_concat_1 : mxxMat3) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_416_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_417_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_416_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_418_0 ∧
  MxxRuntime.concatRows w_415_0 w_417_0 w_419_concat_1 ∧
  MxxRuntime.concatRows w_419_concat_1 w_418_0 w_419_0 ∧
  MxxRuntime.familyGetStatic w_413_0 (0) w_420_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_416_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_421_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_416_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_422_0 ∧
  MxxRuntime.concatRows w_420_0 w_421_0 w_423_concat_1 ∧
  MxxRuntime.concatRows w_423_concat_1 w_422_0 w_423_0 ∧
  MxxRuntime.concatColumns w_419_0 w_423_0 w_424_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_4 w_425_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_0 w_426_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_2 w_427_0 ∧
  MxxRuntime.concatRows w_425_0 w_426_0 w_428_concat_1 ∧
  MxxRuntime.concatRows w_428_concat_1 w_427_0 w_428_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_5 w_429_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_1 w_430_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_3 w_431_0 ∧
  MxxRuntime.concatRows w_429_0 w_430_0 w_432_concat_1 ∧
  MxxRuntime.concatRows w_432_concat_1 w_431_0 w_432_0

abbrev generatedRoot.constraints_16 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_404_0 : Fin 1 → mxxMat0) (w_406_0 : Fin 1 → mxxMat0) (w_408_0 : Fin 1 → mxxMat0) (w_410_0 : Fin 1 → mxxMat0) (w_412_0 : Fin 1 → mxxMat0) (w_413_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_55_0 : mxxMat0) (w_72_0 : mxxMat0) (w_379_0 : mxxMat2) (w_380_0 : mxxMat0) (w_381_0 : mxxMat0) (w_382_0 : mxxMat3) (w_383_0 : mxxMat5) (w_384_0 : mxxMat5) (w_385_0 : mxxMat2) (w_386_0 : mxxMat0) (w_387_0 : mxxMat0) (w_390_0 : mxxMat0) (w_392_0 : mxxMat0) (w_393_0 : mxxMat6) (w_394_0 : mxxMat6) (w_396_0 : mxxMat0) (w_398_0 : mxxMat0) (w_399_0 : mxxMat6) (w_400_0 : mxxMat6) (w_403_0 : mxxMat0) (w_405_0 : mxxMat0) (w_407_0 : mxxMat0) (w_409_0 : mxxMat0) (w_411_0 : mxxMat0) (w_414_0 : mxxMat0) (w_414_1 : mxxMat0) (w_414_2 : mxxMat0) (w_414_3 : mxxMat0) (w_414_4 : mxxMat0) (w_414_5 : mxxMat0) (w_414_6 : mxxMat0) (w_414_7 : mxxMat0) (w_414_8 : mxxMat0) (w_414_9 : mxxMat0) (w_415_0 : mxxMat0) : Prop :=
  MxxRuntime.concatRows w_380_0 w_381_0 w_382_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_382_0 w_383_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_383_0 w_384_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_385_0 0 1 0 25 w_386_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_379_0 0 1 0 25 w_387_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_390_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_392_0 w_393_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_393_0 w_394_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_396_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_398_0 w_399_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_399_0 w_400_0 ∧
  MxxRuntime.familyGetStatic w_404_0 (0) w_405_0 ∧
  MxxRuntime.familyGetStatic w_406_0 (0) w_407_0 ∧
  MxxRuntime.familyGetStatic w_408_0 (0) w_409_0 ∧
  MxxRuntime.familyGetStatic w_410_0 (0) w_411_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_414 backend hashModel tape (path ++ [414, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_412_0, (w_5_0, (w_413_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_404_0, (w_406_0, (w_408_0, (w_410_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_403_0, (w_403_0, (w_403_0, (w_403_0, (w_403_0, (w_403_0, (w_405_0, (w_407_0, (w_409_0, (w_411_0, ())))))))))) (w_414_0, (w_414_1, (w_414_2, (w_414_3, (w_414_4, (w_414_5, (w_414_6, (w_414_7, (w_414_8, (w_414_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_412_0 (0) w_415_0 ∧
  0 ≤ (0) ∧
  (0) < 1

abbrev generatedRoot.constraints_15 (backend : MxxRuntime.BackendContext) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_5 : mxxMat0) (w_356_7 : mxxMat0) (w_356_9 : mxxMat0) (w_367_0 : mxxMat0) (w_368_0 : mxxMat0) (w_369_0 : mxxMat0) (w_370_0 : mxxMat2) (w_370_concat_1 : mxxMat3) (w_371_0 : mxxMat0) (w_372_0 : mxxMat0) (w_373_0 : mxxMat0) (w_374_0 : mxxMat2) (w_374_concat_1 : mxxMat3) (w_375_0 : mxxMat4) (w_376_0 : mxxMat3) (w_377_0 : mxxMat5) (w_378_0 : mxxMat5) (w_379_0 : mxxMat2) (w_380_0 : mxxMat0) (w_381_0 : mxxMat0) : Prop :=
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_2 w_369_0 ∧
  MxxRuntime.concatRows w_367_0 w_368_0 w_370_concat_1 ∧
  MxxRuntime.concatRows w_370_concat_1 w_369_0 w_370_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_5 w_371_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_1 w_372_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_3 w_373_0 ∧
  MxxRuntime.concatRows w_371_0 w_372_0 w_374_concat_1 ∧
  MxxRuntime.concatRows w_374_concat_1 w_373_0 w_374_0 ∧
  MxxRuntime.concatColumns w_370_0 w_374_0 w_375_0 ∧
  MxxRuntime.concatRows w_356_7 w_356_9 w_376_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_376_0 w_377_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_377_0 w_378_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_379_0 1 2 0 25 w_380_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_379_0 2 3 0 25 w_381_0

abbrev generatedRoot.constraints_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_346_0 : Fin 1 → mxxMat0) (w_348_0 : Fin 1 → mxxMat0) (w_350_0 : Fin 1 → mxxMat0) (w_352_0 : Fin 1 → mxxMat0) (w_354_0 : Fin 1 → mxxMat0) (w_355_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_72_0 : mxxMat0) (w_334_0 : mxxMat0) (w_335_0 : mxxMat6) (w_336_0 : mxxMat6) (w_338_0 : mxxMat0) (w_340_0 : mxxMat0) (w_341_0 : mxxMat6) (w_342_0 : mxxMat6) (w_345_0 : mxxMat0) (w_347_0 : mxxMat0) (w_349_0 : mxxMat0) (w_351_0 : mxxMat0) (w_353_0 : mxxMat0) (w_356_0 : mxxMat0) (w_356_1 : mxxMat0) (w_356_2 : mxxMat0) (w_356_3 : mxxMat0) (w_356_4 : mxxMat0) (w_356_5 : mxxMat0) (w_356_6 : mxxMat0) (w_356_7 : mxxMat0) (w_356_8 : mxxMat0) (w_356_9 : mxxMat0) (w_357_0 : mxxMat0) (w_358_0 : Int) (w_359_0 : mxxMat0) (w_360_0 : mxxMat0) (w_361_0 : mxxMat2) (w_361_concat_1 : mxxMat3) (w_362_0 : mxxMat0) (w_363_0 : mxxMat0) (w_364_0 : mxxMat0) (w_365_0 : mxxMat2) (w_365_concat_1 : mxxMat3) (w_366_0 : mxxMat4) (w_367_0 : mxxMat0) (w_368_0 : mxxMat0) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_334_0 w_335_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_335_0 w_336_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_338_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_340_0 w_341_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_341_0 w_342_0 ∧
  MxxRuntime.familyGetStatic w_346_0 (0) w_347_0 ∧
  MxxRuntime.familyGetStatic w_348_0 (0) w_349_0 ∧
  MxxRuntime.familyGetStatic w_350_0 (0) w_351_0 ∧
  MxxRuntime.familyGetStatic w_352_0 (0) w_353_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_356 backend hashModel tape (path ++ [356, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_354_0, (w_5_0, (w_355_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_346_0, (w_348_0, (w_350_0, (w_352_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_345_0, (w_345_0, (w_345_0, (w_345_0, (w_345_0, (w_345_0, (w_347_0, (w_349_0, (w_351_0, (w_353_0, ())))))))))) (w_356_0, (w_356_1, (w_356_2, (w_356_3, (w_356_4, (w_356_5, (w_356_6, (w_356_7, (w_356_8, (w_356_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_354_0 (0) w_357_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_358_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_359_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_358_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_360_0 ∧
  MxxRuntime.concatRows w_357_0 w_359_0 w_361_concat_1 ∧
  MxxRuntime.concatRows w_361_concat_1 w_360_0 w_361_0 ∧
  MxxRuntime.familyGetStatic w_355_0 (0) w_362_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_358_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_363_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_358_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_364_0 ∧
  MxxRuntime.concatRows w_362_0 w_363_0 w_365_concat_1 ∧
  MxxRuntime.concatRows w_365_concat_1 w_364_0 w_365_0 ∧
  MxxRuntime.concatColumns w_361_0 w_365_0 w_366_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_4 w_367_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_0 w_368_0 ∧
  0 ≤ (0) ∧
  (0) < 1

abbrev generatedRoot.constraints_13 (backend : MxxRuntime.BackendContext) (w_55_0 : mxxMat0) (w_321_0 : mxxMat2) (w_322_0 : mxxMat0) (w_323_0 : mxxMat0) (w_324_0 : mxxMat3) (w_325_0 : mxxMat5) (w_326_0 : mxxMat5) (w_327_0 : mxxMat2) (w_328_0 : mxxMat0) (w_329_0 : mxxMat0) (w_332_0 : mxxMat0) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_321_0 1 2 0 25 w_322_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_321_0 2 3 0 25 w_323_0 ∧
  MxxRuntime.concatRows w_322_0 w_323_0 w_324_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_324_0 w_325_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_325_0 w_326_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_327_0 0 1 0 25 w_328_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_321_0 0 1 0 25 w_329_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_332_0

abbrev generatedRoot.constraints_12 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_297_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_298_7 : mxxMat0) (w_298_9 : mxxMat0) (w_300_0 : Int) (w_303_0 : mxxMat2) (w_304_0 : mxxMat0) (w_305_0 : mxxMat0) (w_306_0 : mxxMat0) (w_307_0 : mxxMat2) (w_307_concat_1 : mxxMat3) (w_308_0 : mxxMat4) (w_309_0 : mxxMat0) (w_310_0 : mxxMat0) (w_311_0 : mxxMat0) (w_312_0 : mxxMat2) (w_312_concat_1 : mxxMat3) (w_313_0 : mxxMat0) (w_314_0 : mxxMat0) (w_315_0 : mxxMat0) (w_316_0 : mxxMat2) (w_316_concat_1 : mxxMat3) (w_317_0 : mxxMat4) (w_318_0 : mxxMat3) (w_319_0 : mxxMat5) (w_320_0 : mxxMat5) : Prop :=
  MxxRuntime.familyGetStatic w_297_0 (0) w_304_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_300_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_305_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_300_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_306_0 ∧
  MxxRuntime.concatRows w_304_0 w_305_0 w_307_concat_1 ∧
  MxxRuntime.concatRows w_307_concat_1 w_306_0 w_307_0 ∧
  MxxRuntime.concatColumns w_303_0 w_307_0 w_308_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_4 w_309_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_0 w_310_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_2 w_311_0 ∧
  MxxRuntime.concatRows w_309_0 w_310_0 w_312_concat_1 ∧
  MxxRuntime.concatRows w_312_concat_1 w_311_0 w_312_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_5 w_313_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_1 w_314_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_3 w_315_0 ∧
  MxxRuntime.concatRows w_313_0 w_314_0 w_316_concat_1 ∧
  MxxRuntime.concatRows w_316_concat_1 w_315_0 w_316_0 ∧
  MxxRuntime.concatColumns w_312_0 w_316_0 w_317_0 ∧
  MxxRuntime.concatRows w_298_7 w_298_9 w_318_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_318_0 w_319_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_319_0 w_320_0

abbrev generatedRoot.constraints_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_288_0 : Fin 1 → mxxMat0) (w_290_0 : Fin 1 → mxxMat0) (w_292_0 : Fin 1 → mxxMat0) (w_294_0 : Fin 1 → mxxMat0) (w_296_0 : Fin 1 → mxxMat0) (w_297_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_55_0 : mxxMat0) (w_72_0 : mxxMat0) (w_263_0 : mxxMat2) (w_269_0 : mxxMat2) (w_270_0 : mxxMat0) (w_271_0 : mxxMat0) (w_274_0 : mxxMat0) (w_276_0 : mxxMat0) (w_277_0 : mxxMat6) (w_278_0 : mxxMat6) (w_280_0 : mxxMat0) (w_282_0 : mxxMat0) (w_283_0 : mxxMat6) (w_284_0 : mxxMat6) (w_287_0 : mxxMat0) (w_289_0 : mxxMat0) (w_291_0 : mxxMat0) (w_293_0 : mxxMat0) (w_295_0 : mxxMat0) (w_298_0 : mxxMat0) (w_298_1 : mxxMat0) (w_298_2 : mxxMat0) (w_298_3 : mxxMat0) (w_298_4 : mxxMat0) (w_298_5 : mxxMat0) (w_298_6 : mxxMat0) (w_298_7 : mxxMat0) (w_298_8 : mxxMat0) (w_298_9 : mxxMat0) (w_299_0 : mxxMat0) (w_300_0 : Int) (w_301_0 : mxxMat0) (w_302_0 : mxxMat0) (w_303_0 : mxxMat2) (w_303_concat_1 : mxxMat3) : Prop :=
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_269_0 0 1 0 25 w_270_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_263_0 0 1 0 25 w_271_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_274_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_276_0 w_277_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_277_0 w_278_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_280_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_282_0 w_283_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_283_0 w_284_0 ∧
  MxxRuntime.familyGetStatic w_288_0 (0) w_289_0 ∧
  MxxRuntime.familyGetStatic w_290_0 (0) w_291_0 ∧
  MxxRuntime.familyGetStatic w_292_0 (0) w_293_0 ∧
  MxxRuntime.familyGetStatic w_294_0 (0) w_295_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_298 backend hashModel tape (path ++ [298, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_296_0, (w_5_0, (w_297_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_288_0, (w_290_0, (w_292_0, (w_294_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_287_0, (w_287_0, (w_287_0, (w_287_0, (w_287_0, (w_287_0, (w_289_0, (w_291_0, (w_293_0, (w_295_0, ())))))))))) (w_298_0, (w_298_1, (w_298_2, (w_298_3, (w_298_4, (w_298_5, (w_298_6, (w_298_7, (w_298_8, (w_298_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_296_0 (0) w_299_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_300_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_301_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_300_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_302_0 ∧
  MxxRuntime.concatRows w_299_0 w_301_0 w_303_concat_1 ∧
  MxxRuntime.concatRows w_303_concat_1 w_302_0 w_303_0

abbrev generatedRoot.constraints_10 (backend : MxxRuntime.BackendContext) (w_54_1 : mxxMat0) (w_54_3 : mxxMat0) (w_54_5 : mxxMat0) (w_240_7 : mxxMat0) (w_240_9 : mxxMat0) (w_254_0 : mxxMat2) (w_255_0 : mxxMat0) (w_256_0 : mxxMat0) (w_257_0 : mxxMat0) (w_258_0 : mxxMat2) (w_258_concat_1 : mxxMat3) (w_259_0 : mxxMat4) (w_260_0 : mxxMat3) (w_261_0 : mxxMat5) (w_262_0 : mxxMat5) (w_263_0 : mxxMat2) (w_264_0 : mxxMat0) (w_265_0 : mxxMat0) (w_266_0 : mxxMat3) (w_267_0 : mxxMat5) (w_268_0 : mxxMat5) : Prop :=
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_5 w_255_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_1 w_256_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_3 w_257_0 ∧
  MxxRuntime.concatRows w_255_0 w_256_0 w_258_concat_1 ∧
  MxxRuntime.concatRows w_258_concat_1 w_257_0 w_258_0 ∧
  MxxRuntime.concatColumns w_254_0 w_258_0 w_259_0 ∧
  MxxRuntime.concatRows w_240_7 w_240_9 w_260_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_260_0 w_261_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_261_0 w_262_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_263_0 1 2 0 25 w_264_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_263_0 2 3 0 25 w_265_0 ∧
  MxxRuntime.concatRows w_264_0 w_265_0 w_266_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_266_0 w_267_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_267_0 w_268_0 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_9 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_230_0 : Fin 1 → mxxMat0) (w_232_0 : Fin 1 → mxxMat0) (w_234_0 : Fin 1 → mxxMat0) (w_236_0 : Fin 1 → mxxMat0) (w_238_0 : Fin 1 → mxxMat0) (w_239_0 : Fin 1 → mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) (w_225_0 : mxxMat6) (w_226_0 : mxxMat6) (w_229_0 : mxxMat0) (w_231_0 : mxxMat0) (w_233_0 : mxxMat0) (w_235_0 : mxxMat0) (w_237_0 : mxxMat0) (w_240_0 : mxxMat0) (w_240_1 : mxxMat0) (w_240_2 : mxxMat0) (w_240_3 : mxxMat0) (w_240_4 : mxxMat0) (w_240_5 : mxxMat0) (w_240_6 : mxxMat0) (w_240_7 : mxxMat0) (w_240_8 : mxxMat0) (w_240_9 : mxxMat0) (w_241_0 : mxxMat0) (w_242_0 : Int) (w_243_0 : mxxMat0) (w_244_0 : mxxMat0) (w_245_0 : mxxMat2) (w_245_concat_1 : mxxMat3) (w_246_0 : mxxMat0) (w_247_0 : mxxMat0) (w_248_0 : mxxMat0) (w_249_0 : mxxMat2) (w_249_concat_1 : mxxMat3) (w_250_0 : mxxMat4) (w_251_0 : mxxMat0) (w_252_0 : mxxMat0) (w_253_0 : mxxMat0) (w_254_0 : mxxMat2) (w_254_concat_1 : mxxMat3) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 2048 w_225_0 w_226_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (0) w_231_0 ∧
  MxxRuntime.familyGetStatic w_232_0 (0) w_233_0 ∧
  MxxRuntime.familyGetStatic w_234_0 (0) w_235_0 ∧
  MxxRuntime.familyGetStatic w_236_0 (0) w_237_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_240 backend hashModel tape (path ++ [240, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_238_0, (w_5_0, (w_239_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_230_0, (w_232_0, (w_234_0, (w_236_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_229_0, (w_229_0, (w_229_0, (w_229_0, (w_229_0, (w_229_0, (w_231_0, (w_233_0, (w_235_0, (w_237_0, ())))))))))) (w_240_0, (w_240_1, (w_240_2, (w_240_3, (w_240_4, (w_240_5, (w_240_6, (w_240_7, (w_240_8, (w_240_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_238_0 (0) w_241_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_242_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_243_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_242_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_244_0 ∧
  MxxRuntime.concatRows w_241_0 w_243_0 w_245_concat_1 ∧
  MxxRuntime.concatRows w_245_concat_1 w_244_0 w_245_0 ∧
  MxxRuntime.familyGetStatic w_239_0 (0) w_246_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_242_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_247_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_242_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_248_0 ∧
  MxxRuntime.concatRows w_246_0 w_247_0 w_249_concat_1 ∧
  MxxRuntime.concatRows w_249_concat_1 w_248_0 w_249_0 ∧
  MxxRuntime.concatColumns w_245_0 w_249_0 w_250_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_4 w_251_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_0 w_252_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_2 w_253_0 ∧
  MxxRuntime.concatRows w_251_0 w_252_0 w_254_concat_1 ∧
  MxxRuntime.concatRows w_254_concat_1 w_253_0 w_254_0 ∧
  0 ≤ (0)

abbrev generatedRoot.constraints_8 (backend : MxxRuntime.BackendContext) (w_55_0 : mxxMat0) (w_72_0 : mxxMat0) (w_204_0 : mxxMat2) (w_205_0 : mxxMat0) (w_206_0 : mxxMat0) (w_207_0 : mxxMat3) (w_208_0 : mxxMat5) (w_209_0 : mxxMat5) (w_210_0 : mxxMat2) (w_211_0 : mxxMat0) (w_212_0 : mxxMat0) (w_216_0 : mxxMat0) (w_218_0 : mxxMat0) (w_219_0 : mxxMat6) (w_220_0 : mxxMat6) (w_222_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat6) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_204_0 1 2 0 25 w_205_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_204_0 2 3 0 25 w_206_0 ∧
  MxxRuntime.concatRows w_205_0 w_206_0 w_207_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_207_0 w_208_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_208_0 w_209_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_210_0 0 1 0 25 w_211_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_204_0 0 1 0 25 w_212_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_216_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_218_0 w_219_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_219_0 w_220_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_222_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_224_0 w_225_0

abbrev generatedRoot.constraints_7 (backend : MxxRuntime.BackendContext) (w_72_0 : mxxMat0) (w_161_7 : mxxMat0) (w_161_9 : mxxMat0) (w_184_0 : mxxMat2) (w_185_0 : mxxMat0) (w_186_0 : mxxMat0) (w_187_0 : mxxMat3) (w_188_0 : mxxMat5) (w_189_0 : mxxMat5) (w_190_0 : mxxMat2) (w_191_0 : mxxMat0) (w_192_0 : mxxMat0) (w_195_0 : mxxMat0) (w_197_0 : mxxMat0) (w_198_0 : mxxMat6) (w_199_0 : mxxMat6) (w_201_0 : mxxMat3) (w_202_0 : mxxMat5) (w_203_0 : mxxMat5) : Prop :=
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_184_0 2 3 0 25 w_186_0 ∧
  MxxRuntime.concatRows w_185_0 w_186_0 w_187_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_187_0 w_188_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_188_0 w_189_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_190_0 0 1 0 25 w_191_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_184_0 0 1 0 25 w_192_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_195_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_197_0 w_198_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_198_0 w_199_0 ∧
  MxxRuntime.concatRows w_161_7 w_161_9 w_201_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_201_0 w_202_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_202_0 w_203_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_6 (backend : MxxRuntime.BackendContext) (w_149_0 : mxxMat0) (w_150_0 : mxxMat0) (w_158_0 : mxxMat0) (w_159_0 : mxxMat0) (w_160_0 : mxxMat0) (w_161_6 : mxxMat0) (w_161_8 : mxxMat0) (w_172_0 : mxxMat0) (w_173_0 : mxxMat0) (w_174_0 : mxxMat0) (w_175_0 : mxxMat2) (w_175_concat_1 : mxxMat3) (w_176_0 : mxxMat0) (w_177_0 : mxxMat0) (w_178_0 : mxxMat0) (w_179_0 : mxxMat2) (w_179_concat_1 : mxxMat3) (w_180_0 : mxxMat4) (w_181_0 : mxxMat3) (w_182_0 : mxxMat5) (w_183_0 : mxxMat5) (w_184_0 : mxxMat2) (w_185_0 : mxxMat0) : Prop :=
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_149_0 w_173_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_150_0 w_174_0 ∧
  MxxRuntime.concatRows w_172_0 w_173_0 w_175_concat_1 ∧
  MxxRuntime.concatRows w_175_concat_1 w_174_0 w_175_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_158_0 w_176_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_159_0 w_177_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_160_0 w_178_0 ∧
  MxxRuntime.concatRows w_176_0 w_177_0 w_179_concat_1 ∧
  MxxRuntime.concatRows w_179_concat_1 w_178_0 w_179_0 ∧
  MxxRuntime.concatColumns w_175_0 w_179_0 w_180_0 ∧
  MxxRuntime.concatRows w_161_6 w_161_8 w_181_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_181_0 w_182_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_182_0 w_183_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_184_0 1 2 0 25 w_185_0 ∧
  0 ≤ 2 ∧
  2 < 3

abbrev generatedRoot.constraints_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_5_0 : ByteArray) (w_91_0 : Fin 4 → mxxMat0) (w_93_0 : Fin 4 → mxxMat0) (w_95_0 : Fin 4 → mxxMat0) (w_97_0 : Fin 4 → mxxMat0) (w_99_0 : Fin 4 → mxxMat0) (w_100_0 : Fin 4 → mxxMat0) (w_90_0 : mxxMat0) (w_92_0 : mxxMat0) (w_94_0 : mxxMat0) (w_96_0 : mxxMat0) (w_98_0 : mxxMat0) (w_148_0 : mxxMat0) (w_149_0 : mxxMat0) (w_150_0 : mxxMat0) (w_157_0 : mxxMat2) (w_158_0 : mxxMat0) (w_159_0 : mxxMat0) (w_160_0 : mxxMat0) (w_161_0 : mxxMat0) (w_161_1 : mxxMat0) (w_161_2 : mxxMat0) (w_161_3 : mxxMat0) (w_161_4 : mxxMat0) (w_161_5 : mxxMat0) (w_161_6 : mxxMat0) (w_161_7 : mxxMat0) (w_161_8 : mxxMat0) (w_161_9 : mxxMat0) (w_162_0 : mxxMat0) (w_163_0 : Int) (w_164_0 : mxxMat0) (w_165_0 : mxxMat0) (w_166_0 : mxxMat2) (w_166_concat_1 : mxxMat3) (w_167_0 : mxxMat0) (w_168_0 : mxxMat0) (w_169_0 : mxxMat0) (w_170_0 : mxxMat2) (w_170_concat_1 : mxxMat3) (w_171_0 : mxxMat4) (w_172_0 : mxxMat0) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_157_0 1 2 0 25 w_159_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_157_0 2 3 0 25 w_160_0 ∧
  0 ≤ (3) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_161 backend hashModel tape (path ++ [161, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_99_0, (w_5_0, (w_100_0, (w_148_0, (w_149_0, (w_150_0, (w_158_0, (w_159_0, (w_160_0, (w_91_0, (w_93_0, (w_95_0, (w_97_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_90_0, (w_90_0, (w_90_0, (w_90_0, (w_90_0, (w_90_0, (w_92_0, (w_94_0, (w_96_0, (w_98_0, ())))))))))) (w_161_0, (w_161_1, (w_161_2, (w_161_3, (w_161_4, (w_161_5, (w_161_6, (w_161_7, (w_161_8, (w_161_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_99_0 (3) w_162_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_163_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_164_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_163_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_165_0 ∧
  MxxRuntime.concatRows w_162_0 w_164_0 w_166_concat_1 ∧
  MxxRuntime.concatRows w_166_concat_1 w_165_0 w_166_0 ∧
  MxxRuntime.familyGetStatic w_100_0 (3) w_167_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_163_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_168_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_163_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_169_0 ∧
  MxxRuntime.concatRows w_167_0 w_168_0 w_170_concat_1 ∧
  MxxRuntime.concatRows w_170_concat_1 w_169_0 w_170_0 ∧
  MxxRuntime.concatColumns w_166_0 w_170_0 w_171_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_148_0 w_172_0

abbrev generatedRoot.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_24_0 : ByteArray) (w_152_0 : Fin 1 → mxxMat0) (w_147_0 : mxxMat2) (w_148_0 : mxxMat0) (w_149_0 : mxxMat0) (w_150_0 : mxxMat0) (w_151_0 : mxxMat0) (w_153_0 : mxxMat0) (w_154_0 : mxxMat3) (w_155_0 : mxxMat5) (w_156_0 : mxxMat5) (w_157_0 : mxxMat2) (w_158_0 : mxxMat0) : Prop :=
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_147_0 0 1 0 25 w_148_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_147_0 1 2 0 25 w_149_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_147_0 2 3 0 25 w_150_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_24_0) w_151_0 ∧
  MxxRuntime.familyGetStatic w_152_0 (0) w_153_0 ∧
  MxxRuntime.concatRows w_151_0 w_153_0 w_154_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_154_0 w_155_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_155_0 w_156_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_157_0 0 1 0 25 w_158_0

abbrev generatedRoot.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_24_0 : ByteArray) (w_142_0 : Fin 1 → mxxMat0) (w_126_0 : mxxMat2) (w_129_0 : mxxMat3) (w_130_0 : mxxMat5) (w_131_0 : mxxMat5) (w_132_0 : mxxMat2) (w_133_0 : mxxMat0) (w_136_0 : mxxMat0) (w_137_0 : mxxMat0) (w_138_0 : mxxMat0) (w_139_0 : mxxMat2) (w_139_concat_1 : mxxMat3) (w_140_0 : mxxMat4) (w_141_0 : mxxMat0) (w_143_0 : mxxMat0) (w_144_0 : mxxMat3) (w_145_0 : mxxMat5) (w_146_0 : mxxMat5) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_129_0 w_130_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_130_0 w_131_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_132_0 0 1 0 25 w_133_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_132_0 1 2 0 25 w_137_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_132_0 2 3 0 25 w_138_0 ∧
  MxxRuntime.concatRows w_136_0 w_137_0 w_139_concat_1 ∧
  MxxRuntime.concatRows w_139_concat_1 w_138_0 w_139_0 ∧
  MxxRuntime.concatColumns w_126_0 w_139_0 w_140_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_24_0) w_141_0 ∧
  MxxRuntime.familyGetStatic w_142_0 (0) w_143_0 ∧
  MxxRuntime.concatRows w_141_0 w_143_0 w_144_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_144_0 w_145_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_145_0 w_146_0 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_120_0 : Fin 2 → mxxMat1) (w_112_0 : mxxMat0) (w_113_0 : mxxMat0) (w_114_0 : mxxMat3) (w_115_0 : mxxMat5) (w_116_0 : mxxMat5) (w_117_0 : mxxMat2) (w_118_0 : mxxMat0) (w_121_0 : mxxMat1) (w_123_0 : mxxMat0) (w_124_0 : mxxMat0) (w_125_0 : mxxMat0) (w_126_0 : mxxMat2) (w_126_concat_1 : mxxMat3) (w_127_0 : mxxMat0) (w_128_0 : mxxMat0) (w_129_0 : mxxMat3) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_112_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_113_0 ∧
  MxxRuntime.concatRows w_112_0 w_113_0 w_114_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_114_0 w_115_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_115_0 w_116_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_117_0 0 1 0 25 w_118_0 ∧
  MxxRuntime.familyGetStatic w_120_0 (1) w_121_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_117_0 1 2 0 25 w_124_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_117_0 2 3 0 25 w_125_0 ∧
  MxxRuntime.concatRows w_123_0 w_124_0 w_126_concat_1 ∧
  MxxRuntime.concatRows w_126_concat_1 w_125_0 w_126_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_127_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_128_0 ∧
  MxxRuntime.concatRows w_127_0 w_128_0 w_129_0

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_15_0 : Fin 2 → mxxMat0) (w_91_0 : Fin 4 → mxxMat0) (w_93_0 : Fin 4 → mxxMat0) (w_95_0 : Fin 4 → mxxMat0) (w_97_0 : Fin 4 → mxxMat0) (w_101_0 : Fin 2 → mxxMat0) (w_106_0 : Fin 2 → mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat6) (w_60_0 : mxxMat6) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat6) (w_67_0 : mxxMat6) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_75_0 : mxxMat0) (w_76_0 : mxxMat6) (w_77_0 : mxxMat6) (w_81_0 : mxxMat0) (w_83_0 : mxxMat0) (w_84_0 : mxxMat6) (w_85_0 : mxxMat6) (w_87_0 : mxxMat0) (w_92_0 : mxxMat0) (w_94_0 : mxxMat0) (w_96_0 : mxxMat0) (w_98_0 : mxxMat0) (w_102_0 : mxxMat0) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat2) (w_105_concat_1 : mxxMat3) (w_107_0 : mxxMat0) (w_108_0 : mxxMat0) (w_109_0 : mxxMat0) (w_110_0 : mxxMat2) (w_110_concat_1 : mxxMat3) (w_111_0 : mxxMat4) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_55_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_62_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_63_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_65_0 w_66_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_66_0 w_67_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_70_0 ∧
  MxxRuntime.centeredRebaseRuns w_70_0 w_71_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_73_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_75_0 w_76_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_76_0 w_77_0 ∧
  MxxRuntime.centeredRebaseRuns w_72_0 w_81_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_83_0 w_84_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_84_0 w_85_0 ∧
  MxxRuntime.familyGetStatic w_15_0 (1) w_87_0 ∧
  MxxRuntime.familyGetStatic w_91_0 (0) w_92_0 ∧
  MxxRuntime.familyGetStatic w_93_0 (0) w_94_0 ∧
  MxxRuntime.familyGetStatic w_95_0 (0) w_96_0 ∧
  MxxRuntime.familyGetStatic w_97_0 (0) w_98_0 ∧
  MxxRuntime.familyGetStatic w_101_0 (1) w_102_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_103_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_104_0 ∧
  MxxRuntime.concatRows w_102_0 w_103_0 w_105_concat_1 ∧
  MxxRuntime.concatRows w_105_concat_1 w_104_0 w_105_0 ∧
  MxxRuntime.familyGetStatic w_106_0 (1) w_107_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_108_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_109_0 ∧
  MxxRuntime.concatRows w_107_0 w_108_0 w_110_concat_1 ∧
  MxxRuntime.concatRows w_110_concat_1 w_109_0 w_110_0 ∧
  MxxRuntime.concatColumns w_105_0 w_110_0 w_111_0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 8 → mxxMat0) (w_1_0 : Fin 8 → mxxMat0) (w_2_0 : Fin 8 → mxxMat0) (w_3_0 : Fin 8 → mxxMat0) (w_4_0 : Fin 8 → mxxMat0) (w_5_0 : ByteArray) (w_6_0 : Fin 8 → mxxMat0) (w_7_0 : Fin 8 → mxxMat0) (w_8_0 : Fin 8 → mxxMat0) (w_9_0 : Fin 8 → mxxMat0) (w_10_0 : Fin 8 → mxxMat0) (w_11_0 : Fin 8 → mxxMat0) (w_12_0 : Fin 8 → mxxMat0) (w_13_0 : Fin 16 → Int) (w_14_0 : Fin 2 → mxxMat0) (w_15_0 : Fin 2 → mxxMat0) (w_16_0 : Fin 16 → Int) (w_19_0 : mxxMat0) (w_20_0 : Fin 2 → mxxMat1) (w_23_0 : mxxMat0) (w_24_0 : ByteArray) (w_25_0 : Fin 16 → mxxMat0) (w_26_0 : Fin 16 → mxxMat0) (w_29_0 : Fin 16 → mxxMat0) (w_30_0 : Fin 16 → mxxMat0) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_27_0 : Fin 16 → mxxMat0) (w_27_1 : Fin 16 → mxxMat0) (w_27_2 : Fin 16 → mxxMat0) (w_27_3 : Fin 16 → mxxMat0) (w_27_4 : Fin 16 → mxxMat0) (w_27_5 : Fin 16 → mxxMat0) (w_28_0 : Fin 8 → mxxMat0) (w_28_1 : Fin 8 → mxxMat0) (w_28_2 : Fin 8 → mxxMat0) (w_28_3 : Fin 8 → mxxMat0) (w_28_4 : Fin 8 → mxxMat0) (w_28_5 : Fin 8 → mxxMat0) (w_31_0 : Fin 16 → mxxMat0) (w_31_1 : Fin 16 → mxxMat0) (w_31_2 : Fin 16 → mxxMat0) (w_31_3 : Fin 16 → mxxMat0) (w_31_4 : Fin 16 → mxxMat0) (w_31_5 : Fin 16 → mxxMat0) (w_32_0 : Fin 8 → mxxMat0) (w_32_1 : Fin 8 → mxxMat0) (w_32_2 : Fin 8 → mxxMat0) (w_32_3 : Fin 8 → mxxMat0) (w_32_4 : Fin 8 → mxxMat0) (w_32_5 : Fin 8 → mxxMat0) (w_33_0 : Fin 8 → mxxMat0) (w_33_1 : Fin 8 → mxxMat0) (w_33_2 : Fin 8 → mxxMat0) (w_33_3 : Fin 8 → mxxMat0) (w_33_4 : Fin 8 → mxxMat0) (w_33_5 : Fin 8 → mxxMat0) (w_34_0 : Fin 8 → mxxMat0) (w_34_1 : Fin 8 → mxxMat0) (w_34_2 : Fin 8 → mxxMat0) (w_34_3 : Fin 8 → mxxMat0) (w_34_4 : Fin 8 → mxxMat0) (w_34_5 : Fin 8 → mxxMat0) (w_35_0 : Fin 8 → mxxMat0) (w_35_1 : Fin 8 → mxxMat0) (w_35_2 : Fin 8 → mxxMat0) (w_35_3 : Fin 8 → mxxMat0) (w_35_4 : Fin 8 → mxxMat0) (w_35_5 : Fin 8 → mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat0) (w_49_0 : mxxMat0) (w_50_0 : mxxMat0) (w_51_0 : mxxMat0) (w_52_0 : mxxMat0) (w_53_0 : mxxMat0) (w_54_0 : mxxMat0) (w_54_1 : mxxMat0) (w_54_2 : mxxMat0) (w_54_3 : mxxMat0) (w_54_4 : mxxMat0) (w_54_5 : mxxMat0) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_17_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_21_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_22_0 ∧
  (16) = 16 ∧
  (∀ i : Fin 16, ∃ w_14_0_gathered : mxxMat0, ∃ w_15_0_gathered : mxxMat0, ∃ w_20_0_gathered : mxxMat1, 0 ≤ (w_16_0 i) ∧
  (w_16_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_14_0 (w_16_0 i) w_14_0_gathered ∧
  0 ≤ (w_16_0 i) ∧
  (w_16_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_15_0 (w_16_0 i) w_15_0_gathered ∧
  0 ≤ (w_16_0 i) ∧
  (w_16_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_20_0 (w_16_0 i) w_20_0_gathered ∧
  parallel_generatedRoot_27 backend hashModel tape (path ++ [27, i.val]) params i ((w_16_0 i), (w_14_0_gathered, (w_5_0, (w_15_0_gathered, (w_17_0, (w_18_0, (w_19_0, (w_20_0_gathered, (w_21_0, (w_22_0, (w_23_0, (w_24_0, ((w_25_0 i), ((w_26_0 i), ())))))))))))))) ((w_27_0 i), ((w_27_1 i), ((w_27_2 i), ((w_27_3 i), ((w_27_4 i), ((w_27_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_28 backend hashModel tape (path ++ [28, i.val]) params i (w_27_0, (w_27_1, (w_27_2, (w_27_3, (w_27_4, (w_27_5, ())))))) ((w_28_0 i), ((w_28_1 i), ((w_28_2 i), ((w_28_3 i), ((w_28_4 i), ((w_28_5 i), ()))))))) ∧
  (16) = 16 ∧
  (∀ i : Fin 16, ∃ w_14_0_gathered : mxxMat0, ∃ w_15_0_gathered : mxxMat0, ∃ w_20_0_gathered : mxxMat1, 0 ≤ (w_13_0 i) ∧
  (w_13_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_14_0 (w_13_0 i) w_14_0_gathered ∧
  0 ≤ (w_13_0 i) ∧
  (w_13_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_15_0 (w_13_0 i) w_15_0_gathered ∧
  0 ≤ (w_13_0 i) ∧
  (w_13_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_20_0 (w_13_0 i) w_20_0_gathered ∧
  parallel_generatedRoot_31 backend hashModel tape (path ++ [31, i.val]) params i ((w_13_0 i), (w_14_0_gathered, (w_5_0, (w_15_0_gathered, (w_28_0, (w_28_2, (w_28_4, (w_20_0_gathered, (w_28_1, (w_28_3, (w_28_5, (w_24_0, ((w_29_0 i), ((w_30_0 i), ())))))))))))))) ((w_31_0 i), ((w_31_1 i), ((w_31_2 i), ((w_31_3 i), ((w_31_4 i), ((w_31_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_32 backend hashModel tape (path ++ [32, i.val]) params i (w_31_0, (w_31_1, (w_31_2, (w_31_3, (w_31_4, (w_31_5, ())))))) ((w_32_0 i), ((w_32_1 i), ((w_32_2 i), ((w_32_3 i), ((w_32_4 i), ((w_32_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_33 backend hashModel tape (path ++ [33, i.val]) params i (w_7_0, (w_8_0, (w_9_0, (w_10_0, (w_11_0, (w_5_0, (w_12_0, ((w_32_4 i), ((w_32_0 i), ((w_32_2 i), ((w_32_5 i), ((w_32_1 i), ((w_32_3 i), ()))))))))))))) ((w_33_0 i), ((w_33_1 i), ((w_33_2 i), ((w_33_3 i), ((w_33_4 i), ((w_33_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_34 backend hashModel tape (path ++ [34, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, (w_6_0, ((w_33_4 i), ((w_33_0 i), ((w_33_2 i), ((w_33_5 i), ((w_33_1 i), ((w_33_3 i), ()))))))))))))) ((w_34_0 i), ((w_34_1 i), ((w_34_2 i), ((w_34_3 i), ((w_34_4 i), ((w_34_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_35 backend hashModel tape (path ++ [35, i.val]) params i ((w_34_0 i), ((w_34_1 i), ((w_34_2 i), ((w_34_3 i), ((w_34_4 i), ((w_34_5 i), ())))))) ((w_35_0 i), ((w_35_1 i), ((w_35_2 i), ((w_35_3 i), ((w_35_4 i), ((w_35_5 i), ()))))))) ∧
  MxxRuntime.familyGetStatic w_35_0 (0) w_36_0 ∧
  MxxRuntime.familyGetStatic w_35_0 (1) w_37_0 ∧
  MxxRuntime.familyGetStatic w_35_1 (0) w_39_0 ∧
  MxxRuntime.familyGetStatic w_35_1 (1) w_40_0 ∧
  MxxRuntime.familyGetStatic w_35_2 (0) w_42_0 ∧
  MxxRuntime.familyGetStatic w_35_2 (1) w_43_0 ∧
  MxxRuntime.familyGetStatic w_35_3 (0) w_45_0 ∧
  MxxRuntime.familyGetStatic w_35_3 (1) w_46_0 ∧
  MxxRuntime.familyGetStatic w_35_4 (0) w_48_0 ∧
  MxxRuntime.familyGetStatic w_35_4 (1) w_49_0 ∧
  MxxRuntime.familyGetStatic w_35_5 (0) w_51_0 ∧
  MxxRuntime.familyGetStatic w_35_5 (1) w_52_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_54 backend hashModel tape (path ++ [54, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_35_0, (w_35_1, (w_35_2, (w_35_3, (w_35_4, (w_35_5, ())))))))))))) next) (Int.toNat (6)) (w_38_0, (w_41_0, (w_44_0, (w_47_0, (w_50_0, (w_53_0, ())))))) (w_54_0, (w_54_1, (w_54_2, (w_54_3, (w_54_4, (w_54_5, ()))))))

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Ctx (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) where
  inputs : (generatedRoot.InputType)
  witness : generatedRoot.Witness

noncomputable section
set_option linter.unusedVariables false
variable {backend : MxxRuntime.BackendContext} {hashModel : MxxRuntime.HashModel} {tape : MxxRuntime.SampleTape} {path : List Nat} {params : Params}
local notation "generatedRootCtx" => (generatedRoot.Ctx backend hashModel tape path params)

noncomputable def generatedRoot.w_0_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.1

noncomputable def generatedRoot.w_1_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.1

noncomputable def generatedRoot.w_2_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.1

noncomputable def generatedRoot.w_3_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.1

noncomputable def generatedRoot.w_4_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.1

noncomputable def generatedRoot.w_5_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.1

noncomputable def generatedRoot.w_6_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_7_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_8_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_9_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_10_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_11_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_12_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_13_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_14_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_15_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_16_0  (ctx : generatedRootCtx) : Fin 16 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_19_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_20_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_23_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_24_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_25_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_26_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_29_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_30_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_91_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_93_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_95_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_97_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_99_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_100_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_101_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_106_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_119_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_120_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_134_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_142_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_152_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_230_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_232_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_234_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_236_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_238_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_239_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_288_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_290_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_292_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_294_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_296_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_297_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_346_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_348_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_350_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_352_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_354_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_355_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_404_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_406_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_408_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_410_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_412_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_413_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_462_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_464_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_466_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_468_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_470_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_471_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_520_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_522_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_524_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_526_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_528_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_529_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_578_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_580_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_582_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_584_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_586_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_587_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_638_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_640_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_641_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_643_0  (ctx : generatedRootCtx) : Fin 4 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_649_0  (ctx : generatedRootCtx) : Fin 5 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_753_0  (ctx : generatedRootCtx) : ByteArray :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_858_0  (ctx : generatedRootCtx) : mxxMat27 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_872_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_873_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_874_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_875_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_876_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_877_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_878_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_879_0  (ctx : generatedRootCtx) : Fin 14400 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_882_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_883_0  (ctx : generatedRootCtx) : Fin 2 → mxxMat1 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_886_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_887_0  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_888_0  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_891_0  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_892_0  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_929_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_930_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_931_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_932_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_933_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_934_0  (ctx : generatedRootCtx) : Fin 72000 → Int :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_937_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_940_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_941_0  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_942_0  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_945_0  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_946_0  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1

noncomputable def generatedRoot.w_17_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_17_0

noncomputable def generatedRoot.w_18_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_18_0

noncomputable def generatedRoot.w_21_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_21_0

noncomputable def generatedRoot.w_22_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_22_0

noncomputable def generatedRoot.w_27_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_27_0

noncomputable def generatedRoot.w_27_1  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_27_1

noncomputable def generatedRoot.w_27_2  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_27_2

noncomputable def generatedRoot.w_27_3  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_27_3

noncomputable def generatedRoot.w_27_4  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_27_4

noncomputable def generatedRoot.w_27_5  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_27_5

noncomputable def generatedRoot.w_28_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_28_0

noncomputable def generatedRoot.w_28_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_28_1

noncomputable def generatedRoot.w_28_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_28_2

noncomputable def generatedRoot.w_28_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_28_3

noncomputable def generatedRoot.w_28_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_28_4

noncomputable def generatedRoot.w_28_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_28_5

noncomputable def generatedRoot.w_31_0  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_31_0

noncomputable def generatedRoot.w_31_1  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_31_1

noncomputable def generatedRoot.w_31_2  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_31_2

noncomputable def generatedRoot.w_31_3  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_31_3

noncomputable def generatedRoot.w_31_4  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_31_4

noncomputable def generatedRoot.w_31_5  (ctx : generatedRootCtx) : Fin 16 → mxxMat0 :=
  ctx.witness.w_31_5

noncomputable def generatedRoot.w_32_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_32_0

noncomputable def generatedRoot.w_32_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_32_1

noncomputable def generatedRoot.w_32_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_32_2

noncomputable def generatedRoot.w_32_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_32_3

noncomputable def generatedRoot.w_32_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_32_4

noncomputable def generatedRoot.w_32_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_32_5

noncomputable def generatedRoot.w_33_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_33_0

noncomputable def generatedRoot.w_33_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_33_1

noncomputable def generatedRoot.w_33_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_33_2

noncomputable def generatedRoot.w_33_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_33_3

noncomputable def generatedRoot.w_33_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_33_4

noncomputable def generatedRoot.w_33_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_33_5

noncomputable def generatedRoot.w_34_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_34_0

noncomputable def generatedRoot.w_34_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_34_1

noncomputable def generatedRoot.w_34_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_34_2

noncomputable def generatedRoot.w_34_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_34_3

noncomputable def generatedRoot.w_34_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_34_4

noncomputable def generatedRoot.w_34_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_34_5

noncomputable def generatedRoot.w_35_0  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_35_0

noncomputable def generatedRoot.w_35_1  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_35_1

noncomputable def generatedRoot.w_35_2  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_35_2

noncomputable def generatedRoot.w_35_3  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_35_3

noncomputable def generatedRoot.w_35_4  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_35_4

noncomputable def generatedRoot.w_35_5  (ctx : generatedRootCtx) : Fin 8 → mxxMat0 :=
  ctx.witness.w_35_5

noncomputable def generatedRoot.w_36_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_36_0

noncomputable def generatedRoot.w_37_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_37_0

noncomputable def generatedRoot.w_38_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx)

noncomputable def generatedRoot.w_39_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_39_0

noncomputable def generatedRoot.w_40_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_40_0

noncomputable def generatedRoot.w_41_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx)

noncomputable def generatedRoot.w_42_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_42_0

noncomputable def generatedRoot.w_43_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_43_0

noncomputable def generatedRoot.w_44_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx)

noncomputable def generatedRoot.w_45_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_45_0

noncomputable def generatedRoot.w_46_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_46_0

noncomputable def generatedRoot.w_47_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx)

noncomputable def generatedRoot.w_48_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_48_0

noncomputable def generatedRoot.w_49_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_49_0

noncomputable def generatedRoot.w_50_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx)

noncomputable def generatedRoot.w_51_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_51_0

noncomputable def generatedRoot.w_52_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_52_0

noncomputable def generatedRoot.w_53_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx)

noncomputable def generatedRoot.w_54_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_54_0

noncomputable def generatedRoot.w_54_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_54_1

noncomputable def generatedRoot.w_54_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_54_2

noncomputable def generatedRoot.w_54_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_54_3

noncomputable def generatedRoot.w_54_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_54_4

noncomputable def generatedRoot.w_54_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_54_5

noncomputable def generatedRoot.w_55_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_55_0

noncomputable def generatedRoot.w_56_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_56_0

noncomputable def generatedRoot.w_57_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_58_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_56_0 ctx) (generatedRoot.w_57_0 ctx)

noncomputable def generatedRoot.w_59_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_59_0

noncomputable def generatedRoot.w_60_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_60_0

noncomputable def generatedRoot.w_61_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_54_1 ctx) (generatedRoot.w_60_0 ctx)

noncomputable def generatedRoot.w_62_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_62_0

noncomputable def generatedRoot.w_63_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_63_0

noncomputable def generatedRoot.w_64_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_65_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_63_0 ctx) (generatedRoot.w_64_0 ctx)

noncomputable def generatedRoot.w_66_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_66_0

noncomputable def generatedRoot.w_67_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_67_0

noncomputable def generatedRoot.w_68_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_62_0 ctx) (generatedRoot.w_67_0 ctx)

noncomputable def generatedRoot.w_69_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_70_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_70_0

noncomputable def generatedRoot.w_71_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_71_0

noncomputable def generatedRoot.w_72_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarLeft (generatedRoot.w_69_0 ctx) (generatedRoot.w_71_0 ctx)

noncomputable def generatedRoot.w_73_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_73_0

noncomputable def generatedRoot.w_74_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_75_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_73_0 ctx) (generatedRoot.w_74_0 ctx)

noncomputable def generatedRoot.w_76_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_76_0

noncomputable def generatedRoot.w_77_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_77_0

noncomputable def generatedRoot.w_78_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_19_0 ctx) (generatedRoot.w_77_0 ctx)

noncomputable def generatedRoot.w_79_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_23_0 ctx) (generatedRoot.w_77_0 ctx)

noncomputable def generatedRoot.w_80_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_54_5 ctx) (generatedRoot.w_60_0 ctx)

noncomputable def generatedRoot.w_81_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_81_0

noncomputable def generatedRoot.w_82_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_83_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_81_0 ctx) (generatedRoot.w_82_0 ctx)

noncomputable def generatedRoot.w_84_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_84_0

noncomputable def generatedRoot.w_85_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_85_0

noncomputable def generatedRoot.w_86_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_80_0 ctx) (generatedRoot.w_85_0 ctx)

noncomputable def generatedRoot.w_87_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_87_0

noncomputable def generatedRoot.w_88_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_87_0 ctx) (generatedRoot.w_67_0 ctx)

noncomputable def generatedRoot.w_89_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_85_0 ctx)

noncomputable def generatedRoot.w_90_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_92_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_92_0

noncomputable def generatedRoot.w_94_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_94_0

noncomputable def generatedRoot.w_96_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_96_0

noncomputable def generatedRoot.w_98_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_98_0

noncomputable def generatedRoot.w_102_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_102_0

noncomputable def generatedRoot.w_103_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_103_0

noncomputable def generatedRoot.w_104_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_104_0

noncomputable def generatedRoot.w_105_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_105_0

noncomputable def generatedRoot.w_105_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_105_concat_1

noncomputable def generatedRoot.w_107_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_107_0

noncomputable def generatedRoot.w_108_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_108_0

noncomputable def generatedRoot.w_109_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_109_0

noncomputable def generatedRoot.w_110_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_110_0

noncomputable def generatedRoot.w_110_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_110_concat_1

noncomputable def generatedRoot.w_111_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_111_0

noncomputable def generatedRoot.w_112_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_112_0

noncomputable def generatedRoot.w_113_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_113_0

noncomputable def generatedRoot.w_114_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_114_0

noncomputable def generatedRoot.w_115_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_115_0

noncomputable def generatedRoot.w_116_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_116_0

noncomputable def generatedRoot.w_117_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_111_0 ctx) (generatedRoot.w_116_0 ctx)

noncomputable def generatedRoot.w_118_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_118_0

noncomputable def generatedRoot.w_121_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_121_0

noncomputable def generatedRoot.w_122_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_119_0 ctx) (generatedRoot.w_121_0 ctx)

noncomputable def generatedRoot.w_123_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_118_0 ctx) (generatedRoot.w_122_0 ctx)

noncomputable def generatedRoot.w_124_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_124_0

noncomputable def generatedRoot.w_125_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_125_0

noncomputable def generatedRoot.w_126_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_126_0

noncomputable def generatedRoot.w_126_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_126_concat_1

noncomputable def generatedRoot.w_127_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_127_0

noncomputable def generatedRoot.w_128_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_128_0

noncomputable def generatedRoot.w_129_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_129_0

noncomputable def generatedRoot.w_130_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_130_0

noncomputable def generatedRoot.w_131_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_131_0

noncomputable def generatedRoot.w_132_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_111_0 ctx) (generatedRoot.w_131_0 ctx)

noncomputable def generatedRoot.w_133_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_133_0

noncomputable def generatedRoot.w_135_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_134_0 ctx) (generatedRoot.w_121_0 ctx)

noncomputable def generatedRoot.w_136_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_133_0 ctx) (generatedRoot.w_135_0 ctx)

noncomputable def generatedRoot.w_137_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_137_0

noncomputable def generatedRoot.w_138_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_138_0

noncomputable def generatedRoot.w_139_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_139_0

noncomputable def generatedRoot.w_139_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_139_concat_1

noncomputable def generatedRoot.w_140_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_140_0

noncomputable def generatedRoot.w_141_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_141_0

noncomputable def generatedRoot.w_143_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_143_0

noncomputable def generatedRoot.w_144_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_144_0

noncomputable def generatedRoot.w_145_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_145_0

noncomputable def generatedRoot.w_146_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_146_0

noncomputable def generatedRoot.w_147_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_140_0 ctx) (generatedRoot.w_146_0 ctx)

noncomputable def generatedRoot.w_148_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_148_0

noncomputable def generatedRoot.w_149_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_149_0

noncomputable def generatedRoot.w_150_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_150_0

noncomputable def generatedRoot.w_151_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_151_0

noncomputable def generatedRoot.w_153_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_153_0

noncomputable def generatedRoot.w_154_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_154_0

noncomputable def generatedRoot.w_155_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_155_0

noncomputable def generatedRoot.w_156_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_156_0

noncomputable def generatedRoot.w_157_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_140_0 ctx) (generatedRoot.w_156_0 ctx)

noncomputable def generatedRoot.w_158_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_158_0

noncomputable def generatedRoot.w_159_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_159_0

noncomputable def generatedRoot.w_160_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_160_0

noncomputable def generatedRoot.w_161_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_0

noncomputable def generatedRoot.w_161_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_1

noncomputable def generatedRoot.w_161_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_2

noncomputable def generatedRoot.w_161_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_3

noncomputable def generatedRoot.w_161_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_4

noncomputable def generatedRoot.w_161_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_5

noncomputable def generatedRoot.w_161_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_6

noncomputable def generatedRoot.w_161_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_7

noncomputable def generatedRoot.w_161_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_8

noncomputable def generatedRoot.w_161_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_161_9

noncomputable def generatedRoot.w_162_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_162_0

noncomputable def generatedRoot.w_163_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (3) [(1), (3), (5), (15)])

noncomputable def generatedRoot.w_164_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_164_0

noncomputable def generatedRoot.w_165_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_165_0

noncomputable def generatedRoot.w_166_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_166_0

noncomputable def generatedRoot.w_166_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_166_concat_1

noncomputable def generatedRoot.w_167_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_167_0

noncomputable def generatedRoot.w_168_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_168_0

noncomputable def generatedRoot.w_169_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_169_0

noncomputable def generatedRoot.w_170_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_170_0

noncomputable def generatedRoot.w_170_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_170_concat_1

noncomputable def generatedRoot.w_171_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_171_0

noncomputable def generatedRoot.w_172_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_172_0

noncomputable def generatedRoot.w_173_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_173_0

noncomputable def generatedRoot.w_174_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_174_0

noncomputable def generatedRoot.w_175_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_175_0

noncomputable def generatedRoot.w_175_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_175_concat_1

noncomputable def generatedRoot.w_176_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_176_0

noncomputable def generatedRoot.w_177_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_177_0

noncomputable def generatedRoot.w_178_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_178_0

noncomputable def generatedRoot.w_179_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_179_0

noncomputable def generatedRoot.w_179_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_179_concat_1

noncomputable def generatedRoot.w_180_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_180_0

noncomputable def generatedRoot.w_181_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_181_0

noncomputable def generatedRoot.w_182_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_182_0

noncomputable def generatedRoot.w_183_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_183_0

noncomputable def generatedRoot.w_184_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_180_0 ctx) (generatedRoot.w_183_0 ctx)

noncomputable def generatedRoot.w_185_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_185_0

noncomputable def generatedRoot.w_186_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_186_0

noncomputable def generatedRoot.w_187_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_187_0

noncomputable def generatedRoot.w_188_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_188_0

noncomputable def generatedRoot.w_189_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_189_0

noncomputable def generatedRoot.w_190_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_171_0 ctx) (generatedRoot.w_189_0 ctx)

noncomputable def generatedRoot.w_191_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_191_0

noncomputable def generatedRoot.w_192_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_192_0

noncomputable def generatedRoot.w_193_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx)

noncomputable def generatedRoot.w_194_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_161_0 ctx) (generatedRoot.w_193_0 ctx)

noncomputable def generatedRoot.w_195_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_195_0

noncomputable def generatedRoot.w_196_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_197_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_195_0 ctx) (generatedRoot.w_196_0 ctx)

noncomputable def generatedRoot.w_198_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_198_0

noncomputable def generatedRoot.w_199_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_199_0

noncomputable def generatedRoot.w_200_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_194_0 ctx) (generatedRoot.w_199_0 ctx)

noncomputable def generatedRoot.w_201_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_201_0

noncomputable def generatedRoot.w_202_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_202_0

noncomputable def generatedRoot.w_203_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_203_0

noncomputable def generatedRoot.w_204_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_180_0 ctx) (generatedRoot.w_203_0 ctx)

noncomputable def generatedRoot.w_205_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_205_0

noncomputable def generatedRoot.w_206_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_206_0

noncomputable def generatedRoot.w_207_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_207_0

noncomputable def generatedRoot.w_208_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_208_0

noncomputable def generatedRoot.w_209_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_209_0

noncomputable def generatedRoot.w_210_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_171_0 ctx) (generatedRoot.w_209_0 ctx)

noncomputable def generatedRoot.w_211_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_211_0

noncomputable def generatedRoot.w_212_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_212_0

noncomputable def generatedRoot.w_213_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx)

noncomputable def generatedRoot.w_214_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_161_1 ctx) (generatedRoot.w_213_0 ctx)

noncomputable def generatedRoot.w_215_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_214_0 ctx) (generatedRoot.w_199_0 ctx)

noncomputable def generatedRoot.w_216_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_216_0

noncomputable def generatedRoot.w_217_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_218_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_216_0 ctx) (generatedRoot.w_217_0 ctx)

noncomputable def generatedRoot.w_219_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_219_0

noncomputable def generatedRoot.w_220_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_220_0

noncomputable def generatedRoot.w_221_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_54_5 ctx) (generatedRoot.w_220_0 ctx)

noncomputable def generatedRoot.w_222_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_222_0

noncomputable def generatedRoot.w_223_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_224_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_222_0 ctx) (generatedRoot.w_223_0 ctx)

noncomputable def generatedRoot.w_225_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_225_0

noncomputable def generatedRoot.w_226_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_226_0

noncomputable def generatedRoot.w_227_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_221_0 ctx) (generatedRoot.w_226_0 ctx)

noncomputable def generatedRoot.w_228_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_226_0 ctx)

noncomputable def generatedRoot.w_229_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_231_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_231_0

noncomputable def generatedRoot.w_233_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_233_0

noncomputable def generatedRoot.w_235_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_235_0

noncomputable def generatedRoot.w_237_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_237_0

noncomputable def generatedRoot.w_240_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_0

noncomputable def generatedRoot.w_240_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_1

noncomputable def generatedRoot.w_240_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_2

noncomputable def generatedRoot.w_240_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_3

noncomputable def generatedRoot.w_240_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_4

noncomputable def generatedRoot.w_240_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_5

noncomputable def generatedRoot.w_240_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_6

noncomputable def generatedRoot.w_240_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_7

noncomputable def generatedRoot.w_240_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_8

noncomputable def generatedRoot.w_240_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_240_9

noncomputable def generatedRoot.w_241_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_241_0

noncomputable def generatedRoot.w_242_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (0) [(3)])

noncomputable def generatedRoot.w_243_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_243_0

noncomputable def generatedRoot.w_244_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_244_0

noncomputable def generatedRoot.w_245_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_245_0

noncomputable def generatedRoot.w_245_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_245_concat_1

noncomputable def generatedRoot.w_246_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_246_0

noncomputable def generatedRoot.w_247_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_247_0

noncomputable def generatedRoot.w_248_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_248_0

noncomputable def generatedRoot.w_249_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_249_0

noncomputable def generatedRoot.w_249_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_249_concat_1

noncomputable def generatedRoot.w_250_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_250_0

noncomputable def generatedRoot.w_251_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_251_0

noncomputable def generatedRoot.w_252_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_252_0

noncomputable def generatedRoot.w_253_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_253_0

noncomputable def generatedRoot.w_254_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_254_0

noncomputable def generatedRoot.w_254_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_254_concat_1

noncomputable def generatedRoot.w_255_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_255_0

noncomputable def generatedRoot.w_256_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_256_0

noncomputable def generatedRoot.w_257_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_257_0

noncomputable def generatedRoot.w_258_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_258_0

noncomputable def generatedRoot.w_258_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_258_concat_1

noncomputable def generatedRoot.w_259_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_259_0

noncomputable def generatedRoot.w_260_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_260_0

noncomputable def generatedRoot.w_261_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_261_0

noncomputable def generatedRoot.w_262_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_262_0

noncomputable def generatedRoot.w_263_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_259_0 ctx) (generatedRoot.w_262_0 ctx)

noncomputable def generatedRoot.w_264_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_264_0

noncomputable def generatedRoot.w_265_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_265_0

noncomputable def generatedRoot.w_266_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_266_0

noncomputable def generatedRoot.w_267_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_267_0

noncomputable def generatedRoot.w_268_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_268_0

noncomputable def generatedRoot.w_269_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_250_0 ctx) (generatedRoot.w_268_0 ctx)

noncomputable def generatedRoot.w_270_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_270_0

noncomputable def generatedRoot.w_271_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_271_0

noncomputable def generatedRoot.w_272_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_270_0 ctx) (generatedRoot.w_271_0 ctx)

noncomputable def generatedRoot.w_273_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_240_1 ctx) (generatedRoot.w_272_0 ctx)

noncomputable def generatedRoot.w_274_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_274_0

noncomputable def generatedRoot.w_275_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_276_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_274_0 ctx) (generatedRoot.w_275_0 ctx)

noncomputable def generatedRoot.w_277_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_277_0

noncomputable def generatedRoot.w_278_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_278_0

noncomputable def generatedRoot.w_279_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_273_0 ctx) (generatedRoot.w_278_0 ctx)

noncomputable def generatedRoot.w_280_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_280_0

noncomputable def generatedRoot.w_281_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_282_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_280_0 ctx) (generatedRoot.w_281_0 ctx)

noncomputable def generatedRoot.w_283_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_283_0

noncomputable def generatedRoot.w_284_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_284_0

noncomputable def generatedRoot.w_285_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_279_0 ctx) (generatedRoot.w_284_0 ctx)

noncomputable def generatedRoot.w_286_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_284_0 ctx)

noncomputable def generatedRoot.w_287_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_289_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_289_0

noncomputable def generatedRoot.w_291_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_291_0

noncomputable def generatedRoot.w_293_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_293_0

noncomputable def generatedRoot.w_295_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_295_0

noncomputable def generatedRoot.w_298_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_0

noncomputable def generatedRoot.w_298_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_1

noncomputable def generatedRoot.w_298_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_2

noncomputable def generatedRoot.w_298_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_3

noncomputable def generatedRoot.w_298_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_4

noncomputable def generatedRoot.w_298_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_5

noncomputable def generatedRoot.w_298_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_6

noncomputable def generatedRoot.w_298_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_7

noncomputable def generatedRoot.w_298_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_8

noncomputable def generatedRoot.w_298_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_298_9

noncomputable def generatedRoot.w_299_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_299_0

noncomputable def generatedRoot.w_300_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (0) [(5)])

noncomputable def generatedRoot.w_301_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_301_0

noncomputable def generatedRoot.w_302_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_302_0

noncomputable def generatedRoot.w_303_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_303_0

noncomputable def generatedRoot.w_303_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_303_concat_1

noncomputable def generatedRoot.w_304_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_304_0

noncomputable def generatedRoot.w_305_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_305_0

noncomputable def generatedRoot.w_306_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_306_0

noncomputable def generatedRoot.w_307_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_307_0

noncomputable def generatedRoot.w_307_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_307_concat_1

noncomputable def generatedRoot.w_308_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_308_0

noncomputable def generatedRoot.w_309_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_309_0

noncomputable def generatedRoot.w_310_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_310_0

noncomputable def generatedRoot.w_311_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_311_0

noncomputable def generatedRoot.w_312_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_312_0

noncomputable def generatedRoot.w_312_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_312_concat_1

noncomputable def generatedRoot.w_313_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_313_0

noncomputable def generatedRoot.w_314_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_314_0

noncomputable def generatedRoot.w_315_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_315_0

noncomputable def generatedRoot.w_316_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_316_0

noncomputable def generatedRoot.w_316_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_316_concat_1

noncomputable def generatedRoot.w_317_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_317_0

noncomputable def generatedRoot.w_318_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_318_0

noncomputable def generatedRoot.w_319_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_319_0

noncomputable def generatedRoot.w_320_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_320_0

noncomputable def generatedRoot.w_321_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_317_0 ctx) (generatedRoot.w_320_0 ctx)

noncomputable def generatedRoot.w_322_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_322_0

noncomputable def generatedRoot.w_323_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_323_0

noncomputable def generatedRoot.w_324_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_324_0

noncomputable def generatedRoot.w_325_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_325_0

noncomputable def generatedRoot.w_326_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_326_0

noncomputable def generatedRoot.w_327_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_308_0 ctx) (generatedRoot.w_326_0 ctx)

noncomputable def generatedRoot.w_328_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_328_0

noncomputable def generatedRoot.w_329_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_329_0

noncomputable def generatedRoot.w_330_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_328_0 ctx) (generatedRoot.w_329_0 ctx)

noncomputable def generatedRoot.w_331_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_298_1 ctx) (generatedRoot.w_330_0 ctx)

noncomputable def generatedRoot.w_332_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_332_0

noncomputable def generatedRoot.w_333_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_334_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_332_0 ctx) (generatedRoot.w_333_0 ctx)

noncomputable def generatedRoot.w_335_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_335_0

noncomputable def generatedRoot.w_336_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_336_0

noncomputable def generatedRoot.w_337_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_331_0 ctx) (generatedRoot.w_336_0 ctx)

noncomputable def generatedRoot.w_338_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_338_0

noncomputable def generatedRoot.w_339_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_340_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_338_0 ctx) (generatedRoot.w_339_0 ctx)

noncomputable def generatedRoot.w_341_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_341_0

noncomputable def generatedRoot.w_342_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_342_0

noncomputable def generatedRoot.w_343_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_337_0 ctx) (generatedRoot.w_342_0 ctx)

noncomputable def generatedRoot.w_344_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_342_0 ctx)

noncomputable def generatedRoot.w_345_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_347_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_347_0

noncomputable def generatedRoot.w_349_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_349_0

noncomputable def generatedRoot.w_351_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_351_0

noncomputable def generatedRoot.w_353_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_353_0

noncomputable def generatedRoot.w_356_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_0

noncomputable def generatedRoot.w_356_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_1

noncomputable def generatedRoot.w_356_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_2

noncomputable def generatedRoot.w_356_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_3

noncomputable def generatedRoot.w_356_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_4

noncomputable def generatedRoot.w_356_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_5

noncomputable def generatedRoot.w_356_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_6

noncomputable def generatedRoot.w_356_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_7

noncomputable def generatedRoot.w_356_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_8

noncomputable def generatedRoot.w_356_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_356_9

noncomputable def generatedRoot.w_357_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_357_0

noncomputable def generatedRoot.w_358_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (0) [(7)])

noncomputable def generatedRoot.w_359_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_359_0

noncomputable def generatedRoot.w_360_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_360_0

noncomputable def generatedRoot.w_361_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_361_0

noncomputable def generatedRoot.w_361_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_361_concat_1

noncomputable def generatedRoot.w_362_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_362_0

noncomputable def generatedRoot.w_363_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_363_0

noncomputable def generatedRoot.w_364_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_364_0

noncomputable def generatedRoot.w_365_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_365_0

noncomputable def generatedRoot.w_365_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_365_concat_1

noncomputable def generatedRoot.w_366_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_366_0

noncomputable def generatedRoot.w_367_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_367_0

noncomputable def generatedRoot.w_368_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_368_0

noncomputable def generatedRoot.w_369_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_369_0

noncomputable def generatedRoot.w_370_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_370_0

noncomputable def generatedRoot.w_370_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_370_concat_1

noncomputable def generatedRoot.w_371_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_371_0

noncomputable def generatedRoot.w_372_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_372_0

noncomputable def generatedRoot.w_373_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_373_0

noncomputable def generatedRoot.w_374_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_374_0

noncomputable def generatedRoot.w_374_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_374_concat_1

noncomputable def generatedRoot.w_375_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_375_0

noncomputable def generatedRoot.w_376_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_376_0

noncomputable def generatedRoot.w_377_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_377_0

noncomputable def generatedRoot.w_378_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_378_0

noncomputable def generatedRoot.w_379_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_375_0 ctx) (generatedRoot.w_378_0 ctx)

noncomputable def generatedRoot.w_380_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_380_0

noncomputable def generatedRoot.w_381_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_381_0

noncomputable def generatedRoot.w_382_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_382_0

noncomputable def generatedRoot.w_383_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_383_0

noncomputable def generatedRoot.w_384_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_384_0

noncomputable def generatedRoot.w_385_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_366_0 ctx) (generatedRoot.w_384_0 ctx)

noncomputable def generatedRoot.w_386_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_386_0

noncomputable def generatedRoot.w_387_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_387_0

noncomputable def generatedRoot.w_388_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_386_0 ctx) (generatedRoot.w_387_0 ctx)

noncomputable def generatedRoot.w_389_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_356_1 ctx) (generatedRoot.w_388_0 ctx)

noncomputable def generatedRoot.w_390_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_390_0

noncomputable def generatedRoot.w_391_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_392_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_390_0 ctx) (generatedRoot.w_391_0 ctx)

noncomputable def generatedRoot.w_393_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_393_0

noncomputable def generatedRoot.w_394_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_394_0

noncomputable def generatedRoot.w_395_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_389_0 ctx) (generatedRoot.w_394_0 ctx)

noncomputable def generatedRoot.w_396_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_396_0

noncomputable def generatedRoot.w_397_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_398_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_396_0 ctx) (generatedRoot.w_397_0 ctx)

noncomputable def generatedRoot.w_399_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_399_0

noncomputable def generatedRoot.w_400_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_400_0

noncomputable def generatedRoot.w_401_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_395_0 ctx) (generatedRoot.w_400_0 ctx)

noncomputable def generatedRoot.w_402_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_400_0 ctx)

noncomputable def generatedRoot.w_403_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_405_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_405_0

noncomputable def generatedRoot.w_407_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_407_0

noncomputable def generatedRoot.w_409_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_409_0

noncomputable def generatedRoot.w_411_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_411_0

noncomputable def generatedRoot.w_414_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_0

noncomputable def generatedRoot.w_414_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_1

noncomputable def generatedRoot.w_414_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_2

noncomputable def generatedRoot.w_414_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_3

noncomputable def generatedRoot.w_414_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_4

noncomputable def generatedRoot.w_414_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_5

noncomputable def generatedRoot.w_414_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_6

noncomputable def generatedRoot.w_414_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_7

noncomputable def generatedRoot.w_414_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_8

noncomputable def generatedRoot.w_414_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_414_9

noncomputable def generatedRoot.w_415_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_415_0

noncomputable def generatedRoot.w_416_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (0) [(9)])

noncomputable def generatedRoot.w_417_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_417_0

noncomputable def generatedRoot.w_418_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_418_0

noncomputable def generatedRoot.w_419_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_419_0

noncomputable def generatedRoot.w_419_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_419_concat_1

noncomputable def generatedRoot.w_420_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_420_0

noncomputable def generatedRoot.w_421_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_421_0

noncomputable def generatedRoot.w_422_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_422_0

noncomputable def generatedRoot.w_423_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_423_0

noncomputable def generatedRoot.w_423_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_423_concat_1

noncomputable def generatedRoot.w_424_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_424_0

noncomputable def generatedRoot.w_425_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_425_0

noncomputable def generatedRoot.w_426_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_426_0

noncomputable def generatedRoot.w_427_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_427_0

noncomputable def generatedRoot.w_428_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_428_0

noncomputable def generatedRoot.w_428_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_428_concat_1

noncomputable def generatedRoot.w_429_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_429_0

noncomputable def generatedRoot.w_430_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_430_0

noncomputable def generatedRoot.w_431_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_431_0

noncomputable def generatedRoot.w_432_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_432_0

noncomputable def generatedRoot.w_432_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_432_concat_1

noncomputable def generatedRoot.w_433_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_433_0

noncomputable def generatedRoot.w_434_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_434_0

noncomputable def generatedRoot.w_435_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_435_0

noncomputable def generatedRoot.w_436_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_436_0

noncomputable def generatedRoot.w_437_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_433_0 ctx) (generatedRoot.w_436_0 ctx)

noncomputable def generatedRoot.w_438_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_438_0

noncomputable def generatedRoot.w_439_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_439_0

noncomputable def generatedRoot.w_440_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_440_0

noncomputable def generatedRoot.w_441_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_441_0

noncomputable def generatedRoot.w_442_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_442_0

noncomputable def generatedRoot.w_443_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_424_0 ctx) (generatedRoot.w_442_0 ctx)

noncomputable def generatedRoot.w_444_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_444_0

noncomputable def generatedRoot.w_445_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_445_0

noncomputable def generatedRoot.w_446_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_444_0 ctx) (generatedRoot.w_445_0 ctx)

noncomputable def generatedRoot.w_447_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_414_1 ctx) (generatedRoot.w_446_0 ctx)

noncomputable def generatedRoot.w_448_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_448_0

noncomputable def generatedRoot.w_449_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_450_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_448_0 ctx) (generatedRoot.w_449_0 ctx)

noncomputable def generatedRoot.w_451_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_451_0

noncomputable def generatedRoot.w_452_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_452_0

noncomputable def generatedRoot.w_453_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_447_0 ctx) (generatedRoot.w_452_0 ctx)

noncomputable def generatedRoot.w_454_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_454_0

noncomputable def generatedRoot.w_455_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_456_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_454_0 ctx) (generatedRoot.w_455_0 ctx)

noncomputable def generatedRoot.w_457_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_457_0

noncomputable def generatedRoot.w_458_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_458_0

noncomputable def generatedRoot.w_459_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_453_0 ctx) (generatedRoot.w_458_0 ctx)

noncomputable def generatedRoot.w_460_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_458_0 ctx)

noncomputable def generatedRoot.w_461_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_463_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_463_0

noncomputable def generatedRoot.w_465_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_465_0

noncomputable def generatedRoot.w_467_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_467_0

noncomputable def generatedRoot.w_469_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_469_0

noncomputable def generatedRoot.w_472_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_0

noncomputable def generatedRoot.w_472_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_1

noncomputable def generatedRoot.w_472_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_2

noncomputable def generatedRoot.w_472_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_3

noncomputable def generatedRoot.w_472_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_4

noncomputable def generatedRoot.w_472_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_5

noncomputable def generatedRoot.w_472_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_6

noncomputable def generatedRoot.w_472_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_7

noncomputable def generatedRoot.w_472_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_8

noncomputable def generatedRoot.w_472_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_472_9

noncomputable def generatedRoot.w_473_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_473_0

noncomputable def generatedRoot.w_474_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (0) [(11)])

noncomputable def generatedRoot.w_475_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_475_0

noncomputable def generatedRoot.w_476_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_476_0

noncomputable def generatedRoot.w_477_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_477_0

noncomputable def generatedRoot.w_477_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_477_concat_1

noncomputable def generatedRoot.w_478_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_478_0

noncomputable def generatedRoot.w_479_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_479_0

noncomputable def generatedRoot.w_480_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_480_0

noncomputable def generatedRoot.w_481_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_481_0

noncomputable def generatedRoot.w_481_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_481_concat_1

noncomputable def generatedRoot.w_482_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_482_0

noncomputable def generatedRoot.w_483_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_483_0

noncomputable def generatedRoot.w_484_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_484_0

noncomputable def generatedRoot.w_485_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_485_0

noncomputable def generatedRoot.w_486_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_486_0

noncomputable def generatedRoot.w_486_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_486_concat_1

noncomputable def generatedRoot.w_487_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_487_0

noncomputable def generatedRoot.w_488_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_488_0

noncomputable def generatedRoot.w_489_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_489_0

noncomputable def generatedRoot.w_490_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_490_0

noncomputable def generatedRoot.w_490_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_490_concat_1

noncomputable def generatedRoot.w_491_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_491_0

noncomputable def generatedRoot.w_492_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_492_0

noncomputable def generatedRoot.w_493_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_493_0

noncomputable def generatedRoot.w_494_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_494_0

noncomputable def generatedRoot.w_495_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_491_0 ctx) (generatedRoot.w_494_0 ctx)

noncomputable def generatedRoot.w_496_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_496_0

noncomputable def generatedRoot.w_497_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_497_0

noncomputable def generatedRoot.w_498_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_498_0

noncomputable def generatedRoot.w_499_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_499_0

noncomputable def generatedRoot.w_500_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_500_0

noncomputable def generatedRoot.w_501_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_482_0 ctx) (generatedRoot.w_500_0 ctx)

noncomputable def generatedRoot.w_502_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_502_0

noncomputable def generatedRoot.w_503_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_503_0

noncomputable def generatedRoot.w_504_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_502_0 ctx) (generatedRoot.w_503_0 ctx)

noncomputable def generatedRoot.w_505_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_472_1 ctx) (generatedRoot.w_504_0 ctx)

noncomputable def generatedRoot.w_506_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_506_0

noncomputable def generatedRoot.w_507_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_508_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_506_0 ctx) (generatedRoot.w_507_0 ctx)

noncomputable def generatedRoot.w_509_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_509_0

noncomputable def generatedRoot.w_510_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_510_0

noncomputable def generatedRoot.w_511_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_505_0 ctx) (generatedRoot.w_510_0 ctx)

noncomputable def generatedRoot.w_512_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_512_0

noncomputable def generatedRoot.w_513_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_514_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_512_0 ctx) (generatedRoot.w_513_0 ctx)

noncomputable def generatedRoot.w_515_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_515_0

noncomputable def generatedRoot.w_516_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_516_0

noncomputable def generatedRoot.w_517_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_511_0 ctx) (generatedRoot.w_516_0 ctx)

noncomputable def generatedRoot.w_518_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_516_0 ctx)

noncomputable def generatedRoot.w_519_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_521_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_521_0

noncomputable def generatedRoot.w_523_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_523_0

noncomputable def generatedRoot.w_525_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_525_0

noncomputable def generatedRoot.w_527_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_527_0

noncomputable def generatedRoot.w_530_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_0

noncomputable def generatedRoot.w_530_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_1

noncomputable def generatedRoot.w_530_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_2

noncomputable def generatedRoot.w_530_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_3

noncomputable def generatedRoot.w_530_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_4

noncomputable def generatedRoot.w_530_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_5

noncomputable def generatedRoot.w_530_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_6

noncomputable def generatedRoot.w_530_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_7

noncomputable def generatedRoot.w_530_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_8

noncomputable def generatedRoot.w_530_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_530_9

noncomputable def generatedRoot.w_531_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_531_0

noncomputable def generatedRoot.w_532_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (0) [(13)])

noncomputable def generatedRoot.w_533_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_533_0

noncomputable def generatedRoot.w_534_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_534_0

noncomputable def generatedRoot.w_535_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_535_0

noncomputable def generatedRoot.w_535_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_535_concat_1

noncomputable def generatedRoot.w_536_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_536_0

noncomputable def generatedRoot.w_537_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_537_0

noncomputable def generatedRoot.w_538_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_538_0

noncomputable def generatedRoot.w_539_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_539_0

noncomputable def generatedRoot.w_539_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_539_concat_1

noncomputable def generatedRoot.w_540_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_540_0

noncomputable def generatedRoot.w_541_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_541_0

noncomputable def generatedRoot.w_542_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_542_0

noncomputable def generatedRoot.w_543_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_543_0

noncomputable def generatedRoot.w_544_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_544_0

noncomputable def generatedRoot.w_544_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_544_concat_1

noncomputable def generatedRoot.w_545_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_545_0

noncomputable def generatedRoot.w_546_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_546_0

noncomputable def generatedRoot.w_547_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_547_0

noncomputable def generatedRoot.w_548_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_548_0

noncomputable def generatedRoot.w_548_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_548_concat_1

noncomputable def generatedRoot.w_549_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_549_0

noncomputable def generatedRoot.w_550_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_550_0

noncomputable def generatedRoot.w_551_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_551_0

noncomputable def generatedRoot.w_552_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_552_0

noncomputable def generatedRoot.w_553_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_549_0 ctx) (generatedRoot.w_552_0 ctx)

noncomputable def generatedRoot.w_554_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_554_0

noncomputable def generatedRoot.w_555_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_555_0

noncomputable def generatedRoot.w_556_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_556_0

noncomputable def generatedRoot.w_557_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_557_0

noncomputable def generatedRoot.w_558_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_558_0

noncomputable def generatedRoot.w_559_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_540_0 ctx) (generatedRoot.w_558_0 ctx)

noncomputable def generatedRoot.w_560_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_560_0

noncomputable def generatedRoot.w_561_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_561_0

noncomputable def generatedRoot.w_562_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_560_0 ctx) (generatedRoot.w_561_0 ctx)

noncomputable def generatedRoot.w_563_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_530_1 ctx) (generatedRoot.w_562_0 ctx)

noncomputable def generatedRoot.w_564_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_564_0

noncomputable def generatedRoot.w_565_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_566_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_564_0 ctx) (generatedRoot.w_565_0 ctx)

noncomputable def generatedRoot.w_567_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_567_0

noncomputable def generatedRoot.w_568_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_568_0

noncomputable def generatedRoot.w_569_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_563_0 ctx) (generatedRoot.w_568_0 ctx)

noncomputable def generatedRoot.w_570_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_570_0

noncomputable def generatedRoot.w_571_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_572_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_570_0 ctx) (generatedRoot.w_571_0 ctx)

noncomputable def generatedRoot.w_573_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_573_0

noncomputable def generatedRoot.w_574_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_574_0

noncomputable def generatedRoot.w_575_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_569_0 ctx) (generatedRoot.w_574_0 ctx)

noncomputable def generatedRoot.w_576_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_574_0 ctx)

noncomputable def generatedRoot.w_577_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_579_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_579_0

noncomputable def generatedRoot.w_581_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_581_0

noncomputable def generatedRoot.w_583_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_583_0

noncomputable def generatedRoot.w_585_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_585_0

noncomputable def generatedRoot.w_588_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_0

noncomputable def generatedRoot.w_588_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_1

noncomputable def generatedRoot.w_588_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_2

noncomputable def generatedRoot.w_588_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_3

noncomputable def generatedRoot.w_588_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_4

noncomputable def generatedRoot.w_588_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_5

noncomputable def generatedRoot.w_588_6  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_6

noncomputable def generatedRoot.w_588_7  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_7

noncomputable def generatedRoot.w_588_8  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_8

noncomputable def generatedRoot.w_588_9  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_588_9

noncomputable def generatedRoot.w_589_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_589_0

noncomputable def generatedRoot.w_590_0  (ctx : generatedRootCtx) : Int :=
  (MxxIR.selectInt (0) [(15)])

noncomputable def generatedRoot.w_591_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_591_0

noncomputable def generatedRoot.w_592_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_592_0

noncomputable def generatedRoot.w_593_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_593_0

noncomputable def generatedRoot.w_593_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_593_concat_1

noncomputable def generatedRoot.w_594_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_594_0

noncomputable def generatedRoot.w_595_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_595_0

noncomputable def generatedRoot.w_596_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_596_0

noncomputable def generatedRoot.w_597_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_597_0

noncomputable def generatedRoot.w_597_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_597_concat_1

noncomputable def generatedRoot.w_598_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_598_0

noncomputable def generatedRoot.w_599_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_599_0

noncomputable def generatedRoot.w_600_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_600_0

noncomputable def generatedRoot.w_601_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_601_0

noncomputable def generatedRoot.w_602_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_602_0

noncomputable def generatedRoot.w_602_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_602_concat_1

noncomputable def generatedRoot.w_603_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_603_0

noncomputable def generatedRoot.w_604_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_604_0

noncomputable def generatedRoot.w_605_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_605_0

noncomputable def generatedRoot.w_606_0  (ctx : generatedRootCtx) : mxxMat2 :=
  ctx.witness.w_606_0

noncomputable def generatedRoot.w_606_concat_1  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_606_concat_1

noncomputable def generatedRoot.w_607_0  (ctx : generatedRootCtx) : mxxMat4 :=
  ctx.witness.w_607_0

noncomputable def generatedRoot.w_608_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_608_0

noncomputable def generatedRoot.w_609_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_609_0

noncomputable def generatedRoot.w_610_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_610_0

noncomputable def generatedRoot.w_611_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_607_0 ctx) (generatedRoot.w_610_0 ctx)

noncomputable def generatedRoot.w_612_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_612_0

noncomputable def generatedRoot.w_613_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_613_0

noncomputable def generatedRoot.w_614_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_614_0

noncomputable def generatedRoot.w_615_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_615_0

noncomputable def generatedRoot.w_616_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_616_0

noncomputable def generatedRoot.w_617_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_598_0 ctx) (generatedRoot.w_616_0 ctx)

noncomputable def generatedRoot.w_618_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_618_0

noncomputable def generatedRoot.w_619_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_619_0

noncomputable def generatedRoot.w_620_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_618_0 ctx) (generatedRoot.w_619_0 ctx)

noncomputable def generatedRoot.w_621_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_588_1 ctx) (generatedRoot.w_620_0 ctx)

noncomputable def generatedRoot.w_622_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_622_0

noncomputable def generatedRoot.w_623_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_624_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_622_0 ctx) (generatedRoot.w_623_0 ctx)

noncomputable def generatedRoot.w_625_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_625_0

noncomputable def generatedRoot.w_626_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_626_0

noncomputable def generatedRoot.w_627_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_621_0 ctx) (generatedRoot.w_626_0 ctx)

noncomputable def generatedRoot.w_628_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_628_0

noncomputable def generatedRoot.w_629_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)

noncomputable def generatedRoot.w_630_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMulScalarRight (generatedRoot.w_628_0 ctx) (generatedRoot.w_629_0 ctx)

noncomputable def generatedRoot.w_631_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_631_0

noncomputable def generatedRoot.w_632_0  (ctx : generatedRootCtx) : mxxMat6 :=
  ctx.witness.w_632_0

noncomputable def generatedRoot.w_633_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_627_0 ctx) (generatedRoot.w_632_0 ctx)

noncomputable def generatedRoot.w_634_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_88_0 ctx) (generatedRoot.w_632_0 ctx)

noncomputable def generatedRoot.w_635_0  (ctx : generatedRootCtx) : mxxMat0 :=
  (0 : mxxMat0)

noncomputable def generatedRoot.w_636_0  (ctx : generatedRootCtx) : Fin 26 → mxxMat0 :=
  ctx.witness.w_636_0

noncomputable def generatedRoot.w_637_0  (ctx : generatedRootCtx) : Fin 5 → Int :=
  ctx.witness.w_637_0

noncomputable def generatedRoot.w_637_1  (ctx : generatedRootCtx) : Fin 5 → Int :=
  ctx.witness.w_637_1

noncomputable def generatedRoot.w_639_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_639_0

noncomputable def generatedRoot.w_642_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_642_0

noncomputable def generatedRoot.w_644_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_644_0

noncomputable def generatedRoot.w_645_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_645_0

noncomputable def generatedRoot.w_646_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_646_0

noncomputable def generatedRoot.w_647_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_647_0

noncomputable def generatedRoot.w_648_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (1 : mxxMat1)

noncomputable def generatedRoot.w_650_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_650_0

noncomputable def generatedRoot.w_651_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_61_0 ctx) (generatedRoot.w_85_0 ctx)

noncomputable def generatedRoot.w_652_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_54_3 ctx) (generatedRoot.w_60_0 ctx)

noncomputable def generatedRoot.w_653_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_652_0 ctx) (generatedRoot.w_85_0 ctx)

noncomputable def generatedRoot.w_654_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_85_0 ctx)

noncomputable def generatedRoot.w_655_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_655_0

noncomputable def generatedRoot.w_656_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_655_0 ctx) (generatedRoot.w_67_0 ctx)

noncomputable def generatedRoot.w_657_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_85_0 ctx)

noncomputable def generatedRoot.w_658_0  (ctx : generatedRootCtx) : Fin 5 → mxxMat0 :=
  ctx.witness.w_658_0

noncomputable def generatedRoot.w_658_1  (ctx : generatedRootCtx) : Fin 5 → mxxMat0 :=
  ctx.witness.w_658_1

noncomputable def generatedRoot.w_658_2  (ctx : generatedRootCtx) : Fin 5 → mxxMat0 :=
  ctx.witness.w_658_2

noncomputable def generatedRoot.w_658_3  (ctx : generatedRootCtx) : Fin 5 → mxxMat0 :=
  ctx.witness.w_658_3

noncomputable def generatedRoot.w_658_4  (ctx : generatedRootCtx) : Fin 5 → mxxMat0 :=
  ctx.witness.w_658_4

noncomputable def generatedRoot.w_658_5  (ctx : generatedRootCtx) : Fin 5 → mxxMat0 :=
  ctx.witness.w_658_5

noncomputable def generatedRoot.w_659_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_17_0 ctx) (generatedRoot.w_77_0 ctx)

noncomputable def generatedRoot.w_660_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_21_0 ctx) (generatedRoot.w_77_0 ctx)

noncomputable def generatedRoot.w_661_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_661_0

noncomputable def generatedRoot.w_662_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_161_2 ctx) (generatedRoot.w_661_0 ctx)

noncomputable def generatedRoot.w_663_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_662_0 ctx) (generatedRoot.w_199_0 ctx)

noncomputable def generatedRoot.w_664_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_664_0

noncomputable def generatedRoot.w_665_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_161_3 ctx) (generatedRoot.w_664_0 ctx)

noncomputable def generatedRoot.w_666_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_665_0 ctx) (generatedRoot.w_199_0 ctx)

noncomputable def generatedRoot.w_667_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_54_1 ctx) (generatedRoot.w_220_0 ctx)

noncomputable def generatedRoot.w_668_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_667_0 ctx) (generatedRoot.w_226_0 ctx)

noncomputable def generatedRoot.w_669_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_226_0 ctx)

noncomputable def generatedRoot.w_670_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_670_0

noncomputable def generatedRoot.w_671_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_240_3 ctx) (generatedRoot.w_670_0 ctx)

noncomputable def generatedRoot.w_672_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_671_0 ctx) (generatedRoot.w_278_0 ctx)

noncomputable def generatedRoot.w_673_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_672_0 ctx) (generatedRoot.w_284_0 ctx)

noncomputable def generatedRoot.w_674_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_284_0 ctx)

noncomputable def generatedRoot.w_675_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_675_0

noncomputable def generatedRoot.w_676_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_298_3 ctx) (generatedRoot.w_675_0 ctx)

noncomputable def generatedRoot.w_677_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_676_0 ctx) (generatedRoot.w_336_0 ctx)

noncomputable def generatedRoot.w_678_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_677_0 ctx) (generatedRoot.w_342_0 ctx)

noncomputable def generatedRoot.w_679_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_342_0 ctx)

noncomputable def generatedRoot.w_680_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_680_0

noncomputable def generatedRoot.w_681_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_356_3 ctx) (generatedRoot.w_680_0 ctx)

noncomputable def generatedRoot.w_682_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_681_0 ctx) (generatedRoot.w_394_0 ctx)

noncomputable def generatedRoot.w_683_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_682_0 ctx) (generatedRoot.w_400_0 ctx)

noncomputable def generatedRoot.w_684_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_400_0 ctx)

noncomputable def generatedRoot.w_685_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_685_0

noncomputable def generatedRoot.w_686_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_414_3 ctx) (generatedRoot.w_685_0 ctx)

noncomputable def generatedRoot.w_687_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_686_0 ctx) (generatedRoot.w_452_0 ctx)

noncomputable def generatedRoot.w_688_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_687_0 ctx) (generatedRoot.w_458_0 ctx)

noncomputable def generatedRoot.w_689_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_458_0 ctx)

noncomputable def generatedRoot.w_690_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_690_0

noncomputable def generatedRoot.w_691_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_472_3 ctx) (generatedRoot.w_690_0 ctx)

noncomputable def generatedRoot.w_692_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_691_0 ctx) (generatedRoot.w_510_0 ctx)

noncomputable def generatedRoot.w_693_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_692_0 ctx) (generatedRoot.w_516_0 ctx)

noncomputable def generatedRoot.w_694_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_516_0 ctx)

noncomputable def generatedRoot.w_695_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_695_0

noncomputable def generatedRoot.w_696_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_530_3 ctx) (generatedRoot.w_695_0 ctx)

noncomputable def generatedRoot.w_697_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_696_0 ctx) (generatedRoot.w_568_0 ctx)

noncomputable def generatedRoot.w_698_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_697_0 ctx) (generatedRoot.w_574_0 ctx)

noncomputable def generatedRoot.w_699_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_574_0 ctx)

noncomputable def generatedRoot.w_700_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_700_0

noncomputable def generatedRoot.w_701_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_588_3 ctx) (generatedRoot.w_700_0 ctx)

noncomputable def generatedRoot.w_702_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_701_0 ctx) (generatedRoot.w_626_0 ctx)

noncomputable def generatedRoot.w_703_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_702_0 ctx) (generatedRoot.w_632_0 ctx)

noncomputable def generatedRoot.w_704_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_68_0 ctx) (generatedRoot.w_632_0 ctx)

noncomputable def generatedRoot.w_705_0  (ctx : generatedRootCtx) : Fin 26 → mxxMat0 :=
  ctx.witness.w_705_0

noncomputable def generatedRoot.w_706_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_18_0 ctx) (generatedRoot.w_77_0 ctx)

noncomputable def generatedRoot.w_707_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_22_0 ctx) (generatedRoot.w_77_0 ctx)

noncomputable def generatedRoot.w_708_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_708_0

noncomputable def generatedRoot.w_709_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_161_4 ctx) (generatedRoot.w_708_0 ctx)

noncomputable def generatedRoot.w_710_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_709_0 ctx) (generatedRoot.w_199_0 ctx)

noncomputable def generatedRoot.w_711_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_711_0

noncomputable def generatedRoot.w_712_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_161_5 ctx) (generatedRoot.w_711_0 ctx)

noncomputable def generatedRoot.w_713_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_712_0 ctx) (generatedRoot.w_199_0 ctx)

noncomputable def generatedRoot.w_714_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_54_3 ctx) (generatedRoot.w_220_0 ctx)

noncomputable def generatedRoot.w_715_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_714_0 ctx) (generatedRoot.w_226_0 ctx)

noncomputable def generatedRoot.w_716_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_226_0 ctx)

noncomputable def generatedRoot.w_717_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_717_0

noncomputable def generatedRoot.w_718_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_240_5 ctx) (generatedRoot.w_717_0 ctx)

noncomputable def generatedRoot.w_719_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_718_0 ctx) (generatedRoot.w_278_0 ctx)

noncomputable def generatedRoot.w_720_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_719_0 ctx) (generatedRoot.w_284_0 ctx)

noncomputable def generatedRoot.w_721_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_284_0 ctx)

noncomputable def generatedRoot.w_722_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_722_0

noncomputable def generatedRoot.w_723_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_298_5 ctx) (generatedRoot.w_722_0 ctx)

noncomputable def generatedRoot.w_724_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_723_0 ctx) (generatedRoot.w_336_0 ctx)

noncomputable def generatedRoot.w_725_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_724_0 ctx) (generatedRoot.w_342_0 ctx)

noncomputable def generatedRoot.w_726_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_342_0 ctx)

noncomputable def generatedRoot.w_727_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_727_0

noncomputable def generatedRoot.w_728_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_356_5 ctx) (generatedRoot.w_727_0 ctx)

noncomputable def generatedRoot.w_729_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_728_0 ctx) (generatedRoot.w_394_0 ctx)

noncomputable def generatedRoot.w_730_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_729_0 ctx) (generatedRoot.w_400_0 ctx)

noncomputable def generatedRoot.w_731_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_400_0 ctx)

noncomputable def generatedRoot.w_732_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_732_0

noncomputable def generatedRoot.w_733_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_414_5 ctx) (generatedRoot.w_732_0 ctx)

noncomputable def generatedRoot.w_734_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_733_0 ctx) (generatedRoot.w_452_0 ctx)

noncomputable def generatedRoot.w_735_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_734_0 ctx) (generatedRoot.w_458_0 ctx)

noncomputable def generatedRoot.w_736_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_458_0 ctx)

noncomputable def generatedRoot.w_737_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_737_0

noncomputable def generatedRoot.w_738_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_472_5 ctx) (generatedRoot.w_737_0 ctx)

noncomputable def generatedRoot.w_739_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_738_0 ctx) (generatedRoot.w_510_0 ctx)

noncomputable def generatedRoot.w_740_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_739_0 ctx) (generatedRoot.w_516_0 ctx)

noncomputable def generatedRoot.w_741_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_516_0 ctx)

noncomputable def generatedRoot.w_742_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_742_0

noncomputable def generatedRoot.w_743_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_530_5 ctx) (generatedRoot.w_742_0 ctx)

noncomputable def generatedRoot.w_744_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_743_0 ctx) (generatedRoot.w_568_0 ctx)

noncomputable def generatedRoot.w_745_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_744_0 ctx) (generatedRoot.w_574_0 ctx)

noncomputable def generatedRoot.w_746_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_574_0 ctx)

noncomputable def generatedRoot.w_747_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_747_0

noncomputable def generatedRoot.w_748_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_588_5 ctx) (generatedRoot.w_747_0 ctx)

noncomputable def generatedRoot.w_749_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_748_0 ctx) (generatedRoot.w_626_0 ctx)

noncomputable def generatedRoot.w_750_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_749_0 ctx) (generatedRoot.w_632_0 ctx)

noncomputable def generatedRoot.w_751_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixMul (generatedRoot.w_656_0 ctx) (generatedRoot.w_632_0 ctx)

noncomputable def generatedRoot.w_752_0  (ctx : generatedRootCtx) : Fin 26 → mxxMat0 :=
  ctx.witness.w_752_0

noncomputable def generatedRoot.w_754_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_754_0

noncomputable def generatedRoot.w_755_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_755_0

noncomputable def generatedRoot.w_756_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_756_0

noncomputable def generatedRoot.w_757_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_757_0

noncomputable def generatedRoot.w_758_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_758_0

noncomputable def generatedRoot.w_759_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_759_0

noncomputable def generatedRoot.w_760_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_760_0

noncomputable def generatedRoot.w_761_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_761_0

noncomputable def generatedRoot.w_762_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_762_0

noncomputable def generatedRoot.w_763_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_763_0

noncomputable def generatedRoot.w_764_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_764_0

noncomputable def generatedRoot.w_765_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_765_0

noncomputable def generatedRoot.w_766_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_766_0

noncomputable def generatedRoot.w_767_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_767_0

noncomputable def generatedRoot.w_768_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_768_0

noncomputable def generatedRoot.w_769_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_769_0

noncomputable def generatedRoot.w_770_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_770_0

noncomputable def generatedRoot.w_771_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_771_0

noncomputable def generatedRoot.w_772_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_772_0

noncomputable def generatedRoot.w_773_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_773_0

noncomputable def generatedRoot.w_774_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_774_0

noncomputable def generatedRoot.w_775_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_775_0

noncomputable def generatedRoot.w_776_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_776_0

noncomputable def generatedRoot.w_777_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_777_0

noncomputable def generatedRoot.w_778_0  (ctx : generatedRootCtx) : Int :=
  1

noncomputable def generatedRoot.w_779_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_779_0

noncomputable def generatedRoot.w_780_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_780_0

noncomputable def generatedRoot.w_781_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_781_0

noncomputable def generatedRoot.w_782_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_782_0

noncomputable def generatedRoot.w_783_0  (ctx : generatedRootCtx) : Int :=
  1

noncomputable def generatedRoot.w_784_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_784_0

noncomputable def generatedRoot.w_785_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_785_0

noncomputable def generatedRoot.w_786_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_786_0

noncomputable def generatedRoot.w_787_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_787_0

noncomputable def generatedRoot.w_788_0  (ctx : generatedRootCtx) : Int :=
  3

noncomputable def generatedRoot.w_789_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_789_0

noncomputable def generatedRoot.w_790_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_790_0

noncomputable def generatedRoot.w_791_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_791_0

noncomputable def generatedRoot.w_792_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_792_0

noncomputable def generatedRoot.w_793_0  (ctx : generatedRootCtx) : Int :=
  3

noncomputable def generatedRoot.w_794_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_794_0

noncomputable def generatedRoot.w_795_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_795_0

noncomputable def generatedRoot.w_796_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_796_0

noncomputable def generatedRoot.w_797_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_797_0

noncomputable def generatedRoot.w_798_0  (ctx : generatedRootCtx) : Int :=
  5

noncomputable def generatedRoot.w_799_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_799_0

noncomputable def generatedRoot.w_800_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_800_0

noncomputable def generatedRoot.w_801_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_801_0

noncomputable def generatedRoot.w_802_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_802_0

noncomputable def generatedRoot.w_803_0  (ctx : generatedRootCtx) : Int :=
  5

noncomputable def generatedRoot.w_804_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_804_0

noncomputable def generatedRoot.w_805_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_805_0

noncomputable def generatedRoot.w_806_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_806_0

noncomputable def generatedRoot.w_807_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_807_0

noncomputable def generatedRoot.w_808_0  (ctx : generatedRootCtx) : Int :=
  7

noncomputable def generatedRoot.w_809_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_809_0

noncomputable def generatedRoot.w_810_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_810_0

noncomputable def generatedRoot.w_811_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_811_0

noncomputable def generatedRoot.w_812_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_812_0

noncomputable def generatedRoot.w_813_0  (ctx : generatedRootCtx) : Int :=
  7

noncomputable def generatedRoot.w_814_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_814_0

noncomputable def generatedRoot.w_815_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_815_0

noncomputable def generatedRoot.w_816_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_816_0

noncomputable def generatedRoot.w_817_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_817_0

noncomputable def generatedRoot.w_818_0  (ctx : generatedRootCtx) : Int :=
  9

noncomputable def generatedRoot.w_819_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_819_0

noncomputable def generatedRoot.w_820_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_820_0

noncomputable def generatedRoot.w_821_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_821_0

noncomputable def generatedRoot.w_822_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_822_0

noncomputable def generatedRoot.w_823_0  (ctx : generatedRootCtx) : Int :=
  9

noncomputable def generatedRoot.w_824_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_824_0

noncomputable def generatedRoot.w_825_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_825_0

noncomputable def generatedRoot.w_826_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_826_0

noncomputable def generatedRoot.w_827_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_827_0

noncomputable def generatedRoot.w_828_0  (ctx : generatedRootCtx) : Int :=
  11

noncomputable def generatedRoot.w_829_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_829_0

noncomputable def generatedRoot.w_830_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_830_0

noncomputable def generatedRoot.w_831_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_831_0

noncomputable def generatedRoot.w_832_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_832_0

noncomputable def generatedRoot.w_833_0  (ctx : generatedRootCtx) : Int :=
  11

noncomputable def generatedRoot.w_834_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_834_0

noncomputable def generatedRoot.w_835_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_835_0

noncomputable def generatedRoot.w_836_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_836_0

noncomputable def generatedRoot.w_837_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_837_0

noncomputable def generatedRoot.w_838_0  (ctx : generatedRootCtx) : Int :=
  13

noncomputable def generatedRoot.w_839_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_839_0

noncomputable def generatedRoot.w_840_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_840_0

noncomputable def generatedRoot.w_841_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_841_0

noncomputable def generatedRoot.w_842_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_842_0

noncomputable def generatedRoot.w_843_0  (ctx : generatedRootCtx) : Int :=
  13

noncomputable def generatedRoot.w_844_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_844_0

noncomputable def generatedRoot.w_845_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_845_0

noncomputable def generatedRoot.w_846_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_846_0

noncomputable def generatedRoot.w_847_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_847_0

noncomputable def generatedRoot.w_848_0  (ctx : generatedRootCtx) : Int :=
  15

noncomputable def generatedRoot.w_849_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_849_0

noncomputable def generatedRoot.w_850_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_850_0

noncomputable def generatedRoot.w_851_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_851_0

noncomputable def generatedRoot.w_852_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_852_0

noncomputable def generatedRoot.w_853_0  (ctx : generatedRootCtx) : Int :=
  15

noncomputable def generatedRoot.w_854_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_854_0

noncomputable def generatedRoot.w_855_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_855_0

noncomputable def generatedRoot.w_856_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_856_0

noncomputable def generatedRoot.w_857_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_857_0

noncomputable def generatedRoot.w_859_0  (ctx : generatedRootCtx) : mxxMat28 :=
  ctx.witness.w_859_0

noncomputable def generatedRoot.w_860_0  (ctx : generatedRootCtx) : mxxMat29 :=
  ctx.witness.w_860_0

noncomputable def generatedRoot.w_861_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_861_0

noncomputable def generatedRoot.w_862_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_862_0

noncomputable def generatedRoot.w_863_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_863_0

noncomputable def generatedRoot.w_864_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_864_0

noncomputable def generatedRoot.w_865_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_865_0

noncomputable def generatedRoot.w_866_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_866_0

noncomputable def generatedRoot.w_867_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_867_0

noncomputable def generatedRoot.w_868_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_868_0

noncomputable def generatedRoot.w_869_0  (ctx : generatedRootCtx) : Fin 26 → mxxMat3 :=
  ctx.witness.w_869_0

noncomputable def generatedRoot.w_870_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat0 :=
  ctx.witness.w_870_0

noncomputable def generatedRoot.w_870_1  (ctx : generatedRootCtx) : Fin 36 → mxxMat0 :=
  ctx.witness.w_870_1

noncomputable def generatedRoot.w_870_2  (ctx : generatedRootCtx) : Fin 36 → mxxMat0 :=
  ctx.witness.w_870_2

noncomputable def generatedRoot.w_870_3  (ctx : generatedRootCtx) : Fin 36 → mxxMat3 :=
  ctx.witness.w_870_3

noncomputable def generatedRoot.w_871_0  (ctx : generatedRootCtx) : mxxMat8 :=
  (MxxRuntime.unitRow (0) : mxxMat8)

noncomputable def generatedRoot.w_880_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_880_0

noncomputable def generatedRoot.w_881_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_881_0

noncomputable def generatedRoot.w_884_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_884_0

noncomputable def generatedRoot.w_885_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_885_0

noncomputable def generatedRoot.w_889_0  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_889_0

noncomputable def generatedRoot.w_889_1  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_889_1

noncomputable def generatedRoot.w_889_2  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_889_2

noncomputable def generatedRoot.w_889_3  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_889_3

noncomputable def generatedRoot.w_889_4  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_889_4

noncomputable def generatedRoot.w_889_5  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_889_5

noncomputable def generatedRoot.w_890_0  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_890_0

noncomputable def generatedRoot.w_890_1  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_890_1

noncomputable def generatedRoot.w_890_2  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_890_2

noncomputable def generatedRoot.w_890_3  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_890_3

noncomputable def generatedRoot.w_890_4  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_890_4

noncomputable def generatedRoot.w_890_5  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_890_5

noncomputable def generatedRoot.w_893_0  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_893_0

noncomputable def generatedRoot.w_893_1  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_893_1

noncomputable def generatedRoot.w_893_2  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_893_2

noncomputable def generatedRoot.w_893_3  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_893_3

noncomputable def generatedRoot.w_893_4  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_893_4

noncomputable def generatedRoot.w_893_5  (ctx : generatedRootCtx) : Fin 14400 → mxxMat0 :=
  ctx.witness.w_893_5

noncomputable def generatedRoot.w_894_0  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_894_0

noncomputable def generatedRoot.w_894_1  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_894_1

noncomputable def generatedRoot.w_894_2  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_894_2

noncomputable def generatedRoot.w_894_3  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_894_3

noncomputable def generatedRoot.w_894_4  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_894_4

noncomputable def generatedRoot.w_894_5  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_894_5

noncomputable def generatedRoot.w_895_0  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_895_0

noncomputable def generatedRoot.w_895_1  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_895_1

noncomputable def generatedRoot.w_895_2  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_895_2

noncomputable def generatedRoot.w_895_3  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_895_3

noncomputable def generatedRoot.w_895_4  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_895_4

noncomputable def generatedRoot.w_895_5  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_895_5

noncomputable def generatedRoot.w_896_0  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_896_0

noncomputable def generatedRoot.w_896_1  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_896_1

noncomputable def generatedRoot.w_896_2  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_896_2

noncomputable def generatedRoot.w_896_3  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_896_3

noncomputable def generatedRoot.w_896_4  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_896_4

noncomputable def generatedRoot.w_896_5  (ctx : generatedRootCtx) : Fin 7200 → mxxMat0 :=
  ctx.witness.w_896_5

noncomputable def generatedRoot.w_897_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_897_0

noncomputable def generatedRoot.w_898_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_898_0

noncomputable def generatedRoot.w_899_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_897_0 ctx) (generatedRoot.w_898_0 ctx)

noncomputable def generatedRoot.w_900_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_900_0

noncomputable def generatedRoot.w_901_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_901_0

noncomputable def generatedRoot.w_902_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_900_0 ctx) (generatedRoot.w_901_0 ctx)

noncomputable def generatedRoot.w_903_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_903_0

noncomputable def generatedRoot.w_904_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_904_0

noncomputable def generatedRoot.w_905_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_903_0 ctx) (generatedRoot.w_904_0 ctx)

noncomputable def generatedRoot.w_906_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_906_0

noncomputable def generatedRoot.w_907_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_907_0

noncomputable def generatedRoot.w_908_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_906_0 ctx) (generatedRoot.w_907_0 ctx)

noncomputable def generatedRoot.w_909_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_909_0

noncomputable def generatedRoot.w_910_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_910_0

noncomputable def generatedRoot.w_911_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_909_0 ctx) (generatedRoot.w_910_0 ctx)

noncomputable def generatedRoot.w_912_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_912_0

noncomputable def generatedRoot.w_913_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_913_0

noncomputable def generatedRoot.w_914_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_912_0 ctx) (generatedRoot.w_913_0 ctx)

noncomputable def generatedRoot.w_915_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_915_0

noncomputable def generatedRoot.w_915_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_915_1

noncomputable def generatedRoot.w_915_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_915_2

noncomputable def generatedRoot.w_915_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_915_3

noncomputable def generatedRoot.w_915_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_915_4

noncomputable def generatedRoot.w_915_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_915_5

noncomputable def generatedRoot.w_916_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_916_0

noncomputable def generatedRoot.w_917_0  (ctx : generatedRootCtx) : mxxMat8 :=
  (MxxRuntime.unitRow (0) : mxxMat8)

noncomputable def generatedRoot.w_918_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_918_0

noncomputable def generatedRoot.w_919_0  (ctx : generatedRootCtx) : mxxMat8 :=
  (MxxRuntime.unitRow (0) : mxxMat8)

noncomputable def generatedRoot.w_920_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_920_0

noncomputable def generatedRoot.w_921_0  (ctx : generatedRootCtx) : mxxMat8 :=
  (MxxRuntime.unitRow (0) : mxxMat8)

noncomputable def generatedRoot.w_922_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_922_0

noncomputable def generatedRoot.w_923_0  (ctx : generatedRootCtx) : mxxMat8 :=
  (MxxRuntime.unitRow (0) : mxxMat8)

noncomputable def generatedRoot.w_924_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_924_0

noncomputable def generatedRoot.w_925_0  (ctx : generatedRootCtx) : mxxMat8 :=
  (MxxRuntime.unitRow (0) : mxxMat8)

noncomputable def generatedRoot.w_926_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_926_0

noncomputable def generatedRoot.w_927_0  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_927_0

noncomputable def generatedRoot.w_927_1  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_927_1

noncomputable def generatedRoot.w_927_2  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_927_2

noncomputable def generatedRoot.w_927_3  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_927_3

noncomputable def generatedRoot.w_927_4  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_927_4

noncomputable def generatedRoot.w_927_5  (ctx : generatedRootCtx) : mxxMat7 :=
  ctx.witness.w_927_5

noncomputable def generatedRoot.w_928_0  (ctx : generatedRootCtx) : mxxMat10 :=
  (MxxRuntime.unitRow (0) : mxxMat10)

noncomputable def generatedRoot.w_935_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_935_0

noncomputable def generatedRoot.w_936_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_936_0

noncomputable def generatedRoot.w_938_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_938_0

noncomputable def generatedRoot.w_939_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_939_0

noncomputable def generatedRoot.w_943_0  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_943_0

noncomputable def generatedRoot.w_943_1  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_943_1

noncomputable def generatedRoot.w_943_2  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_943_2

noncomputable def generatedRoot.w_943_3  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_943_3

noncomputable def generatedRoot.w_943_4  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_943_4

noncomputable def generatedRoot.w_943_5  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_943_5

noncomputable def generatedRoot.w_944_0  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_944_0

noncomputable def generatedRoot.w_944_1  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_944_1

noncomputable def generatedRoot.w_944_2  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_944_2

noncomputable def generatedRoot.w_944_3  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_944_3

noncomputable def generatedRoot.w_944_4  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_944_4

noncomputable def generatedRoot.w_944_5  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_944_5

noncomputable def generatedRoot.w_947_0  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_947_0

noncomputable def generatedRoot.w_947_1  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_947_1

noncomputable def generatedRoot.w_947_2  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_947_2

noncomputable def generatedRoot.w_947_3  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_947_3

noncomputable def generatedRoot.w_947_4  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_947_4

noncomputable def generatedRoot.w_947_5  (ctx : generatedRootCtx) : Fin 72000 → mxxMat0 :=
  ctx.witness.w_947_5

noncomputable def generatedRoot.w_948_0  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_948_0

noncomputable def generatedRoot.w_948_1  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_948_1

noncomputable def generatedRoot.w_948_2  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_948_2

noncomputable def generatedRoot.w_948_3  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_948_3

noncomputable def generatedRoot.w_948_4  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_948_4

noncomputable def generatedRoot.w_948_5  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_948_5

noncomputable def generatedRoot.w_949_0  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_949_0

noncomputable def generatedRoot.w_949_1  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_949_1

noncomputable def generatedRoot.w_949_2  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_949_2

noncomputable def generatedRoot.w_949_3  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_949_3

noncomputable def generatedRoot.w_949_4  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_949_4

noncomputable def generatedRoot.w_949_5  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_949_5

noncomputable def generatedRoot.w_950_0  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_950_0

noncomputable def generatedRoot.w_950_1  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_950_1

noncomputable def generatedRoot.w_950_2  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_950_2

noncomputable def generatedRoot.w_950_3  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_950_3

noncomputable def generatedRoot.w_950_4  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_950_4

noncomputable def generatedRoot.w_950_5  (ctx : generatedRootCtx) : Fin 36000 → mxxMat0 :=
  ctx.witness.w_950_5

noncomputable def generatedRoot.w_951_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_951_0

noncomputable def generatedRoot.w_952_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_952_0

noncomputable def generatedRoot.w_953_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_951_0 ctx) (generatedRoot.w_952_0 ctx)

noncomputable def generatedRoot.w_954_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_954_0

noncomputable def generatedRoot.w_955_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_955_0

noncomputable def generatedRoot.w_956_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_954_0 ctx) (generatedRoot.w_955_0 ctx)

noncomputable def generatedRoot.w_957_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_957_0

noncomputable def generatedRoot.w_958_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_958_0

noncomputable def generatedRoot.w_959_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_957_0 ctx) (generatedRoot.w_958_0 ctx)

noncomputable def generatedRoot.w_960_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_960_0

noncomputable def generatedRoot.w_961_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_961_0

noncomputable def generatedRoot.w_962_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_960_0 ctx) (generatedRoot.w_961_0 ctx)

noncomputable def generatedRoot.w_963_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_963_0

noncomputable def generatedRoot.w_964_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_964_0

noncomputable def generatedRoot.w_965_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_963_0 ctx) (generatedRoot.w_964_0 ctx)

noncomputable def generatedRoot.w_966_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_966_0

noncomputable def generatedRoot.w_967_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_967_0

noncomputable def generatedRoot.w_968_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_966_0 ctx) (generatedRoot.w_967_0 ctx)

noncomputable def generatedRoot.w_969_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_969_0

noncomputable def generatedRoot.w_969_1  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_969_1

noncomputable def generatedRoot.w_969_2  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_969_2

noncomputable def generatedRoot.w_969_3  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_969_3

noncomputable def generatedRoot.w_969_4  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_969_4

noncomputable def generatedRoot.w_969_5  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_969_5

noncomputable def generatedRoot.w_970_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_970_0

noncomputable def generatedRoot.w_971_0  (ctx : generatedRootCtx) : mxxMat10 :=
  (MxxRuntime.unitRow (0) : mxxMat10)

noncomputable def generatedRoot.w_972_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_972_0

noncomputable def generatedRoot.w_973_0  (ctx : generatedRootCtx) : mxxMat10 :=
  (MxxRuntime.unitRow (0) : mxxMat10)

noncomputable def generatedRoot.w_974_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_974_0

noncomputable def generatedRoot.w_975_0  (ctx : generatedRootCtx) : mxxMat10 :=
  (MxxRuntime.unitRow (0) : mxxMat10)

noncomputable def generatedRoot.w_976_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_976_0

noncomputable def generatedRoot.w_977_0  (ctx : generatedRootCtx) : mxxMat10 :=
  (MxxRuntime.unitRow (0) : mxxMat10)

noncomputable def generatedRoot.w_978_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_978_0

noncomputable def generatedRoot.w_979_0  (ctx : generatedRootCtx) : mxxMat10 :=
  (MxxRuntime.unitRow (0) : mxxMat10)

noncomputable def generatedRoot.w_980_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_980_0

noncomputable def generatedRoot.w_981_0  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_981_0

noncomputable def generatedRoot.w_981_1  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_981_1

noncomputable def generatedRoot.w_981_2  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_981_2

noncomputable def generatedRoot.w_981_3  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_981_3

noncomputable def generatedRoot.w_981_4  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_981_4

noncomputable def generatedRoot.w_981_5  (ctx : generatedRootCtx) : mxxMat9 :=
  ctx.witness.w_981_5

noncomputable def generatedRoot.w_982_0  (ctx : generatedRootCtx) : mxxMat1 :=
  (MxxRuntime.matrixPolynomial [(mxxNat8)] : mxxMat1)

noncomputable def generatedRoot.w_983_0  (ctx : generatedRootCtx) : Fin 1 → mxxMat1 :=
  ctx.witness.w_983_0

noncomputable def generatedRoot.w_984_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat3 :=
  ctx.witness.w_984_0

noncomputable def generatedRoot.w_985_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat3 :=
  ctx.witness.w_985_0

noncomputable def generatedRoot.w_986_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat3 :=
  ctx.witness.w_986_0

noncomputable def generatedRoot.w_987_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat3 :=
  ctx.witness.w_987_0

noncomputable def generatedRoot.w_988_0  (ctx : generatedRootCtx) : Fin 36 → mxxMat3 :=
  ctx.witness.w_988_0

noncomputable def generatedRoot.w_989_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_989_0

noncomputable def generatedRoot.w_990_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_990_0

noncomputable def generatedRoot.w_991_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_991_0

noncomputable def generatedRoot.w_992_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_549_0 ctx) (generatedRoot.w_991_0 ctx)

noncomputable def generatedRoot.w_993_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_993_0

noncomputable def generatedRoot.w_994_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_994_0

noncomputable def generatedRoot.w_995_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_995_0

noncomputable def generatedRoot.w_996_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_996_0

noncomputable def generatedRoot.w_997_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_997_0

noncomputable def generatedRoot.w_998_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_540_0 ctx) (generatedRoot.w_997_0 ctx)

noncomputable def generatedRoot.w_999_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_999_0

noncomputable def generatedRoot.w_1000_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_530_2 ctx) (generatedRoot.w_999_0 ctx)

noncomputable def generatedRoot.w_1001_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1001_0

noncomputable def generatedRoot.w_1002_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_530_4 ctx) (generatedRoot.w_1001_0 ctx)

noncomputable def generatedRoot.w_1003_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1003_0

noncomputable def generatedRoot.w_1004_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1004_0

noncomputable def generatedRoot.w_1005_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_1003_0 ctx) (generatedRoot.w_1004_0 ctx)

noncomputable def generatedRoot.w_1006_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_530_0 ctx) (generatedRoot.w_1005_0 ctx)

noncomputable def generatedRoot.w_1007_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1007_0

noncomputable def generatedRoot.w_1008_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1008_0

noncomputable def generatedRoot.w_1009_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1009_0

noncomputable def generatedRoot.w_1010_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_607_0 ctx) (generatedRoot.w_1009_0 ctx)

noncomputable def generatedRoot.w_1011_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1011_0

noncomputable def generatedRoot.w_1012_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1012_0

noncomputable def generatedRoot.w_1013_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1013_0

noncomputable def generatedRoot.w_1014_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1014_0

noncomputable def generatedRoot.w_1015_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1015_0

noncomputable def generatedRoot.w_1016_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_598_0 ctx) (generatedRoot.w_1015_0 ctx)

noncomputable def generatedRoot.w_1017_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1017_0

noncomputable def generatedRoot.w_1018_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_588_2 ctx) (generatedRoot.w_1017_0 ctx)

noncomputable def generatedRoot.w_1019_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1019_0

noncomputable def generatedRoot.w_1020_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_588_4 ctx) (generatedRoot.w_1019_0 ctx)

noncomputable def generatedRoot.w_1021_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1021_0

noncomputable def generatedRoot.w_1022_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1022_0

noncomputable def generatedRoot.w_1023_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_1021_0 ctx) (generatedRoot.w_1022_0 ctx)

noncomputable def generatedRoot.w_1024_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_588_0 ctx) (generatedRoot.w_1023_0 ctx)

noncomputable def generatedRoot.w_1025_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1025_0

noncomputable def generatedRoot.w_1026_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1026_0

noncomputable def generatedRoot.w_1027_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1027_0

noncomputable def generatedRoot.w_1028_0  (ctx : generatedRootCtx) : mxxMat1 :=
  ctx.witness.w_1028_0

noncomputable def generatedRoot.w_1029_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1029_0

noncomputable def generatedRoot.w_1030_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1030_0

noncomputable def generatedRoot.w_1031_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1031_0

noncomputable def generatedRoot.w_1032_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_259_0 ctx) (generatedRoot.w_1031_0 ctx)

noncomputable def generatedRoot.w_1033_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1033_0

noncomputable def generatedRoot.w_1034_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1034_0

noncomputable def generatedRoot.w_1035_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1035_0

noncomputable def generatedRoot.w_1036_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1036_0

noncomputable def generatedRoot.w_1037_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1037_0

noncomputable def generatedRoot.w_1038_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_250_0 ctx) (generatedRoot.w_1037_0 ctx)

noncomputable def generatedRoot.w_1039_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1039_0

noncomputable def generatedRoot.w_1040_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_240_2 ctx) (generatedRoot.w_1039_0 ctx)

noncomputable def generatedRoot.w_1041_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1041_0

noncomputable def generatedRoot.w_1042_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_240_4 ctx) (generatedRoot.w_1041_0 ctx)

noncomputable def generatedRoot.w_1043_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1043_0

noncomputable def generatedRoot.w_1044_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1044_0

noncomputable def generatedRoot.w_1045_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_1043_0 ctx) (generatedRoot.w_1044_0 ctx)

noncomputable def generatedRoot.w_1046_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_240_0 ctx) (generatedRoot.w_1045_0 ctx)

noncomputable def generatedRoot.w_1047_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1047_0

noncomputable def generatedRoot.w_1048_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1048_0

noncomputable def generatedRoot.w_1049_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1049_0

noncomputable def generatedRoot.w_1050_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_317_0 ctx) (generatedRoot.w_1049_0 ctx)

noncomputable def generatedRoot.w_1051_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1051_0

noncomputable def generatedRoot.w_1052_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1052_0

noncomputable def generatedRoot.w_1053_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1053_0

noncomputable def generatedRoot.w_1054_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1054_0

noncomputable def generatedRoot.w_1055_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1055_0

noncomputable def generatedRoot.w_1056_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_308_0 ctx) (generatedRoot.w_1055_0 ctx)

noncomputable def generatedRoot.w_1057_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1057_0

noncomputable def generatedRoot.w_1058_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_298_2 ctx) (generatedRoot.w_1057_0 ctx)

noncomputable def generatedRoot.w_1059_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1059_0

noncomputable def generatedRoot.w_1060_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_298_4 ctx) (generatedRoot.w_1059_0 ctx)

noncomputable def generatedRoot.w_1061_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1061_0

noncomputable def generatedRoot.w_1062_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1062_0

noncomputable def generatedRoot.w_1063_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_1061_0 ctx) (generatedRoot.w_1062_0 ctx)

noncomputable def generatedRoot.w_1064_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_298_0 ctx) (generatedRoot.w_1063_0 ctx)

noncomputable def generatedRoot.w_1065_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1065_0

noncomputable def generatedRoot.w_1066_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1066_0

noncomputable def generatedRoot.w_1067_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1067_0

noncomputable def generatedRoot.w_1068_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_375_0 ctx) (generatedRoot.w_1067_0 ctx)

noncomputable def generatedRoot.w_1069_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1069_0

noncomputable def generatedRoot.w_1070_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1070_0

noncomputable def generatedRoot.w_1071_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1071_0

noncomputable def generatedRoot.w_1072_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1072_0

noncomputable def generatedRoot.w_1073_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1073_0

noncomputable def generatedRoot.w_1074_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_366_0 ctx) (generatedRoot.w_1073_0 ctx)

noncomputable def generatedRoot.w_1075_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1075_0

noncomputable def generatedRoot.w_1076_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_356_2 ctx) (generatedRoot.w_1075_0 ctx)

noncomputable def generatedRoot.w_1077_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1077_0

noncomputable def generatedRoot.w_1078_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_356_4 ctx) (generatedRoot.w_1077_0 ctx)

noncomputable def generatedRoot.w_1079_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1079_0

noncomputable def generatedRoot.w_1080_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1080_0

noncomputable def generatedRoot.w_1081_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_1079_0 ctx) (generatedRoot.w_1080_0 ctx)

noncomputable def generatedRoot.w_1082_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_356_0 ctx) (generatedRoot.w_1081_0 ctx)

noncomputable def generatedRoot.w_1083_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1083_0

noncomputable def generatedRoot.w_1084_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1084_0

noncomputable def generatedRoot.w_1085_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1085_0

noncomputable def generatedRoot.w_1086_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_433_0 ctx) (generatedRoot.w_1085_0 ctx)

noncomputable def generatedRoot.w_1087_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1087_0

noncomputable def generatedRoot.w_1088_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1088_0

noncomputable def generatedRoot.w_1089_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1089_0

noncomputable def generatedRoot.w_1090_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1090_0

noncomputable def generatedRoot.w_1091_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1091_0

noncomputable def generatedRoot.w_1092_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_424_0 ctx) (generatedRoot.w_1091_0 ctx)

noncomputable def generatedRoot.w_1093_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1093_0

noncomputable def generatedRoot.w_1094_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_414_2 ctx) (generatedRoot.w_1093_0 ctx)

noncomputable def generatedRoot.w_1095_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1095_0

noncomputable def generatedRoot.w_1096_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_414_4 ctx) (generatedRoot.w_1095_0 ctx)

noncomputable def generatedRoot.w_1097_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1097_0

noncomputable def generatedRoot.w_1098_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1098_0

noncomputable def generatedRoot.w_1099_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_1097_0 ctx) (generatedRoot.w_1098_0 ctx)

noncomputable def generatedRoot.w_1100_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_414_0 ctx) (generatedRoot.w_1099_0 ctx)

noncomputable def generatedRoot.w_1101_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1101_0

noncomputable def generatedRoot.w_1102_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1102_0

noncomputable def generatedRoot.w_1103_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1103_0

noncomputable def generatedRoot.w_1104_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_491_0 ctx) (generatedRoot.w_1103_0 ctx)

noncomputable def generatedRoot.w_1105_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1105_0

noncomputable def generatedRoot.w_1106_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1106_0

noncomputable def generatedRoot.w_1107_0  (ctx : generatedRootCtx) : mxxMat3 :=
  ctx.witness.w_1107_0

noncomputable def generatedRoot.w_1108_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1108_0

noncomputable def generatedRoot.w_1109_0  (ctx : generatedRootCtx) : mxxMat5 :=
  ctx.witness.w_1109_0

noncomputable def generatedRoot.w_1110_0  (ctx : generatedRootCtx) : mxxMat2 :=
  MxxRuntime.matrixMul (generatedRoot.w_482_0 ctx) (generatedRoot.w_1109_0 ctx)

noncomputable def generatedRoot.w_1111_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1111_0

noncomputable def generatedRoot.w_1112_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_472_2 ctx) (generatedRoot.w_1111_0 ctx)

noncomputable def generatedRoot.w_1113_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1113_0

noncomputable def generatedRoot.w_1114_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_472_4 ctx) (generatedRoot.w_1113_0 ctx)

noncomputable def generatedRoot.w_1115_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1115_0

noncomputable def generatedRoot.w_1116_0  (ctx : generatedRootCtx) : mxxMat0 :=
  ctx.witness.w_1116_0

noncomputable def generatedRoot.w_1117_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_1115_0 ctx) (generatedRoot.w_1116_0 ctx)

noncomputable def generatedRoot.w_1118_0  (ctx : generatedRootCtx) : mxxMat0 :=
  MxxRuntime.matrixAdd (generatedRoot.w_472_0 ctx) (generatedRoot.w_1117_0 ctx)

noncomputable def generatedRoot.outs  (ctx : generatedRootCtx) : (generatedRoot.OutputType) :=
  ((generatedRoot.w_984_0 ctx), ((generatedRoot.w_985_0 ctx), ((generatedRoot.w_986_0 ctx), ((generatedRoot.w_987_0 ctx), ((generatedRoot.w_988_0 ctx), ((generatedRoot.w_662_0 ctx), ((generatedRoot.w_665_0 ctx), ((generatedRoot.w_709_0 ctx), ((generatedRoot.w_712_0 ctx), ((generatedRoot.w_194_0 ctx), ((generatedRoot.w_214_0 ctx), ((generatedRoot.w_981_0 ctx), ((generatedRoot.w_981_1 ctx), ((generatedRoot.w_981_2 ctx), ((generatedRoot.w_981_3 ctx), ((generatedRoot.w_981_4 ctx), ((generatedRoot.w_981_5 ctx), ((generatedRoot.w_1000_0 ctx), ((generatedRoot.w_696_0 ctx), ((generatedRoot.w_1002_0 ctx), ((generatedRoot.w_743_0 ctx), ((generatedRoot.w_1006_0 ctx), ((generatedRoot.w_563_0 ctx), ((generatedRoot.w_1018_0 ctx), ((generatedRoot.w_701_0 ctx), ((generatedRoot.w_1020_0 ctx), ((generatedRoot.w_748_0 ctx), ((generatedRoot.w_1024_0 ctx), ((generatedRoot.w_621_0 ctx), ((generatedRoot.w_927_0 ctx), ((generatedRoot.w_927_1 ctx), ((generatedRoot.w_927_2 ctx), ((generatedRoot.w_927_3 ctx), ((generatedRoot.w_927_4 ctx), ((generatedRoot.w_927_5 ctx), ((generatedRoot.w_17_0 ctx), ((generatedRoot.w_21_0 ctx), ((generatedRoot.w_18_0 ctx), ((generatedRoot.w_22_0 ctx), ((generatedRoot.w_19_0 ctx), ((generatedRoot.w_23_0 ctx), ((generatedRoot.w_1025_0 ctx), ((generatedRoot.w_62_0 ctx), ((generatedRoot.w_1026_0 ctx), ((generatedRoot.w_655_0 ctx), ((generatedRoot.w_1027_0 ctx), ((generatedRoot.w_87_0 ctx), ((generatedRoot.w_1028_0 ctx), ((generatedRoot.w_54_0 ctx), ((generatedRoot.w_54_1 ctx), ((generatedRoot.w_54_2 ctx), ((generatedRoot.w_54_3 ctx), ((generatedRoot.w_54_4 ctx), ((generatedRoot.w_54_5 ctx), ((generatedRoot.w_1040_0 ctx), ((generatedRoot.w_671_0 ctx), ((generatedRoot.w_1042_0 ctx), ((generatedRoot.w_718_0 ctx), ((generatedRoot.w_1046_0 ctx), ((generatedRoot.w_273_0 ctx), ((generatedRoot.w_1058_0 ctx), ((generatedRoot.w_676_0 ctx), ((generatedRoot.w_1060_0 ctx), ((generatedRoot.w_723_0 ctx), ((generatedRoot.w_1064_0 ctx), ((generatedRoot.w_331_0 ctx), ((generatedRoot.w_1076_0 ctx), ((generatedRoot.w_681_0 ctx), ((generatedRoot.w_1078_0 ctx), ((generatedRoot.w_728_0 ctx), ((generatedRoot.w_1082_0 ctx), ((generatedRoot.w_389_0 ctx), ((generatedRoot.w_1094_0 ctx), ((generatedRoot.w_686_0 ctx), ((generatedRoot.w_1096_0 ctx), ((generatedRoot.w_733_0 ctx), ((generatedRoot.w_1100_0 ctx), ((generatedRoot.w_447_0 ctx), ((generatedRoot.w_1112_0 ctx), ((generatedRoot.w_691_0 ctx), ((generatedRoot.w_1114_0 ctx), ((generatedRoot.w_738_0 ctx), ((generatedRoot.w_1118_0 ctx), ((generatedRoot.w_505_0 ctx), ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

end

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (generatedRoot.InputType)) (outputs : (generatedRoot.OutputType)) (witness : generatedRoot.Witness) : Prop :=
  let ctx : generatedRoot.Ctx backend hashModel tape path params := ⟨inputs, witness⟩
  (((((generatedRoot.constraints_0 backend hashModel tape path params (generatedRoot.w_0_0 ctx) (generatedRoot.w_1_0 ctx) (generatedRoot.w_2_0 ctx) (generatedRoot.w_3_0 ctx) (generatedRoot.w_4_0 ctx) (generatedRoot.w_5_0 ctx) (generatedRoot.w_6_0 ctx) (generatedRoot.w_7_0 ctx) (generatedRoot.w_8_0 ctx) (generatedRoot.w_9_0 ctx) (generatedRoot.w_10_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_13_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_16_0 ctx) (generatedRoot.w_19_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_25_0 ctx) (generatedRoot.w_26_0 ctx) (generatedRoot.w_29_0 ctx) (generatedRoot.w_30_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_27_0 ctx) (generatedRoot.w_27_1 ctx) (generatedRoot.w_27_2 ctx) (generatedRoot.w_27_3 ctx) (generatedRoot.w_27_4 ctx) (generatedRoot.w_27_5 ctx) (generatedRoot.w_28_0 ctx) (generatedRoot.w_28_1 ctx) (generatedRoot.w_28_2 ctx) (generatedRoot.w_28_3 ctx) (generatedRoot.w_28_4 ctx) (generatedRoot.w_28_5 ctx) (generatedRoot.w_31_0 ctx) (generatedRoot.w_31_1 ctx) (generatedRoot.w_31_2 ctx) (generatedRoot.w_31_3 ctx) (generatedRoot.w_31_4 ctx) (generatedRoot.w_31_5 ctx) (generatedRoot.w_32_0 ctx) (generatedRoot.w_32_1 ctx) (generatedRoot.w_32_2 ctx) (generatedRoot.w_32_3 ctx) (generatedRoot.w_32_4 ctx) (generatedRoot.w_32_5 ctx) (generatedRoot.w_33_0 ctx) (generatedRoot.w_33_1 ctx) (generatedRoot.w_33_2 ctx) (generatedRoot.w_33_3 ctx) (generatedRoot.w_33_4 ctx) (generatedRoot.w_33_5 ctx) (generatedRoot.w_34_0 ctx) (generatedRoot.w_34_1 ctx) (generatedRoot.w_34_2 ctx) (generatedRoot.w_34_3 ctx) (generatedRoot.w_34_4 ctx) (generatedRoot.w_34_5 ctx) (generatedRoot.w_35_0 ctx) (generatedRoot.w_35_1 ctx) (generatedRoot.w_35_2 ctx) (generatedRoot.w_35_3 ctx) (generatedRoot.w_35_4 ctx) (generatedRoot.w_35_5 ctx) (generatedRoot.w_36_0 ctx) (generatedRoot.w_37_0 ctx) (generatedRoot.w_38_0 ctx) (generatedRoot.w_39_0 ctx) (generatedRoot.w_40_0 ctx) (generatedRoot.w_41_0 ctx) (generatedRoot.w_42_0 ctx) (generatedRoot.w_43_0 ctx) (generatedRoot.w_44_0 ctx) (generatedRoot.w_45_0 ctx) (generatedRoot.w_46_0 ctx) (generatedRoot.w_47_0 ctx) (generatedRoot.w_48_0 ctx) (generatedRoot.w_49_0 ctx) (generatedRoot.w_50_0 ctx) (generatedRoot.w_51_0 ctx) (generatedRoot.w_52_0 ctx) (generatedRoot.w_53_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) ∧ (generatedRoot.constraints_1 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_15_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_56_0 ctx) (generatedRoot.w_58_0 ctx) (generatedRoot.w_59_0 ctx) (generatedRoot.w_60_0 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_63_0 ctx) (generatedRoot.w_65_0 ctx) (generatedRoot.w_66_0 ctx) (generatedRoot.w_67_0 ctx) (generatedRoot.w_70_0 ctx) (generatedRoot.w_71_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_73_0 ctx) (generatedRoot.w_75_0 ctx) (generatedRoot.w_76_0 ctx) (generatedRoot.w_77_0 ctx) (generatedRoot.w_81_0 ctx) (generatedRoot.w_83_0 ctx) (generatedRoot.w_84_0 ctx) (generatedRoot.w_85_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_102_0 ctx) (generatedRoot.w_103_0 ctx) (generatedRoot.w_104_0 ctx) (generatedRoot.w_105_0 ctx) (generatedRoot.w_105_concat_1 ctx) (generatedRoot.w_107_0 ctx) (generatedRoot.w_108_0 ctx) (generatedRoot.w_109_0 ctx) (generatedRoot.w_110_0 ctx) (generatedRoot.w_110_concat_1 ctx) (generatedRoot.w_111_0 ctx) ∧ generatedRoot.constraints_2 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_112_0 ctx) (generatedRoot.w_113_0 ctx) (generatedRoot.w_114_0 ctx) (generatedRoot.w_115_0 ctx) (generatedRoot.w_116_0 ctx) (generatedRoot.w_117_0 ctx) (generatedRoot.w_118_0 ctx) (generatedRoot.w_121_0 ctx) (generatedRoot.w_123_0 ctx) (generatedRoot.w_124_0 ctx) (generatedRoot.w_125_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_126_concat_1 ctx) (generatedRoot.w_127_0 ctx) (generatedRoot.w_128_0 ctx) (generatedRoot.w_129_0 ctx))) ∧ (generatedRoot.constraints_3 backend hashModel (generatedRoot.w_24_0 ctx) (generatedRoot.w_142_0 ctx) (generatedRoot.w_126_0 ctx) (generatedRoot.w_129_0 ctx) (generatedRoot.w_130_0 ctx) (generatedRoot.w_131_0 ctx) (generatedRoot.w_132_0 ctx) (generatedRoot.w_133_0 ctx) (generatedRoot.w_136_0 ctx) (generatedRoot.w_137_0 ctx) (generatedRoot.w_138_0 ctx) (generatedRoot.w_139_0 ctx) (generatedRoot.w_139_concat_1 ctx) (generatedRoot.w_140_0 ctx) (generatedRoot.w_141_0 ctx) (generatedRoot.w_143_0 ctx) (generatedRoot.w_144_0 ctx) (generatedRoot.w_145_0 ctx) (generatedRoot.w_146_0 ctx) ∧ (generatedRoot.constraints_4 backend hashModel (generatedRoot.w_24_0 ctx) (generatedRoot.w_152_0 ctx) (generatedRoot.w_147_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_151_0 ctx) (generatedRoot.w_153_0 ctx) (generatedRoot.w_154_0 ctx) (generatedRoot.w_155_0 ctx) (generatedRoot.w_156_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) ∧ generatedRoot.constraints_5 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_91_0 ctx) (generatedRoot.w_93_0 ctx) (generatedRoot.w_95_0 ctx) (generatedRoot.w_97_0 ctx) (generatedRoot.w_99_0 ctx) (generatedRoot.w_100_0 ctx) (generatedRoot.w_90_0 ctx) (generatedRoot.w_92_0 ctx) (generatedRoot.w_94_0 ctx) (generatedRoot.w_96_0 ctx) (generatedRoot.w_98_0 ctx) (generatedRoot.w_148_0 ctx) (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_157_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_0 ctx) (generatedRoot.w_161_1 ctx) (generatedRoot.w_161_2 ctx) (generatedRoot.w_161_3 ctx) (generatedRoot.w_161_4 ctx) (generatedRoot.w_161_5 ctx) (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_7 ctx) (generatedRoot.w_161_8 ctx) (generatedRoot.w_161_9 ctx) (generatedRoot.w_162_0 ctx) (generatedRoot.w_163_0 ctx) (generatedRoot.w_164_0 ctx) (generatedRoot.w_165_0 ctx) (generatedRoot.w_166_0 ctx) (generatedRoot.w_166_concat_1 ctx) (generatedRoot.w_167_0 ctx) (generatedRoot.w_168_0 ctx) (generatedRoot.w_169_0 ctx) (generatedRoot.w_170_0 ctx) (generatedRoot.w_170_concat_1 ctx) (generatedRoot.w_171_0 ctx) (generatedRoot.w_172_0 ctx)))) ∧ ((generatedRoot.constraints_6 backend (generatedRoot.w_149_0 ctx) (generatedRoot.w_150_0 ctx) (generatedRoot.w_158_0 ctx) (generatedRoot.w_159_0 ctx) (generatedRoot.w_160_0 ctx) (generatedRoot.w_161_6 ctx) (generatedRoot.w_161_8 ctx) (generatedRoot.w_172_0 ctx) (generatedRoot.w_173_0 ctx) (generatedRoot.w_174_0 ctx) (generatedRoot.w_175_0 ctx) (generatedRoot.w_175_concat_1 ctx) (generatedRoot.w_176_0 ctx) (generatedRoot.w_177_0 ctx) (generatedRoot.w_178_0 ctx) (generatedRoot.w_179_0 ctx) (generatedRoot.w_179_concat_1 ctx) (generatedRoot.w_180_0 ctx) (generatedRoot.w_181_0 ctx) (generatedRoot.w_182_0 ctx) (generatedRoot.w_183_0 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) ∧ (generatedRoot.constraints_7 backend (generatedRoot.w_72_0 ctx) (generatedRoot.w_161_7 ctx) (generatedRoot.w_161_9 ctx) (generatedRoot.w_184_0 ctx) (generatedRoot.w_185_0 ctx) (generatedRoot.w_186_0 ctx) (generatedRoot.w_187_0 ctx) (generatedRoot.w_188_0 ctx) (generatedRoot.w_189_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_191_0 ctx) (generatedRoot.w_192_0 ctx) (generatedRoot.w_195_0 ctx) (generatedRoot.w_197_0 ctx) (generatedRoot.w_198_0 ctx) (generatedRoot.w_199_0 ctx) (generatedRoot.w_201_0 ctx) (generatedRoot.w_202_0 ctx) (generatedRoot.w_203_0 ctx) ∧ generatedRoot.constraints_8 backend (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_204_0 ctx) (generatedRoot.w_205_0 ctx) (generatedRoot.w_206_0 ctx) (generatedRoot.w_207_0 ctx) (generatedRoot.w_208_0 ctx) (generatedRoot.w_209_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_211_0 ctx) (generatedRoot.w_212_0 ctx) (generatedRoot.w_216_0 ctx) (generatedRoot.w_218_0 ctx) (generatedRoot.w_219_0 ctx) (generatedRoot.w_220_0 ctx) (generatedRoot.w_222_0 ctx) (generatedRoot.w_224_0 ctx) (generatedRoot.w_225_0 ctx))) ∧ (generatedRoot.constraints_9 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_230_0 ctx) (generatedRoot.w_232_0 ctx) (generatedRoot.w_234_0 ctx) (generatedRoot.w_236_0 ctx) (generatedRoot.w_238_0 ctx) (generatedRoot.w_239_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_225_0 ctx) (generatedRoot.w_226_0 ctx) (generatedRoot.w_229_0 ctx) (generatedRoot.w_231_0 ctx) (generatedRoot.w_233_0 ctx) (generatedRoot.w_235_0 ctx) (generatedRoot.w_237_0 ctx) (generatedRoot.w_240_0 ctx) (generatedRoot.w_240_1 ctx) (generatedRoot.w_240_2 ctx) (generatedRoot.w_240_3 ctx) (generatedRoot.w_240_4 ctx) (generatedRoot.w_240_5 ctx) (generatedRoot.w_240_6 ctx) (generatedRoot.w_240_7 ctx) (generatedRoot.w_240_8 ctx) (generatedRoot.w_240_9 ctx) (generatedRoot.w_241_0 ctx) (generatedRoot.w_242_0 ctx) (generatedRoot.w_243_0 ctx) (generatedRoot.w_244_0 ctx) (generatedRoot.w_245_0 ctx) (generatedRoot.w_245_concat_1 ctx) (generatedRoot.w_246_0 ctx) (generatedRoot.w_247_0 ctx) (generatedRoot.w_248_0 ctx) (generatedRoot.w_249_0 ctx) (generatedRoot.w_249_concat_1 ctx) (generatedRoot.w_250_0 ctx) (generatedRoot.w_251_0 ctx) (generatedRoot.w_252_0 ctx) (generatedRoot.w_253_0 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_254_concat_1 ctx) ∧ (generatedRoot.constraints_10 backend (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_240_7 ctx) (generatedRoot.w_240_9 ctx) (generatedRoot.w_254_0 ctx) (generatedRoot.w_255_0 ctx) (generatedRoot.w_256_0 ctx) (generatedRoot.w_257_0 ctx) (generatedRoot.w_258_0 ctx) (generatedRoot.w_258_concat_1 ctx) (generatedRoot.w_259_0 ctx) (generatedRoot.w_260_0 ctx) (generatedRoot.w_261_0 ctx) (generatedRoot.w_262_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_264_0 ctx) (generatedRoot.w_265_0 ctx) (generatedRoot.w_266_0 ctx) (generatedRoot.w_267_0 ctx) (generatedRoot.w_268_0 ctx) ∧ generatedRoot.constraints_11 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_288_0 ctx) (generatedRoot.w_290_0 ctx) (generatedRoot.w_292_0 ctx) (generatedRoot.w_294_0 ctx) (generatedRoot.w_296_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_263_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_270_0 ctx) (generatedRoot.w_271_0 ctx) (generatedRoot.w_274_0 ctx) (generatedRoot.w_276_0 ctx) (generatedRoot.w_277_0 ctx) (generatedRoot.w_278_0 ctx) (generatedRoot.w_280_0 ctx) (generatedRoot.w_282_0 ctx) (generatedRoot.w_283_0 ctx) (generatedRoot.w_284_0 ctx) (generatedRoot.w_287_0 ctx) (generatedRoot.w_289_0 ctx) (generatedRoot.w_291_0 ctx) (generatedRoot.w_293_0 ctx) (generatedRoot.w_295_0 ctx) (generatedRoot.w_298_0 ctx) (generatedRoot.w_298_1 ctx) (generatedRoot.w_298_2 ctx) (generatedRoot.w_298_3 ctx) (generatedRoot.w_298_4 ctx) (generatedRoot.w_298_5 ctx) (generatedRoot.w_298_6 ctx) (generatedRoot.w_298_7 ctx) (generatedRoot.w_298_8 ctx) (generatedRoot.w_298_9 ctx) (generatedRoot.w_299_0 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_301_0 ctx) (generatedRoot.w_302_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_303_concat_1 ctx))))) ∧ (((generatedRoot.constraints_12 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_297_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_298_7 ctx) (generatedRoot.w_298_9 ctx) (generatedRoot.w_300_0 ctx) (generatedRoot.w_303_0 ctx) (generatedRoot.w_304_0 ctx) (generatedRoot.w_305_0 ctx) (generatedRoot.w_306_0 ctx) (generatedRoot.w_307_0 ctx) (generatedRoot.w_307_concat_1 ctx) (generatedRoot.w_308_0 ctx) (generatedRoot.w_309_0 ctx) (generatedRoot.w_310_0 ctx) (generatedRoot.w_311_0 ctx) (generatedRoot.w_312_0 ctx) (generatedRoot.w_312_concat_1 ctx) (generatedRoot.w_313_0 ctx) (generatedRoot.w_314_0 ctx) (generatedRoot.w_315_0 ctx) (generatedRoot.w_316_0 ctx) (generatedRoot.w_316_concat_1 ctx) (generatedRoot.w_317_0 ctx) (generatedRoot.w_318_0 ctx) (generatedRoot.w_319_0 ctx) (generatedRoot.w_320_0 ctx) ∧ (generatedRoot.constraints_13 backend (generatedRoot.w_55_0 ctx) (generatedRoot.w_321_0 ctx) (generatedRoot.w_322_0 ctx) (generatedRoot.w_323_0 ctx) (generatedRoot.w_324_0 ctx) (generatedRoot.w_325_0 ctx) (generatedRoot.w_326_0 ctx) (generatedRoot.w_327_0 ctx) (generatedRoot.w_328_0 ctx) (generatedRoot.w_329_0 ctx) (generatedRoot.w_332_0 ctx) ∧ generatedRoot.constraints_14 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_346_0 ctx) (generatedRoot.w_348_0 ctx) (generatedRoot.w_350_0 ctx) (generatedRoot.w_352_0 ctx) (generatedRoot.w_354_0 ctx) (generatedRoot.w_355_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_334_0 ctx) (generatedRoot.w_335_0 ctx) (generatedRoot.w_336_0 ctx) (generatedRoot.w_338_0 ctx) (generatedRoot.w_340_0 ctx) (generatedRoot.w_341_0 ctx) (generatedRoot.w_342_0 ctx) (generatedRoot.w_345_0 ctx) (generatedRoot.w_347_0 ctx) (generatedRoot.w_349_0 ctx) (generatedRoot.w_351_0 ctx) (generatedRoot.w_353_0 ctx) (generatedRoot.w_356_0 ctx) (generatedRoot.w_356_1 ctx) (generatedRoot.w_356_2 ctx) (generatedRoot.w_356_3 ctx) (generatedRoot.w_356_4 ctx) (generatedRoot.w_356_5 ctx) (generatedRoot.w_356_6 ctx) (generatedRoot.w_356_7 ctx) (generatedRoot.w_356_8 ctx) (generatedRoot.w_356_9 ctx) (generatedRoot.w_357_0 ctx) (generatedRoot.w_358_0 ctx) (generatedRoot.w_359_0 ctx) (generatedRoot.w_360_0 ctx) (generatedRoot.w_361_0 ctx) (generatedRoot.w_361_concat_1 ctx) (generatedRoot.w_362_0 ctx) (generatedRoot.w_363_0 ctx) (generatedRoot.w_364_0 ctx) (generatedRoot.w_365_0 ctx) (generatedRoot.w_365_concat_1 ctx) (generatedRoot.w_366_0 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx))) ∧ (generatedRoot.constraints_15 backend (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_356_7 ctx) (generatedRoot.w_356_9 ctx) (generatedRoot.w_367_0 ctx) (generatedRoot.w_368_0 ctx) (generatedRoot.w_369_0 ctx) (generatedRoot.w_370_0 ctx) (generatedRoot.w_370_concat_1 ctx) (generatedRoot.w_371_0 ctx) (generatedRoot.w_372_0 ctx) (generatedRoot.w_373_0 ctx) (generatedRoot.w_374_0 ctx) (generatedRoot.w_374_concat_1 ctx) (generatedRoot.w_375_0 ctx) (generatedRoot.w_376_0 ctx) (generatedRoot.w_377_0 ctx) (generatedRoot.w_378_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_381_0 ctx) ∧ (generatedRoot.constraints_16 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_404_0 ctx) (generatedRoot.w_406_0 ctx) (generatedRoot.w_408_0 ctx) (generatedRoot.w_410_0 ctx) (generatedRoot.w_412_0 ctx) (generatedRoot.w_413_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_379_0 ctx) (generatedRoot.w_380_0 ctx) (generatedRoot.w_381_0 ctx) (generatedRoot.w_382_0 ctx) (generatedRoot.w_383_0 ctx) (generatedRoot.w_384_0 ctx) (generatedRoot.w_385_0 ctx) (generatedRoot.w_386_0 ctx) (generatedRoot.w_387_0 ctx) (generatedRoot.w_390_0 ctx) (generatedRoot.w_392_0 ctx) (generatedRoot.w_393_0 ctx) (generatedRoot.w_394_0 ctx) (generatedRoot.w_396_0 ctx) (generatedRoot.w_398_0 ctx) (generatedRoot.w_399_0 ctx) (generatedRoot.w_400_0 ctx) (generatedRoot.w_403_0 ctx) (generatedRoot.w_405_0 ctx) (generatedRoot.w_407_0 ctx) (generatedRoot.w_409_0 ctx) (generatedRoot.w_411_0 ctx) (generatedRoot.w_414_0 ctx) (generatedRoot.w_414_1 ctx) (generatedRoot.w_414_2 ctx) (generatedRoot.w_414_3 ctx) (generatedRoot.w_414_4 ctx) (generatedRoot.w_414_5 ctx) (generatedRoot.w_414_6 ctx) (generatedRoot.w_414_7 ctx) (generatedRoot.w_414_8 ctx) (generatedRoot.w_414_9 ctx) (generatedRoot.w_415_0 ctx) ∧ generatedRoot.constraints_17 hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_413_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_415_0 ctx) (generatedRoot.w_416_0 ctx) (generatedRoot.w_417_0 ctx) (generatedRoot.w_418_0 ctx) (generatedRoot.w_419_0 ctx) (generatedRoot.w_419_concat_1 ctx) (generatedRoot.w_420_0 ctx) (generatedRoot.w_421_0 ctx) (generatedRoot.w_422_0 ctx) (generatedRoot.w_423_0 ctx) (generatedRoot.w_423_concat_1 ctx) (generatedRoot.w_424_0 ctx) (generatedRoot.w_425_0 ctx) (generatedRoot.w_426_0 ctx) (generatedRoot.w_427_0 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_428_concat_1 ctx) (generatedRoot.w_429_0 ctx) (generatedRoot.w_430_0 ctx) (generatedRoot.w_431_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_432_concat_1 ctx)))) ∧ ((generatedRoot.constraints_18 backend (generatedRoot.w_414_7 ctx) (generatedRoot.w_414_9 ctx) (generatedRoot.w_428_0 ctx) (generatedRoot.w_432_0 ctx) (generatedRoot.w_433_0 ctx) (generatedRoot.w_434_0 ctx) (generatedRoot.w_435_0 ctx) (generatedRoot.w_436_0 ctx) (generatedRoot.w_437_0 ctx) (generatedRoot.w_438_0 ctx) (generatedRoot.w_439_0 ctx) (generatedRoot.w_440_0 ctx) (generatedRoot.w_441_0 ctx) (generatedRoot.w_442_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_444_0 ctx) ∧ (generatedRoot.constraints_19 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_462_0 ctx) (generatedRoot.w_464_0 ctx) (generatedRoot.w_466_0 ctx) (generatedRoot.w_468_0 ctx) (generatedRoot.w_470_0 ctx) (generatedRoot.w_471_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_437_0 ctx) (generatedRoot.w_445_0 ctx) (generatedRoot.w_448_0 ctx) (generatedRoot.w_450_0 ctx) (generatedRoot.w_451_0 ctx) (generatedRoot.w_452_0 ctx) (generatedRoot.w_454_0 ctx) (generatedRoot.w_456_0 ctx) (generatedRoot.w_457_0 ctx) (generatedRoot.w_458_0 ctx) (generatedRoot.w_461_0 ctx) (generatedRoot.w_463_0 ctx) (generatedRoot.w_465_0 ctx) (generatedRoot.w_467_0 ctx) (generatedRoot.w_469_0 ctx) (generatedRoot.w_472_0 ctx) (generatedRoot.w_472_1 ctx) (generatedRoot.w_472_2 ctx) (generatedRoot.w_472_3 ctx) (generatedRoot.w_472_4 ctx) (generatedRoot.w_472_5 ctx) (generatedRoot.w_472_6 ctx) (generatedRoot.w_472_7 ctx) (generatedRoot.w_472_8 ctx) (generatedRoot.w_472_9 ctx) (generatedRoot.w_473_0 ctx) (generatedRoot.w_474_0 ctx) (generatedRoot.w_475_0 ctx) (generatedRoot.w_476_0 ctx) (generatedRoot.w_477_0 ctx) (generatedRoot.w_477_concat_1 ctx) (generatedRoot.w_478_0 ctx) (generatedRoot.w_479_0 ctx) (generatedRoot.w_480_0 ctx) (generatedRoot.w_481_0 ctx) (generatedRoot.w_481_concat_1 ctx) (generatedRoot.w_482_0 ctx) (generatedRoot.w_483_0 ctx) ∧ generatedRoot.constraints_20 backend (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_472_7 ctx) (generatedRoot.w_472_9 ctx) (generatedRoot.w_483_0 ctx) (generatedRoot.w_484_0 ctx) (generatedRoot.w_485_0 ctx) (generatedRoot.w_486_0 ctx) (generatedRoot.w_486_concat_1 ctx) (generatedRoot.w_487_0 ctx) (generatedRoot.w_488_0 ctx) (generatedRoot.w_489_0 ctx) (generatedRoot.w_490_0 ctx) (generatedRoot.w_490_concat_1 ctx) (generatedRoot.w_491_0 ctx) (generatedRoot.w_492_0 ctx) (generatedRoot.w_493_0 ctx) (generatedRoot.w_494_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx))) ∧ ((generatedRoot.constraints_21 backend (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_495_0 ctx) (generatedRoot.w_496_0 ctx) (generatedRoot.w_497_0 ctx) (generatedRoot.w_498_0 ctx) (generatedRoot.w_499_0 ctx) (generatedRoot.w_500_0 ctx) (generatedRoot.w_501_0 ctx) (generatedRoot.w_502_0 ctx) (generatedRoot.w_503_0 ctx) (generatedRoot.w_506_0 ctx) (generatedRoot.w_508_0 ctx) (generatedRoot.w_509_0 ctx) (generatedRoot.w_510_0 ctx) (generatedRoot.w_512_0 ctx) (generatedRoot.w_514_0 ctx) (generatedRoot.w_515_0 ctx) (generatedRoot.w_516_0 ctx) (generatedRoot.w_521_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_527_0 ctx) ∧ generatedRoot.constraints_22 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_520_0 ctx) (generatedRoot.w_522_0 ctx) (generatedRoot.w_524_0 ctx) (generatedRoot.w_526_0 ctx) (generatedRoot.w_528_0 ctx) (generatedRoot.w_529_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_519_0 ctx) (generatedRoot.w_521_0 ctx) (generatedRoot.w_523_0 ctx) (generatedRoot.w_525_0 ctx) (generatedRoot.w_527_0 ctx) (generatedRoot.w_530_0 ctx) (generatedRoot.w_530_1 ctx) (generatedRoot.w_530_2 ctx) (generatedRoot.w_530_3 ctx) (generatedRoot.w_530_4 ctx) (generatedRoot.w_530_5 ctx) (generatedRoot.w_530_6 ctx) (generatedRoot.w_530_7 ctx) (generatedRoot.w_530_8 ctx) (generatedRoot.w_530_9 ctx) (generatedRoot.w_531_0 ctx) (generatedRoot.w_532_0 ctx) (generatedRoot.w_533_0 ctx) (generatedRoot.w_534_0 ctx) (generatedRoot.w_535_0 ctx) (generatedRoot.w_535_concat_1 ctx) (generatedRoot.w_536_0 ctx) (generatedRoot.w_537_0 ctx) (generatedRoot.w_538_0 ctx) (generatedRoot.w_539_0 ctx) (generatedRoot.w_539_concat_1 ctx) (generatedRoot.w_540_0 ctx) (generatedRoot.w_541_0 ctx) (generatedRoot.w_542_0 ctx) (generatedRoot.w_543_0 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_544_concat_1 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_546_0 ctx)) ∧ (generatedRoot.constraints_23 backend (generatedRoot.w_54_3 ctx) (generatedRoot.w_530_7 ctx) (generatedRoot.w_530_9 ctx) (generatedRoot.w_544_0 ctx) (generatedRoot.w_545_0 ctx) (generatedRoot.w_546_0 ctx) (generatedRoot.w_547_0 ctx) (generatedRoot.w_548_0 ctx) (generatedRoot.w_548_concat_1 ctx) (generatedRoot.w_549_0 ctx) (generatedRoot.w_550_0 ctx) (generatedRoot.w_551_0 ctx) (generatedRoot.w_552_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_554_0 ctx) (generatedRoot.w_555_0 ctx) (generatedRoot.w_556_0 ctx) (generatedRoot.w_557_0 ctx) (generatedRoot.w_558_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_560_0 ctx) ∧ generatedRoot.constraints_24 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_578_0 ctx) (generatedRoot.w_580_0 ctx) (generatedRoot.w_582_0 ctx) (generatedRoot.w_584_0 ctx) (generatedRoot.w_586_0 ctx) (generatedRoot.w_587_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_553_0 ctx) (generatedRoot.w_561_0 ctx) (generatedRoot.w_564_0 ctx) (generatedRoot.w_566_0 ctx) (generatedRoot.w_567_0 ctx) (generatedRoot.w_568_0 ctx) (generatedRoot.w_570_0 ctx) (generatedRoot.w_572_0 ctx) (generatedRoot.w_573_0 ctx) (generatedRoot.w_574_0 ctx) (generatedRoot.w_577_0 ctx) (generatedRoot.w_579_0 ctx) (generatedRoot.w_581_0 ctx) (generatedRoot.w_583_0 ctx) (generatedRoot.w_585_0 ctx) (generatedRoot.w_588_0 ctx) (generatedRoot.w_588_1 ctx) (generatedRoot.w_588_2 ctx) (generatedRoot.w_588_3 ctx) (generatedRoot.w_588_4 ctx) (generatedRoot.w_588_5 ctx) (generatedRoot.w_588_6 ctx) (generatedRoot.w_588_7 ctx) (generatedRoot.w_588_8 ctx) (generatedRoot.w_588_9 ctx) (generatedRoot.w_589_0 ctx) (generatedRoot.w_590_0 ctx) (generatedRoot.w_591_0 ctx) (generatedRoot.w_592_0 ctx) (generatedRoot.w_593_0 ctx) (generatedRoot.w_593_concat_1 ctx) (generatedRoot.w_594_0 ctx) (generatedRoot.w_595_0 ctx) (generatedRoot.w_596_0 ctx) (generatedRoot.w_597_0 ctx) (generatedRoot.w_597_concat_1 ctx) (generatedRoot.w_598_0 ctx)))))) ∧ ((((generatedRoot.constraints_25 backend (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_588_7 ctx) (generatedRoot.w_588_9 ctx) (generatedRoot.w_599_0 ctx) (generatedRoot.w_600_0 ctx) (generatedRoot.w_601_0 ctx) (generatedRoot.w_602_0 ctx) (generatedRoot.w_602_concat_1 ctx) (generatedRoot.w_603_0 ctx) (generatedRoot.w_604_0 ctx) (generatedRoot.w_605_0 ctx) (generatedRoot.w_606_0 ctx) (generatedRoot.w_606_concat_1 ctx) (generatedRoot.w_607_0 ctx) (generatedRoot.w_608_0 ctx) (generatedRoot.w_609_0 ctx) (generatedRoot.w_610_0 ctx) ∧ (generatedRoot.constraints_26 backend (generatedRoot.w_55_0 ctx) (generatedRoot.w_72_0 ctx) (generatedRoot.w_78_0 ctx) (generatedRoot.w_79_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_200_0 ctx) (generatedRoot.w_215_0 ctx) (generatedRoot.w_227_0 ctx) (generatedRoot.w_228_0 ctx) (generatedRoot.w_285_0 ctx) (generatedRoot.w_286_0 ctx) (generatedRoot.w_343_0 ctx) (generatedRoot.w_344_0 ctx) (generatedRoot.w_401_0 ctx) (generatedRoot.w_402_0 ctx) (generatedRoot.w_459_0 ctx) (generatedRoot.w_460_0 ctx) (generatedRoot.w_517_0 ctx) (generatedRoot.w_518_0 ctx) (generatedRoot.w_575_0 ctx) (generatedRoot.w_576_0 ctx) (generatedRoot.w_611_0 ctx) (generatedRoot.w_612_0 ctx) (generatedRoot.w_613_0 ctx) (generatedRoot.w_614_0 ctx) (generatedRoot.w_615_0 ctx) (generatedRoot.w_616_0 ctx) (generatedRoot.w_617_0 ctx) (generatedRoot.w_618_0 ctx) (generatedRoot.w_619_0 ctx) (generatedRoot.w_622_0 ctx) (generatedRoot.w_624_0 ctx) (generatedRoot.w_625_0 ctx) (generatedRoot.w_626_0 ctx) (generatedRoot.w_628_0 ctx) (generatedRoot.w_630_0 ctx) (generatedRoot.w_631_0 ctx) (generatedRoot.w_632_0 ctx) (generatedRoot.w_633_0 ctx) (generatedRoot.w_634_0 ctx) (generatedRoot.w_635_0 ctx) (generatedRoot.w_636_0 ctx) ∧ generatedRoot.constraints_27 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_638_0 ctx) (generatedRoot.w_640_0 ctx) (generatedRoot.w_641_0 ctx) (generatedRoot.w_643_0 ctx) (generatedRoot.w_649_0 ctx) (generatedRoot.w_86_0 ctx) (generatedRoot.w_89_0 ctx) (generatedRoot.w_190_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_637_0 ctx) (generatedRoot.w_637_1 ctx) (generatedRoot.w_639_0 ctx) (generatedRoot.w_642_0 ctx) (generatedRoot.w_644_0 ctx) (generatedRoot.w_645_0 ctx) (generatedRoot.w_646_0 ctx) (generatedRoot.w_647_0 ctx) (generatedRoot.w_648_0 ctx) (generatedRoot.w_650_0 ctx) (generatedRoot.w_651_0 ctx) (generatedRoot.w_653_0 ctx) (generatedRoot.w_654_0 ctx) (generatedRoot.w_655_0 ctx) (generatedRoot.w_657_0 ctx) (generatedRoot.w_658_0 ctx) (generatedRoot.w_658_1 ctx) (generatedRoot.w_658_2 ctx) (generatedRoot.w_658_3 ctx) (generatedRoot.w_658_4 ctx) (generatedRoot.w_658_5 ctx) (generatedRoot.w_661_0 ctx) (generatedRoot.w_664_0 ctx) (generatedRoot.w_670_0 ctx))) ∧ (generatedRoot.constraints_28 (generatedRoot.w_327_0 ctx) (generatedRoot.w_385_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_501_0 ctx) (generatedRoot.w_675_0 ctx) (generatedRoot.w_680_0 ctx) (generatedRoot.w_685_0 ctx) (generatedRoot.w_690_0 ctx) ∧ (generatedRoot.constraints_29 (generatedRoot.w_190_0 ctx) (generatedRoot.w_210_0 ctx) (generatedRoot.w_269_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_617_0 ctx) (generatedRoot.w_635_0 ctx) (generatedRoot.w_651_0 ctx) (generatedRoot.w_654_0 ctx) (generatedRoot.w_659_0 ctx) (generatedRoot.w_660_0 ctx) (generatedRoot.w_663_0 ctx) (generatedRoot.w_666_0 ctx) (generatedRoot.w_668_0 ctx) (generatedRoot.w_669_0 ctx) (generatedRoot.w_673_0 ctx) (generatedRoot.w_674_0 ctx) (generatedRoot.w_678_0 ctx) (generatedRoot.w_679_0 ctx) (generatedRoot.w_683_0 ctx) (generatedRoot.w_684_0 ctx) (generatedRoot.w_688_0 ctx) (generatedRoot.w_689_0 ctx) (generatedRoot.w_693_0 ctx) (generatedRoot.w_694_0 ctx) (generatedRoot.w_695_0 ctx) (generatedRoot.w_698_0 ctx) (generatedRoot.w_699_0 ctx) (generatedRoot.w_700_0 ctx) (generatedRoot.w_703_0 ctx) (generatedRoot.w_704_0 ctx) (generatedRoot.w_705_0 ctx) (generatedRoot.w_708_0 ctx) (generatedRoot.w_711_0 ctx) (generatedRoot.w_717_0 ctx) ∧ generatedRoot.constraints_30 (generatedRoot.w_327_0 ctx) (generatedRoot.w_385_0 ctx) (generatedRoot.w_443_0 ctx) (generatedRoot.w_501_0 ctx) (generatedRoot.w_722_0 ctx) (generatedRoot.w_727_0 ctx) (generatedRoot.w_732_0 ctx) (generatedRoot.w_737_0 ctx)))) ∧ ((generatedRoot.constraints_31 hashModel (generatedRoot.w_753_0 ctx) (generatedRoot.w_559_0 ctx) (generatedRoot.w_617_0 ctx) (generatedRoot.w_635_0 ctx) (generatedRoot.w_653_0 ctx) (generatedRoot.w_657_0 ctx) (generatedRoot.w_706_0 ctx) (generatedRoot.w_707_0 ctx) (generatedRoot.w_710_0 ctx) (generatedRoot.w_713_0 ctx) (generatedRoot.w_715_0 ctx) (generatedRoot.w_716_0 ctx) (generatedRoot.w_720_0 ctx) (generatedRoot.w_721_0 ctx) (generatedRoot.w_725_0 ctx) (generatedRoot.w_726_0 ctx) (generatedRoot.w_730_0 ctx) (generatedRoot.w_731_0 ctx) (generatedRoot.w_735_0 ctx) (generatedRoot.w_736_0 ctx) (generatedRoot.w_740_0 ctx) (generatedRoot.w_741_0 ctx) (generatedRoot.w_742_0 ctx) (generatedRoot.w_745_0 ctx) (generatedRoot.w_746_0 ctx) (generatedRoot.w_747_0 ctx) (generatedRoot.w_750_0 ctx) (generatedRoot.w_751_0 ctx) (generatedRoot.w_752_0 ctx) (generatedRoot.w_754_0 ctx) (generatedRoot.w_755_0 ctx) (generatedRoot.w_756_0 ctx) (generatedRoot.w_757_0 ctx) (generatedRoot.w_758_0 ctx) (generatedRoot.w_759_0 ctx) (generatedRoot.w_760_0 ctx) (generatedRoot.w_761_0 ctx) (generatedRoot.w_762_0 ctx) (generatedRoot.w_763_0 ctx) (generatedRoot.w_764_0 ctx) (generatedRoot.w_765_0 ctx) (generatedRoot.w_766_0 ctx) (generatedRoot.w_767_0 ctx) (generatedRoot.w_768_0 ctx) (generatedRoot.w_769_0 ctx) (generatedRoot.w_770_0 ctx) (generatedRoot.w_771_0 ctx) (generatedRoot.w_772_0 ctx) (generatedRoot.w_773_0 ctx) (generatedRoot.w_774_0 ctx) ∧ (generatedRoot.constraints_32 hashModel (generatedRoot.w_753_0 ctx) (generatedRoot.w_774_0 ctx) (generatedRoot.w_775_0 ctx) (generatedRoot.w_776_0 ctx) (generatedRoot.w_777_0 ctx) (generatedRoot.w_778_0 ctx) (generatedRoot.w_779_0 ctx) (generatedRoot.w_780_0 ctx) (generatedRoot.w_781_0 ctx) (generatedRoot.w_782_0 ctx) (generatedRoot.w_783_0 ctx) (generatedRoot.w_784_0 ctx) (generatedRoot.w_785_0 ctx) (generatedRoot.w_786_0 ctx) (generatedRoot.w_787_0 ctx) (generatedRoot.w_788_0 ctx) (generatedRoot.w_789_0 ctx) (generatedRoot.w_790_0 ctx) (generatedRoot.w_791_0 ctx) (generatedRoot.w_792_0 ctx) (generatedRoot.w_793_0 ctx) (generatedRoot.w_794_0 ctx) (generatedRoot.w_795_0 ctx) (generatedRoot.w_796_0 ctx) (generatedRoot.w_797_0 ctx) (generatedRoot.w_798_0 ctx) (generatedRoot.w_799_0 ctx) (generatedRoot.w_800_0 ctx) (generatedRoot.w_801_0 ctx) (generatedRoot.w_802_0 ctx) (generatedRoot.w_803_0 ctx) (generatedRoot.w_804_0 ctx) (generatedRoot.w_805_0 ctx) (generatedRoot.w_806_0 ctx) (generatedRoot.w_807_0 ctx) (generatedRoot.w_808_0 ctx) (generatedRoot.w_809_0 ctx) (generatedRoot.w_810_0 ctx) (generatedRoot.w_811_0 ctx) (generatedRoot.w_812_0 ctx) (generatedRoot.w_813_0 ctx) (generatedRoot.w_814_0 ctx) ∧ generatedRoot.constraints_33 hashModel (generatedRoot.w_753_0 ctx) (generatedRoot.w_813_0 ctx) (generatedRoot.w_814_0 ctx) (generatedRoot.w_815_0 ctx) (generatedRoot.w_816_0 ctx) (generatedRoot.w_817_0 ctx) (generatedRoot.w_818_0 ctx) (generatedRoot.w_819_0 ctx) (generatedRoot.w_820_0 ctx) (generatedRoot.w_821_0 ctx) (generatedRoot.w_822_0 ctx) (generatedRoot.w_823_0 ctx) (generatedRoot.w_824_0 ctx) (generatedRoot.w_825_0 ctx) (generatedRoot.w_826_0 ctx) (generatedRoot.w_827_0 ctx) (generatedRoot.w_828_0 ctx) (generatedRoot.w_829_0 ctx) (generatedRoot.w_830_0 ctx) (generatedRoot.w_831_0 ctx) (generatedRoot.w_832_0 ctx) (generatedRoot.w_833_0 ctx) (generatedRoot.w_834_0 ctx) (generatedRoot.w_835_0 ctx) (generatedRoot.w_836_0 ctx) (generatedRoot.w_837_0 ctx) (generatedRoot.w_838_0 ctx) (generatedRoot.w_839_0 ctx) (generatedRoot.w_840_0 ctx) (generatedRoot.w_841_0 ctx) (generatedRoot.w_842_0 ctx) (generatedRoot.w_843_0 ctx) (generatedRoot.w_844_0 ctx) (generatedRoot.w_845_0 ctx) (generatedRoot.w_846_0 ctx) (generatedRoot.w_847_0 ctx) (generatedRoot.w_848_0 ctx) (generatedRoot.w_849_0 ctx) (generatedRoot.w_850_0 ctx) (generatedRoot.w_851_0 ctx) (generatedRoot.w_852_0 ctx) (generatedRoot.w_853_0 ctx) (generatedRoot.w_854_0 ctx))) ∧ (generatedRoot.constraints_34 hashModel (generatedRoot.w_753_0 ctx) (generatedRoot.w_858_0 ctx) (generatedRoot.w_853_0 ctx) (generatedRoot.w_854_0 ctx) (generatedRoot.w_855_0 ctx) (generatedRoot.w_856_0 ctx) (generatedRoot.w_857_0 ctx) (generatedRoot.w_859_0 ctx) (generatedRoot.w_860_0 ctx) (generatedRoot.w_861_0 ctx) (generatedRoot.w_862_0 ctx) (generatedRoot.w_863_0 ctx) (generatedRoot.w_864_0 ctx) (generatedRoot.w_865_0 ctx) (generatedRoot.w_866_0 ctx) ∧ (generatedRoot.constraints_35 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_753_0 ctx) (generatedRoot.w_872_0 ctx) (generatedRoot.w_873_0 ctx) (generatedRoot.w_874_0 ctx) (generatedRoot.w_875_0 ctx) (generatedRoot.w_876_0 ctx) (generatedRoot.w_877_0 ctx) (generatedRoot.w_878_0 ctx) (generatedRoot.w_879_0 ctx) (generatedRoot.w_882_0 ctx) (generatedRoot.w_883_0 ctx) (generatedRoot.w_886_0 ctx) (generatedRoot.w_887_0 ctx) (generatedRoot.w_888_0 ctx) (generatedRoot.w_891_0 ctx) (generatedRoot.w_892_0 ctx) (generatedRoot.w_636_0 ctx) (generatedRoot.w_658_0 ctx) (generatedRoot.w_658_1 ctx) (generatedRoot.w_658_2 ctx) (generatedRoot.w_658_3 ctx) (generatedRoot.w_658_4 ctx) (generatedRoot.w_658_5 ctx) (generatedRoot.w_705_0 ctx) (generatedRoot.w_752_0 ctx) (generatedRoot.w_757_0 ctx) (generatedRoot.w_761_0 ctx) (generatedRoot.w_765_0 ctx) (generatedRoot.w_769_0 ctx) (generatedRoot.w_773_0 ctx) (generatedRoot.w_777_0 ctx) (generatedRoot.w_782_0 ctx) (generatedRoot.w_787_0 ctx) (generatedRoot.w_792_0 ctx) (generatedRoot.w_797_0 ctx) (generatedRoot.w_802_0 ctx) (generatedRoot.w_807_0 ctx) (generatedRoot.w_812_0 ctx) (generatedRoot.w_817_0 ctx) (generatedRoot.w_822_0 ctx) (generatedRoot.w_827_0 ctx) (generatedRoot.w_832_0 ctx) (generatedRoot.w_837_0 ctx) (generatedRoot.w_842_0 ctx) (generatedRoot.w_847_0 ctx) (generatedRoot.w_852_0 ctx) (generatedRoot.w_857_0 ctx) (generatedRoot.w_860_0 ctx) (generatedRoot.w_862_0 ctx) (generatedRoot.w_864_0 ctx) (generatedRoot.w_866_0 ctx) (generatedRoot.w_867_0 ctx) (generatedRoot.w_868_0 ctx) (generatedRoot.w_869_0 ctx) (generatedRoot.w_870_0 ctx) (generatedRoot.w_870_1 ctx) (generatedRoot.w_870_2 ctx) (generatedRoot.w_870_3 ctx) (generatedRoot.w_880_0 ctx) (generatedRoot.w_881_0 ctx) (generatedRoot.w_884_0 ctx) (generatedRoot.w_885_0 ctx) (generatedRoot.w_889_0 ctx) (generatedRoot.w_889_1 ctx) (generatedRoot.w_889_2 ctx) (generatedRoot.w_889_3 ctx) (generatedRoot.w_889_4 ctx) (generatedRoot.w_889_5 ctx) (generatedRoot.w_890_0 ctx) (generatedRoot.w_890_1 ctx) (generatedRoot.w_890_2 ctx) (generatedRoot.w_890_3 ctx) (generatedRoot.w_890_4 ctx) (generatedRoot.w_890_5 ctx) (generatedRoot.w_893_0 ctx) (generatedRoot.w_893_1 ctx) (generatedRoot.w_893_2 ctx) (generatedRoot.w_893_3 ctx) (generatedRoot.w_893_4 ctx) (generatedRoot.w_893_5 ctx) (generatedRoot.w_894_0 ctx) (generatedRoot.w_894_1 ctx) (generatedRoot.w_894_2 ctx) (generatedRoot.w_894_3 ctx) (generatedRoot.w_894_4 ctx) (generatedRoot.w_894_5 ctx) (generatedRoot.w_895_0 ctx) (generatedRoot.w_895_1 ctx) (generatedRoot.w_895_2 ctx) (generatedRoot.w_895_3 ctx) (generatedRoot.w_895_4 ctx) (generatedRoot.w_895_5 ctx) (generatedRoot.w_896_0 ctx) (generatedRoot.w_896_1 ctx) (generatedRoot.w_896_2 ctx) (generatedRoot.w_896_3 ctx) (generatedRoot.w_896_4 ctx) (generatedRoot.w_896_5 ctx) (generatedRoot.w_897_0 ctx) (generatedRoot.w_898_0 ctx) (generatedRoot.w_900_0 ctx) (generatedRoot.w_901_0 ctx) (generatedRoot.w_903_0 ctx) (generatedRoot.w_904_0 ctx) ∧ generatedRoot.constraints_36 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_871_0 ctx) (generatedRoot.w_896_0 ctx) (generatedRoot.w_896_1 ctx) (generatedRoot.w_896_2 ctx) (generatedRoot.w_896_3 ctx) (generatedRoot.w_896_4 ctx) (generatedRoot.w_896_5 ctx) (generatedRoot.w_899_0 ctx) (generatedRoot.w_902_0 ctx) (generatedRoot.w_905_0 ctx) (generatedRoot.w_906_0 ctx) (generatedRoot.w_907_0 ctx) (generatedRoot.w_908_0 ctx) (generatedRoot.w_909_0 ctx) (generatedRoot.w_910_0 ctx) (generatedRoot.w_911_0 ctx) (generatedRoot.w_912_0 ctx) (generatedRoot.w_913_0 ctx) (generatedRoot.w_914_0 ctx) (generatedRoot.w_915_0 ctx) (generatedRoot.w_915_1 ctx) (generatedRoot.w_915_2 ctx) (generatedRoot.w_915_3 ctx) (generatedRoot.w_915_4 ctx) (generatedRoot.w_915_5 ctx) (generatedRoot.w_916_0 ctx) (generatedRoot.w_917_0 ctx) (generatedRoot.w_918_0 ctx) (generatedRoot.w_919_0 ctx) (generatedRoot.w_920_0 ctx) (generatedRoot.w_921_0 ctx) (generatedRoot.w_922_0 ctx) (generatedRoot.w_923_0 ctx) (generatedRoot.w_924_0 ctx) (generatedRoot.w_925_0 ctx) (generatedRoot.w_926_0 ctx) (generatedRoot.w_927_0 ctx) (generatedRoot.w_927_1 ctx) (generatedRoot.w_927_2 ctx) (generatedRoot.w_927_3 ctx) (generatedRoot.w_927_4 ctx) (generatedRoot.w_927_5 ctx) (generatedRoot.w_935_0 ctx) (generatedRoot.w_936_0 ctx) (generatedRoot.w_938_0 ctx) (generatedRoot.w_939_0 ctx))))) ∧ (((generatedRoot.constraints_37 backend hashModel tape path params (generatedRoot.w_5_0 ctx) (generatedRoot.w_11_0 ctx) (generatedRoot.w_12_0 ctx) (generatedRoot.w_24_0 ctx) (generatedRoot.w_101_0 ctx) (generatedRoot.w_106_0 ctx) (generatedRoot.w_120_0 ctx) (generatedRoot.w_929_0 ctx) (generatedRoot.w_930_0 ctx) (generatedRoot.w_931_0 ctx) (generatedRoot.w_932_0 ctx) (generatedRoot.w_933_0 ctx) (generatedRoot.w_934_0 ctx) (generatedRoot.w_937_0 ctx) (generatedRoot.w_940_0 ctx) (generatedRoot.w_941_0 ctx) (generatedRoot.w_942_0 ctx) (generatedRoot.w_945_0 ctx) (generatedRoot.w_946_0 ctx) (generatedRoot.w_928_0 ctx) (generatedRoot.w_935_0 ctx) (generatedRoot.w_936_0 ctx) (generatedRoot.w_938_0 ctx) (generatedRoot.w_939_0 ctx) (generatedRoot.w_943_0 ctx) (generatedRoot.w_943_1 ctx) (generatedRoot.w_943_2 ctx) (generatedRoot.w_943_3 ctx) (generatedRoot.w_943_4 ctx) (generatedRoot.w_943_5 ctx) (generatedRoot.w_944_0 ctx) (generatedRoot.w_944_1 ctx) (generatedRoot.w_944_2 ctx) (generatedRoot.w_944_3 ctx) (generatedRoot.w_944_4 ctx) (generatedRoot.w_944_5 ctx) (generatedRoot.w_947_0 ctx) (generatedRoot.w_947_1 ctx) (generatedRoot.w_947_2 ctx) (generatedRoot.w_947_3 ctx) (generatedRoot.w_947_4 ctx) (generatedRoot.w_947_5 ctx) (generatedRoot.w_948_0 ctx) (generatedRoot.w_948_1 ctx) (generatedRoot.w_948_2 ctx) (generatedRoot.w_948_3 ctx) (generatedRoot.w_948_4 ctx) (generatedRoot.w_948_5 ctx) (generatedRoot.w_949_0 ctx) (generatedRoot.w_949_1 ctx) (generatedRoot.w_949_2 ctx) (generatedRoot.w_949_3 ctx) (generatedRoot.w_949_4 ctx) (generatedRoot.w_949_5 ctx) (generatedRoot.w_950_0 ctx) (generatedRoot.w_950_1 ctx) (generatedRoot.w_950_2 ctx) (generatedRoot.w_950_3 ctx) (generatedRoot.w_950_4 ctx) (generatedRoot.w_950_5 ctx) (generatedRoot.w_951_0 ctx) (generatedRoot.w_952_0 ctx) (generatedRoot.w_953_0 ctx) (generatedRoot.w_954_0 ctx) (generatedRoot.w_955_0 ctx) (generatedRoot.w_956_0 ctx) (generatedRoot.w_957_0 ctx) (generatedRoot.w_958_0 ctx) (generatedRoot.w_959_0 ctx) (generatedRoot.w_960_0 ctx) (generatedRoot.w_961_0 ctx) (generatedRoot.w_962_0 ctx) (generatedRoot.w_963_0 ctx) (generatedRoot.w_964_0 ctx) (generatedRoot.w_965_0 ctx) (generatedRoot.w_966_0 ctx) (generatedRoot.w_967_0 ctx) (generatedRoot.w_968_0 ctx) (generatedRoot.w_969_0 ctx) (generatedRoot.w_969_1 ctx) (generatedRoot.w_969_2 ctx) (generatedRoot.w_969_3 ctx) (generatedRoot.w_969_4 ctx) (generatedRoot.w_969_5 ctx) (generatedRoot.w_970_0 ctx) (generatedRoot.w_971_0 ctx) (generatedRoot.w_972_0 ctx) ∧ (generatedRoot.constraints_38 backend hashModel tape path params (generatedRoot.w_55_0 ctx) (generatedRoot.w_61_0 ctx) (generatedRoot.w_68_0 ctx) (generatedRoot.w_530_6 ctx) (generatedRoot.w_530_8 ctx) (generatedRoot.w_652_0 ctx) (generatedRoot.w_656_0 ctx) (generatedRoot.w_870_1 ctx) (generatedRoot.w_870_2 ctx) (generatedRoot.w_870_3 ctx) (generatedRoot.w_927_1 ctx) (generatedRoot.w_927_3 ctx) (generatedRoot.w_950_0 ctx) (generatedRoot.w_950_1 ctx) (generatedRoot.w_950_2 ctx) (generatedRoot.w_950_3 ctx) (generatedRoot.w_950_4 ctx) (generatedRoot.w_950_5 ctx) (generatedRoot.w_969_2 ctx) (generatedRoot.w_969_3 ctx) (generatedRoot.w_969_4 ctx) (generatedRoot.w_969_5 ctx) (generatedRoot.w_970_0 ctx) (generatedRoot.w_972_0 ctx) (generatedRoot.w_973_0 ctx) (generatedRoot.w_974_0 ctx) (generatedRoot.w_975_0 ctx) (generatedRoot.w_976_0 ctx) (generatedRoot.w_977_0 ctx) (generatedRoot.w_978_0 ctx) (generatedRoot.w_979_0 ctx) (generatedRoot.w_980_0 ctx) (generatedRoot.w_981_0 ctx) (generatedRoot.w_981_1 ctx) (generatedRoot.w_981_2 ctx) (generatedRoot.w_981_3 ctx) (generatedRoot.w_981_4 ctx) (generatedRoot.w_981_5 ctx) (generatedRoot.w_982_0 ctx) (generatedRoot.w_983_0 ctx) (generatedRoot.w_984_0 ctx) (generatedRoot.w_985_0 ctx) (generatedRoot.w_986_0 ctx) (generatedRoot.w_987_0 ctx) (generatedRoot.w_988_0 ctx) (generatedRoot.w_989_0 ctx) (generatedRoot.w_990_0 ctx) (generatedRoot.w_991_0 ctx) ∧ generatedRoot.constraints_39 backend (generatedRoot.w_992_0 ctx) (generatedRoot.w_993_0 ctx) (generatedRoot.w_994_0 ctx) (generatedRoot.w_995_0 ctx) (generatedRoot.w_996_0 ctx) (generatedRoot.w_997_0 ctx) (generatedRoot.w_998_0 ctx) (generatedRoot.w_999_0 ctx) (generatedRoot.w_1001_0 ctx))) ∧ (generatedRoot.constraints_40 backend (generatedRoot.w_588_6 ctx) (generatedRoot.w_588_8 ctx) (generatedRoot.w_992_0 ctx) (generatedRoot.w_998_0 ctx) (generatedRoot.w_1003_0 ctx) (generatedRoot.w_1004_0 ctx) (generatedRoot.w_1007_0 ctx) (generatedRoot.w_1008_0 ctx) (generatedRoot.w_1009_0 ctx) (generatedRoot.w_1010_0 ctx) (generatedRoot.w_1011_0 ctx) (generatedRoot.w_1012_0 ctx) (generatedRoot.w_1013_0 ctx) (generatedRoot.w_1014_0 ctx) (generatedRoot.w_1015_0 ctx) ∧ (generatedRoot.constraints_41 backend hashModel (generatedRoot.w_5_0 ctx) (generatedRoot.w_14_0 ctx) (generatedRoot.w_20_0 ctx) (generatedRoot.w_240_6 ctx) (generatedRoot.w_240_8 ctx) (generatedRoot.w_1010_0 ctx) (generatedRoot.w_1016_0 ctx) (generatedRoot.w_1017_0 ctx) (generatedRoot.w_1019_0 ctx) (generatedRoot.w_1021_0 ctx) (generatedRoot.w_1022_0 ctx) (generatedRoot.w_1025_0 ctx) (generatedRoot.w_1026_0 ctx) (generatedRoot.w_1027_0 ctx) (generatedRoot.w_1028_0 ctx) (generatedRoot.w_1029_0 ctx) (generatedRoot.w_1030_0 ctx) (generatedRoot.w_1031_0 ctx) ∧ generatedRoot.constraints_42 backend (generatedRoot.w_1032_0 ctx) (generatedRoot.w_1033_0 ctx) (generatedRoot.w_1034_0 ctx) (generatedRoot.w_1035_0 ctx) (generatedRoot.w_1036_0 ctx) (generatedRoot.w_1037_0 ctx) (generatedRoot.w_1038_0 ctx) (generatedRoot.w_1039_0 ctx) (generatedRoot.w_1041_0 ctx)))) ∧ ((generatedRoot.constraints_43 backend (generatedRoot.w_298_6 ctx) (generatedRoot.w_298_8 ctx) (generatedRoot.w_1032_0 ctx) (generatedRoot.w_1038_0 ctx) (generatedRoot.w_1043_0 ctx) (generatedRoot.w_1044_0 ctx) (generatedRoot.w_1047_0 ctx) (generatedRoot.w_1048_0 ctx) (generatedRoot.w_1049_0 ctx) (generatedRoot.w_1050_0 ctx) (generatedRoot.w_1051_0 ctx) (generatedRoot.w_1052_0 ctx) (generatedRoot.w_1053_0 ctx) (generatedRoot.w_1054_0 ctx) (generatedRoot.w_1055_0 ctx) ∧ (generatedRoot.constraints_44 backend (generatedRoot.w_356_6 ctx) (generatedRoot.w_356_8 ctx) (generatedRoot.w_1050_0 ctx) (generatedRoot.w_1056_0 ctx) (generatedRoot.w_1057_0 ctx) (generatedRoot.w_1059_0 ctx) (generatedRoot.w_1061_0 ctx) (generatedRoot.w_1062_0 ctx) (generatedRoot.w_1065_0 ctx) (generatedRoot.w_1066_0 ctx) (generatedRoot.w_1067_0 ctx) ∧ generatedRoot.constraints_45 backend (generatedRoot.w_1068_0 ctx) (generatedRoot.w_1069_0 ctx) (generatedRoot.w_1070_0 ctx) (generatedRoot.w_1071_0 ctx) (generatedRoot.w_1072_0 ctx) (generatedRoot.w_1073_0 ctx) (generatedRoot.w_1074_0 ctx) (generatedRoot.w_1075_0 ctx) (generatedRoot.w_1077_0 ctx))) ∧ ((generatedRoot.constraints_46 backend (generatedRoot.w_414_6 ctx) (generatedRoot.w_414_8 ctx) (generatedRoot.w_1068_0 ctx) (generatedRoot.w_1074_0 ctx) (generatedRoot.w_1079_0 ctx) (generatedRoot.w_1080_0 ctx) (generatedRoot.w_1083_0 ctx) (generatedRoot.w_1084_0 ctx) (generatedRoot.w_1085_0 ctx) (generatedRoot.w_1086_0 ctx) (generatedRoot.w_1087_0 ctx) (generatedRoot.w_1088_0 ctx) (generatedRoot.w_1089_0 ctx) (generatedRoot.w_1090_0 ctx) (generatedRoot.w_1091_0 ctx) ∧ generatedRoot.constraints_47 backend (generatedRoot.w_472_6 ctx) (generatedRoot.w_472_8 ctx) (generatedRoot.w_1086_0 ctx) (generatedRoot.w_1092_0 ctx) (generatedRoot.w_1093_0 ctx) (generatedRoot.w_1095_0 ctx) (generatedRoot.w_1097_0 ctx) (generatedRoot.w_1098_0 ctx) (generatedRoot.w_1101_0 ctx) (generatedRoot.w_1102_0 ctx) (generatedRoot.w_1103_0 ctx)) ∧ (generatedRoot.constraints_48 backend (generatedRoot.w_1104_0 ctx) (generatedRoot.w_1105_0 ctx) (generatedRoot.w_1106_0 ctx) (generatedRoot.w_1107_0 ctx) (generatedRoot.w_1108_0 ctx) (generatedRoot.w_1109_0 ctx) (generatedRoot.w_1110_0 ctx) (generatedRoot.w_1111_0 ctx) (generatedRoot.w_1113_0 ctx) ∧ generatedRoot.constraints_49 (generatedRoot.w_19_0 ctx) (generatedRoot.w_23_0 ctx) (generatedRoot.w_17_0 ctx) (generatedRoot.w_18_0 ctx) (generatedRoot.w_21_0 ctx) (generatedRoot.w_22_0 ctx) (generatedRoot.w_54_0 ctx) (generatedRoot.w_54_1 ctx) (generatedRoot.w_54_2 ctx) (generatedRoot.w_54_3 ctx) (generatedRoot.w_54_4 ctx) (generatedRoot.w_54_5 ctx) (generatedRoot.w_62_0 ctx) (generatedRoot.w_87_0 ctx) (generatedRoot.w_194_0 ctx) (generatedRoot.w_214_0 ctx) (generatedRoot.w_273_0 ctx) (generatedRoot.w_331_0 ctx) (generatedRoot.w_389_0 ctx) (generatedRoot.w_447_0 ctx) (generatedRoot.w_505_0 ctx) (generatedRoot.w_563_0 ctx) (generatedRoot.w_621_0 ctx) (generatedRoot.w_655_0 ctx) (generatedRoot.w_662_0 ctx) (generatedRoot.w_665_0 ctx) (generatedRoot.w_671_0 ctx) (generatedRoot.w_676_0 ctx) (generatedRoot.w_681_0 ctx) (generatedRoot.w_686_0 ctx) (generatedRoot.w_691_0 ctx) (generatedRoot.w_696_0 ctx) (generatedRoot.w_701_0 ctx) (generatedRoot.w_709_0 ctx) (generatedRoot.w_712_0 ctx) (generatedRoot.w_718_0 ctx) (generatedRoot.w_723_0 ctx) (generatedRoot.w_728_0 ctx) (generatedRoot.w_733_0 ctx) (generatedRoot.w_738_0 ctx) (generatedRoot.w_743_0 ctx) (generatedRoot.w_748_0 ctx) (generatedRoot.w_927_0 ctx) (generatedRoot.w_927_1 ctx) (generatedRoot.w_927_2 ctx) (generatedRoot.w_927_3 ctx) (generatedRoot.w_927_4 ctx) (generatedRoot.w_927_5 ctx) (generatedRoot.w_981_0 ctx) (generatedRoot.w_981_1 ctx) (generatedRoot.w_981_2 ctx) (generatedRoot.w_981_3 ctx) (generatedRoot.w_981_4 ctx) (generatedRoot.w_981_5 ctx) (generatedRoot.w_984_0 ctx) (generatedRoot.w_985_0 ctx) (generatedRoot.w_986_0 ctx) (generatedRoot.w_987_0 ctx) (generatedRoot.w_988_0 ctx) (generatedRoot.w_1000_0 ctx) (generatedRoot.w_1002_0 ctx) (generatedRoot.w_1006_0 ctx) (generatedRoot.w_1018_0 ctx) (generatedRoot.w_1020_0 ctx) (generatedRoot.w_1024_0 ctx) (generatedRoot.w_1025_0 ctx) (generatedRoot.w_1026_0 ctx) (generatedRoot.w_1027_0 ctx) (generatedRoot.w_1028_0 ctx) (generatedRoot.w_1040_0 ctx) (generatedRoot.w_1042_0 ctx) (generatedRoot.w_1046_0 ctx) (generatedRoot.w_1058_0 ctx) (generatedRoot.w_1060_0 ctx) (generatedRoot.w_1064_0 ctx) (generatedRoot.w_1076_0 ctx) (generatedRoot.w_1078_0 ctx) (generatedRoot.w_1082_0 ctx) (generatedRoot.w_1094_0 ctx) (generatedRoot.w_1096_0 ctx) (generatedRoot.w_1100_0 ctx) (generatedRoot.w_1104_0 ctx) (generatedRoot.w_1110_0 ctx) (generatedRoot.w_1112_0 ctx) (generatedRoot.w_1114_0 ctx) (generatedRoot.w_1115_0 ctx) (generatedRoot.w_1116_0 ctx) (generatedRoot.w_1118_0 ctx) outputs))))))

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (generatedRoot.InputType)) (outputs : (generatedRoot.OutputType)) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_keys_online_0_0
