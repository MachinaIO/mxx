import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Stage_keys_online_0_0

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993
local notation "mxxNat1" => 29270270146341990167195247313302370525046749248022274291861926128744555509284
local notation "mxxNat2" => 60863387405588526063623438538454545359885565288667817947515354790518700822262
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

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 3 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 3 50
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 50 25
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 8 1 22500
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 8 1 900
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix modulus_0 8 1 112500
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix modulus_0 8 1 4500
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix modulus_0 8 1 625
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix modulus_0 8 25 1
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix modulus_0 8 1 50
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix modulus_1 8 1 25
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix modulus_0 8 25 25
local notation "mxxMat15" => Mxx.Primitives.ExactMatrix modulus_0 8 25 625
local notation "mxxMat16" => Mxx.Primitives.ExactMatrix modulus_0 8 2 50
local notation "mxxMat17" => Mxx.Primitives.ExactMatrix modulus_0 8 3 75
local notation "mxxMat18" => Mxx.Primitives.ExactMatrix modulus_0 8 4 100
local notation "mxxMat19" => Mxx.Primitives.ExactMatrix modulus_0 8 5 125
local notation "mxxMat20" => Mxx.Primitives.ExactMatrix modulus_0 8 6 150
local notation "mxxMat21" => Mxx.Primitives.ExactMatrix modulus_0 8 7 175
local notation "mxxMat22" => Mxx.Primitives.ExactMatrix modulus_0 8 8 200
local notation "mxxMat23" => Mxx.Primitives.ExactMatrix modulus_0 8 9 225
local notation "mxxMat24" => Mxx.Primitives.ExactMatrix modulus_0 8 10 250
local notation "mxxMat25" => Mxx.Primitives.ExactMatrix modulus_0 8 11 275
local notation "mxxMat26" => Mxx.Primitives.ExactMatrix modulus_0 8 12 300
local notation "mxxMat27" => Mxx.Primitives.ExactMatrix modulus_0 8 13 325
local notation "mxxMat28" => Mxx.Primitives.ExactMatrix modulus_0 8 14 350
local notation "mxxMat29" => Mxx.Primitives.ExactMatrix modulus_0 8 15 375
local notation "mxxMat30" => Mxx.Primitives.ExactMatrix modulus_0 8 16 400
local notation "mxxMat31" => Mxx.Primitives.ExactMatrix modulus_0 8 17 425
local notation "mxxMat32" => Mxx.Primitives.ExactMatrix modulus_0 8 18 450
local notation "mxxMat33" => Mxx.Primitives.ExactMatrix modulus_0 8 19 475
local notation "mxxMat34" => Mxx.Primitives.ExactMatrix modulus_0 8 20 500
local notation "mxxMat35" => Mxx.Primitives.ExactMatrix modulus_0 8 21 525
local notation "mxxMat36" => Mxx.Primitives.ExactMatrix modulus_0 8 22 550
local notation "mxxMat37" => Mxx.Primitives.ExactMatrix modulus_0 8 23 575
local notation "mxxMat38" => Mxx.Primitives.ExactMatrix modulus_0 8 24 600
local notation "mxxMat39" => Mxx.Primitives.ExactMatrix modulus_0 8 625 625
local notation "mxxMat40" => Mxx.Primitives.ExactMatrix modulus_0 8 2 2
local notation "mxxMat41" => Mxx.Primitives.ExactMatrix modulus_0 8 3 3
local notation "mxxMat42" => Mxx.Primitives.ExactMatrix modulus_0 8 4 4
local notation "mxxMat43" => Mxx.Primitives.ExactMatrix modulus_0 8 5 5
local notation "mxxMat44" => Mxx.Primitives.ExactMatrix modulus_0 8 6 6
local notation "mxxMat45" => Mxx.Primitives.ExactMatrix modulus_0 8 7 7
local notation "mxxMat46" => Mxx.Primitives.ExactMatrix modulus_0 8 8 8
local notation "mxxMat47" => Mxx.Primitives.ExactMatrix modulus_0 8 9 9
local notation "mxxMat48" => Mxx.Primitives.ExactMatrix modulus_0 8 10 10
local notation "mxxMat49" => Mxx.Primitives.ExactMatrix modulus_0 8 11 11
local notation "mxxMat50" => Mxx.Primitives.ExactMatrix modulus_0 8 12 12
local notation "mxxMat51" => Mxx.Primitives.ExactMatrix modulus_0 8 13 13
local notation "mxxMat52" => Mxx.Primitives.ExactMatrix modulus_0 8 14 14
local notation "mxxMat53" => Mxx.Primitives.ExactMatrix modulus_0 8 15 15
local notation "mxxMat54" => Mxx.Primitives.ExactMatrix modulus_0 8 16 16
local notation "mxxMat55" => Mxx.Primitives.ExactMatrix modulus_0 8 17 17
local notation "mxxMat56" => Mxx.Primitives.ExactMatrix modulus_0 8 18 18
local notation "mxxMat57" => Mxx.Primitives.ExactMatrix modulus_0 8 19 19
local notation "mxxMat58" => Mxx.Primitives.ExactMatrix modulus_0 8 20 20
local notation "mxxMat59" => Mxx.Primitives.ExactMatrix modulus_0 8 21 21
local notation "mxxMat60" => Mxx.Primitives.ExactMatrix modulus_0 8 22 22
local notation "mxxMat61" => Mxx.Primitives.ExactMatrix modulus_0 8 23 23
local notation "mxxMat62" => Mxx.Primitives.ExactMatrix modulus_0 8 24 24
local notation "mxxMat63" => Mxx.Primitives.ExactMatrix modulus_0 8 625 25
local notation "mxxMat64" => Mxx.Primitives.ExactMatrix modulus_1 8 1 1
local notation "mxxMat65" => Mxx.Primitives.ExactMatrix modulus_2 8 1 25
local notation "mxxMat66" => Mxx.Primitives.ExactMatrix modulus_2 8 1 1
local notation "mxxMat67" => Mxx.Primitives.ExactMatrix modulus_3 8 1 25
local notation "mxxMat68" => Mxx.Primitives.ExactMatrix modulus_3 8 1 1
local notation "mxxMat69" => Mxx.Primitives.ExactMatrix modulus_4 8 1 25
local notation "mxxMat70" => Mxx.Primitives.ExactMatrix modulus_4 8 1 1
local notation "mxxMat71" => Mxx.Primitives.ExactMatrix modulus_5 8 1 25
local notation "mxxMat72" => Mxx.Primitives.ExactMatrix modulus_5 8 1 1
local notation "mxxMat73" => Mxx.Primitives.ExactMatrix modulus_0 8 2 46
local notation "mxxMat74" => Mxx.Primitives.ExactMatrix modulus_0 8 2 100
local notation "mxxMat75" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54

abbrev parallel_generatedRoot_27.constraints_3 (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_56_0 : mxxMat1) (w_57_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_27.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_39_0 : ByteArray) (w_41_0 : mxxMat1) (w_48_0 : mxxMat1) (w_24_0 : mxxMat0) (w_30_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_37_concat_1 : mxxMat2) (w_38_0 : mxxMat3) (w_40_0 : mxxMat1) (w_42_0 : mxxMat2) (w_43_0 : mxxMat4) (w_44_0 : mxxMat4) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_27.constraints_1 (backend : MxxRuntime.BackendContext) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_16_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat0) (w_24_concat_1 : mxxMat2) (w_27_0 : mxxMat2) (w_28_0 : mxxMat4) (w_29_0 : mxxMat4) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_35_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_27.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat0) (w_5_concat_1 : mxxMat2) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_9_concat_1 : mxxMat2) (w_10_0 : mxxMat3) (w_13_0 : mxxMat2) (w_14_0 : mxxMat4) (w_15_0 : mxxMat4) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) : Prop :=
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

def parallel_generatedRoot_27 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat1 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.1
  let w_11_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_18_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_19_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_5_0 : mxxMat0),
        ∃ (w_5_concat_1 : mxxMat2),
          ∃ (w_7_0 : mxxMat1),
            ∃ (w_8_0 : mxxMat1),
              ∃ (w_9_0 : mxxMat0),
                ∃ (w_9_concat_1 : mxxMat2),
                  ∃ (w_10_0 : mxxMat3),
                    ∃ (w_13_0 : mxxMat2),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_15_0 : mxxMat4),
                          let w_16_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_15_0
                          ∃ (w_17_0 : mxxMat1),
                            let w_20_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_18_0 w_19_0
                            let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_20_0
                            ∃ (w_22_0 : mxxMat1),
                              ∃ (w_23_0 : mxxMat1),
                                ∃ (w_24_0 : mxxMat0),
                                  ∃ (w_24_concat_1 : mxxMat2),
                                    ∃ (w_27_0 : mxxMat2),
                                      ∃ (w_28_0 : mxxMat4),
                                        ∃ (w_29_0 : mxxMat4),
                                          let w_30_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_29_0
                                          ∃ (w_31_0 : mxxMat1),
                                            let w_33_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_32_0 w_19_0
                                            let w_34_0 : mxxMat1 := MxxRuntime.matrixAdd w_31_0 w_33_0
                                            ∃ (w_35_0 : mxxMat1),
                                              ∃ (w_36_0 : mxxMat1),
                                                ∃ (w_37_0 : mxxMat0),
                                                  ∃ (w_37_concat_1 : mxxMat2),
                                                    ∃ (w_38_0 : mxxMat3),
                                                      ∃ (w_40_0 : mxxMat1),
                                                        ∃ (w_42_0 : mxxMat2),
                                                          ∃ (w_43_0 : mxxMat4),
                                                            ∃ (w_44_0 : mxxMat4),
                                                              let w_45_0 : mxxMat0 := MxxRuntime.matrixMul w_38_0 w_44_0
                                                              ∃ (w_46_0 : mxxMat1),
                                                                ∃ (w_47_0 : mxxMat1),
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_38_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat1),
                                                                              ∃ (w_56_0 : mxxMat1),
                                                                                ∃ (w_57_0 : mxxMat1),
                                                                                  parallel_generatedRoot_27.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_12_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 ∧
    parallel_generatedRoot_27.constraints_1 backend w_25_0 w_26_0 w_16_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_concat_1 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 ∧
    parallel_generatedRoot_27.constraints_2 backend hashModel i_0 w_39_0 w_41_0 w_48_0 w_24_0 w_30_0 w_34_0 w_35_0 w_36_0 w_37_0 w_37_concat_1 w_38_0 w_40_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 ∧
    parallel_generatedRoot_27.constraints_3 w_45_0 w_46_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 outputs

abbrev parallel_parallel_generatedRoot_28_6.constraints_0 (w_0_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_5_0 : Fin 16 → mxxMat1) (w_7_0 : Fin 16 → mxxMat1) (w_9_0 : Fin 16 → mxxMat1) (w_11_0 : Fin 16 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_10_0 : mxxMat1) (w_12_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_28_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat1 := inputs.1
  let w_3_0 : Fin 16 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 16 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 16 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.1
  let w_11_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          ∃ (w_10_0 : mxxMat1),
            ∃ (w_12_0 : mxxMat1),
              parallel_parallel_generatedRoot_28_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_28_25.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_18_0 : Fin 2 → mxxMat1) (w_22_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_28_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_parallel_generatedRoot_28_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_28.Witness where
  w_6_0 : Fin 2 → mxxMat1
  w_6_1 : Fin 2 → mxxMat1
  w_6_2 : Fin 2 → mxxMat1
  w_6_3 : Fin 2 → mxxMat1
  w_6_4 : Fin 2 → mxxMat1
  w_6_5 : Fin 2 → mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat1
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_16_0 : mxxMat1
  w_17_0 : mxxMat1
  w_19_0 : mxxMat1
  w_20_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat1
  w_25_0 : mxxMat1
  w_25_1 : mxxMat1
  w_25_2 : mxxMat1
  w_25_3 : mxxMat1
  w_25_4 : mxxMat1
  w_25_5 : mxxMat1

abbrev parallel_generatedRoot_28.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat1) (w_1_0 : Fin 16 → mxxMat1) (w_2_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_4_0 : Fin 16 → mxxMat1) (w_5_0 : Fin 16 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_6_1 : Fin 2 → mxxMat1) (w_6_2 : Fin 2 → mxxMat1) (w_6_3 : Fin 2 → mxxMat1) (w_6_4 : Fin 2 → mxxMat1) (w_6_5 : Fin 2 → mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_25_1 : mxxMat1) (w_25_2 : mxxMat1) (w_25_3 : mxxMat1) (w_25_4 : mxxMat1) (w_25_5 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_28_6 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_28_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_28.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_28.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat1 := inputs.1
  let w_1_0 : Fin 16 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 16 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 16 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.1
  let w_5_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_6_4 := witness.w_6_4
  let w_6_5 := witness.w_6_5
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat1 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_25_2 := witness.w_25_2
  let w_25_3 := witness.w_25_3
  let w_25_4 := witness.w_25_4
  let w_25_5 := witness.w_25_5
  parallel_generatedRoot_28.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_28 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_28.Witness,
    parallel_generatedRoot_28.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_31.constraints_4 (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) (w_61_0 : mxxMat1) (w_62_0 : mxxMat1) (w_63_0 : mxxMat1) (w_64_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_31.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_46_0 : ByteArray) (w_55_0 : mxxMat1) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_56_0 : mxxMat2) (w_57_0 : mxxMat4) (w_58_0 : mxxMat4) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) (w_61_0 : mxxMat1) (w_62_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_31.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_38_0 : Fin 8 → mxxMat1) (w_46_0 : ByteArray) (w_48_0 : mxxMat1) (w_12_0 : Int) (w_28_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat1) (w_39_0 : mxxMat1) (w_41_0 : mxxMat1) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat0) (w_44_concat_1 : mxxMat2) (w_45_0 : mxxMat3) (w_47_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_31.constraints_1 (backend : MxxRuntime.BackendContext) (w_21_0 : Fin 8 → mxxMat1) (w_29_0 : Fin 8 → mxxMat1) (w_31_0 : Fin 8 → mxxMat1) (w_12_0 : Int) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_30_0 : mxxMat1) (w_32_0 : mxxMat1) (w_33_0 : mxxMat2) (w_34_0 : mxxMat4) (w_35_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_31.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_11_0 : Fin 8 → mxxMat1) (w_14_0 : Fin 8 → mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat0) (w_5_concat_1 : mxxMat2) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_9_concat_1 : mxxMat2) (w_10_0 : mxxMat3) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat2) (w_17_0 : mxxMat4) (w_18_0 : mxxMat4) : Prop :=
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

def parallel_generatedRoot_31 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat1 × ByteArray × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat5 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat1 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.1
  let w_14_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_21_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_23_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_5_0 : mxxMat0),
        ∃ (w_5_concat_1 : mxxMat2),
          ∃ (w_7_0 : mxxMat1),
            ∃ (w_8_0 : mxxMat1),
              ∃ (w_9_0 : mxxMat0),
                ∃ (w_9_concat_1 : mxxMat2),
                  ∃ (w_10_0 : mxxMat3),
                    let w_12_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (2))
                    ∃ (w_13_0 : mxxMat1),
                      ∃ (w_15_0 : mxxMat1),
                        ∃ (w_16_0 : mxxMat2),
                          ∃ (w_17_0 : mxxMat4),
                            ∃ (w_18_0 : mxxMat4),
                              let w_19_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_18_0
                              ∃ (w_20_0 : mxxMat1),
                                ∃ (w_22_0 : mxxMat1),
                                  let w_24_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_22_0 w_23_0
                                  let w_25_0 : mxxMat1 := MxxRuntime.matrixAdd w_20_0 w_24_0
                                  ∃ (w_26_0 : mxxMat1),
                                    ∃ (w_27_0 : mxxMat1),
                                      ∃ (w_28_0 : mxxMat0),
                                        ∃ (w_28_concat_1 : mxxMat2),
                                          ∃ (w_30_0 : mxxMat1),
                                            ∃ (w_32_0 : mxxMat1),
                                              ∃ (w_33_0 : mxxMat2),
                                                ∃ (w_34_0 : mxxMat4),
                                                  ∃ (w_35_0 : mxxMat4),
                                                    let w_36_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_35_0
                                                    ∃ (w_37_0 : mxxMat1),
                                                      ∃ (w_39_0 : mxxMat1),
                                                        let w_40_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_39_0 w_23_0
                                                        let w_41_0 : mxxMat1 := MxxRuntime.matrixAdd w_37_0 w_40_0
                                                        ∃ (w_42_0 : mxxMat1),
                                                          ∃ (w_43_0 : mxxMat1),
                                                            ∃ (w_44_0 : mxxMat0),
                                                              ∃ (w_44_concat_1 : mxxMat2),
                                                                ∃ (w_45_0 : mxxMat3),
                                                                  ∃ (w_47_0 : mxxMat1),
                                                                    ∃ (w_49_0 : mxxMat2),
                                                                      ∃ (w_50_0 : mxxMat4),
                                                                        ∃ (w_51_0 : mxxMat4),
                                                                          let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_45_0 w_51_0
                                                                          ∃ (w_53_0 : mxxMat1),
                                                                            ∃ (w_54_0 : mxxMat1),
                                                                              ∃ (w_56_0 : mxxMat2),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  ∃ (w_58_0 : mxxMat4),
                                                                                    let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_45_0 w_58_0
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      ∃ (w_61_0 : mxxMat1),
                                                                                        ∃ (w_62_0 : mxxMat1),
                                                                                          ∃ (w_63_0 : mxxMat1),
                                                                                            ∃ (w_64_0 : mxxMat1),
                                                                                              parallel_generatedRoot_31.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_14_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_31.constraints_1 backend w_21_0 w_29_0 w_31_0 w_12_0 w_19_0 w_20_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_concat_1 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 ∧
    parallel_generatedRoot_31.constraints_2 backend hashModel i_0 w_38_0 w_46_0 w_48_0 w_12_0 w_28_0 w_36_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_concat_1 w_45_0 w_47_0 w_49_0 w_50_0 w_51_0 ∧
    parallel_generatedRoot_31.constraints_3 backend hashModel i_0 w_46_0 w_55_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 ∧
    parallel_generatedRoot_31.constraints_4 w_52_0 w_53_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_parallel_generatedRoot_32_6.constraints_0 (w_0_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_5_0 : Fin 16 → mxxMat1) (w_7_0 : Fin 16 → mxxMat1) (w_9_0 : Fin 16 → mxxMat1) (w_11_0 : Fin 16 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_10_0 : mxxMat1) (w_12_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_32_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat1 := inputs.1
  let w_3_0 : Fin 16 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 16 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 16 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.1
  let w_11_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          ∃ (w_10_0 : mxxMat1),
            ∃ (w_12_0 : mxxMat1),
              parallel_parallel_generatedRoot_32_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_32_25.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_18_0 : Fin 2 → mxxMat1) (w_22_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_32_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_parallel_generatedRoot_32_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_32.Witness where
  w_6_0 : Fin 2 → mxxMat1
  w_6_1 : Fin 2 → mxxMat1
  w_6_2 : Fin 2 → mxxMat1
  w_6_3 : Fin 2 → mxxMat1
  w_6_4 : Fin 2 → mxxMat1
  w_6_5 : Fin 2 → mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat1
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_16_0 : mxxMat1
  w_17_0 : mxxMat1
  w_19_0 : mxxMat1
  w_20_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat1
  w_25_0 : mxxMat1
  w_25_1 : mxxMat1
  w_25_2 : mxxMat1
  w_25_3 : mxxMat1
  w_25_4 : mxxMat1
  w_25_5 : mxxMat1

abbrev parallel_generatedRoot_32.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat1) (w_1_0 : Fin 16 → mxxMat1) (w_2_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_4_0 : Fin 16 → mxxMat1) (w_5_0 : Fin 16 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_6_1 : Fin 2 → mxxMat1) (w_6_2 : Fin 2 → mxxMat1) (w_6_3 : Fin 2 → mxxMat1) (w_6_4 : Fin 2 → mxxMat1) (w_6_5 : Fin 2 → mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_25_1 : mxxMat1) (w_25_2 : mxxMat1) (w_25_3 : mxxMat1) (w_25_4 : mxxMat1) (w_25_5 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_32_6 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_32_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_32.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_32.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat1 := inputs.1
  let w_1_0 : Fin 16 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 16 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 16 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.1
  let w_5_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_6_4 := witness.w_6_4
  let w_6_5 := witness.w_6_5
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat1 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_25_2 := witness.w_25_2
  let w_25_3 := witness.w_25_3
  let w_25_4 := witness.w_25_4
  let w_25_5 := witness.w_25_5
  parallel_generatedRoot_32.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_32 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_32.Witness,
    parallel_generatedRoot_32.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_33_18.constraints_4 (w_75_0 : Fin 8 → mxxMat1) (w_79_0 : Fin 8 → mxxMat1) (w_81_0 : Fin 8 → mxxMat1) (w_83_0 : Fin 8 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_33_18.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_parallel_generatedRoot_33_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_parallel_generatedRoot_33_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_parallel_generatedRoot_33_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_33_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_parallel_generatedRoot_33_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_33_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_33_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_33_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_33_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_33.Witness where
  w_2_0 : mxxMat1
  w_4_0 : mxxMat1
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_18_0 : mxxMat1
  w_18_1 : mxxMat1
  w_18_2 : mxxMat1
  w_18_3 : mxxMat1
  w_18_4 : mxxMat1
  w_18_5 : mxxMat1
  w_18_6 : mxxMat1
  w_18_7 : mxxMat1
  w_18_8 : mxxMat1
  w_18_9 : mxxMat1
  w_19_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat0
  w_23_concat_1 : mxxMat2
  w_24_0 : mxxMat1
  w_25_0 : mxxMat1
  w_26_0 : mxxMat1
  w_27_0 : mxxMat0
  w_27_concat_1 : mxxMat2
  w_28_0 : mxxMat3
  w_29_0 : mxxMat1
  w_30_0 : mxxMat1
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_32_concat_1 : mxxMat2
  w_33_0 : mxxMat1
  w_34_0 : mxxMat1
  w_35_0 : mxxMat1
  w_36_0 : mxxMat0
  w_36_concat_1 : mxxMat2
  w_37_0 : mxxMat3
  w_38_0 : mxxMat2
  w_39_0 : mxxMat4
  w_40_0 : mxxMat4
  w_42_0 : mxxMat1
  w_43_0 : mxxMat1
  w_44_0 : mxxMat2
  w_45_0 : mxxMat4
  w_46_0 : mxxMat4
  w_48_0 : mxxMat1
  w_50_0 : mxxMat2
  w_51_0 : mxxMat4
  w_52_0 : mxxMat4
  w_54_0 : mxxMat1
  w_55_0 : mxxMat1
  w_56_0 : mxxMat2
  w_57_0 : mxxMat4
  w_58_0 : mxxMat4
  w_60_0 : mxxMat1
  w_62_0 : mxxMat1
  w_64_0 : mxxMat1
  w_66_0 : mxxMat1
  w_67_0 : mxxMat1
  w_70_0 : mxxMat1
  w_71_0 : mxxMat1

abbrev parallel_generatedRoot_33.constraints_4 (w_49_0 : mxxMat1) (w_53_0 : mxxMat0) (w_59_0 : mxxMat0) (w_61_0 : mxxMat1) (w_63_0 : mxxMat1) (w_65_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_33.constraints_3 (w_41_0 : mxxMat0) (w_47_0 : mxxMat0) (w_59_0 : mxxMat0) (w_62_0 : mxxMat1) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_33.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat1) (w_18_9 : mxxMat1) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_50_0 : mxxMat2) (w_51_0 : mxxMat4) (w_52_0 : mxxMat4) (w_53_0 : mxxMat0) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_56_0 : mxxMat2) (w_57_0 : mxxMat4) (w_58_0 : mxxMat4) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_33.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_6 : mxxMat1) (w_18_8 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat1) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_36_0 : mxxMat0) (w_36_concat_1 : mxxMat2) (w_37_0 : mxxMat3) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat2) (w_45_0 : mxxMat4) (w_46_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_33.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat1) (w_3_0 : Fin 8 → mxxMat1) (w_5_0 : Fin 8 → mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_0_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_18_0 : mxxMat1) (w_18_1 : mxxMat1) (w_18_2 : mxxMat1) (w_18_3 : mxxMat1) (w_18_4 : mxxMat1) (w_18_5 : mxxMat1) (w_18_6 : mxxMat1) (w_18_7 : mxxMat1) (w_18_8 : mxxMat1) (w_18_9 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : Int) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat0) (w_23_concat_1 : mxxMat2) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat0) (w_27_concat_1 : mxxMat2) (w_28_0 : mxxMat3) (w_29_0 : mxxMat1) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_32_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_33_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
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

abbrev parallel_generatedRoot_33.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_33.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 8 → mxxMat1 := inputs.1
  let w_3_0 : Fin 8 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 8 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 8 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.1
  let w_10_0 : ByteArray := inputs.2.2.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
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
  let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_28_0 w_58_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_3 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_4 w_62_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_5 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 : mxxMat1 := MxxRuntime.matrixAdd w_70_0 w_71_0
  let w_73_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_1 w_72_0
  parallel_generatedRoot_33.constraints_0 backend hashModel params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_33.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_33.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_33.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_33.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_33 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_33.Witness,
    parallel_generatedRoot_33.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_34_18.constraints_4 (w_75_0 : Fin 8 → mxxMat1) (w_79_0 : Fin 8 → mxxMat1) (w_81_0 : Fin 8 → mxxMat1) (w_83_0 : Fin 8 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_34_18.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_parallel_generatedRoot_34_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_parallel_generatedRoot_34_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_parallel_generatedRoot_34_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_34_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_parallel_generatedRoot_34_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_34_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_34_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_34_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_34_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_34.Witness where
  w_2_0 : mxxMat1
  w_4_0 : mxxMat1
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_18_0 : mxxMat1
  w_18_1 : mxxMat1
  w_18_2 : mxxMat1
  w_18_3 : mxxMat1
  w_18_4 : mxxMat1
  w_18_5 : mxxMat1
  w_18_6 : mxxMat1
  w_18_7 : mxxMat1
  w_18_8 : mxxMat1
  w_18_9 : mxxMat1
  w_19_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat0
  w_23_concat_1 : mxxMat2
  w_24_0 : mxxMat1
  w_25_0 : mxxMat1
  w_26_0 : mxxMat1
  w_27_0 : mxxMat0
  w_27_concat_1 : mxxMat2
  w_28_0 : mxxMat3
  w_29_0 : mxxMat1
  w_30_0 : mxxMat1
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_32_concat_1 : mxxMat2
  w_33_0 : mxxMat1
  w_34_0 : mxxMat1
  w_35_0 : mxxMat1
  w_36_0 : mxxMat0
  w_36_concat_1 : mxxMat2
  w_37_0 : mxxMat3
  w_38_0 : mxxMat2
  w_39_0 : mxxMat4
  w_40_0 : mxxMat4
  w_42_0 : mxxMat1
  w_43_0 : mxxMat1
  w_44_0 : mxxMat2
  w_45_0 : mxxMat4
  w_46_0 : mxxMat4
  w_48_0 : mxxMat1
  w_50_0 : mxxMat2
  w_51_0 : mxxMat4
  w_52_0 : mxxMat4
  w_54_0 : mxxMat1
  w_55_0 : mxxMat1
  w_56_0 : mxxMat2
  w_57_0 : mxxMat4
  w_58_0 : mxxMat4
  w_60_0 : mxxMat1
  w_62_0 : mxxMat1
  w_64_0 : mxxMat1
  w_66_0 : mxxMat1
  w_67_0 : mxxMat1
  w_70_0 : mxxMat1
  w_71_0 : mxxMat1

abbrev parallel_generatedRoot_34.constraints_4 (w_49_0 : mxxMat1) (w_53_0 : mxxMat0) (w_59_0 : mxxMat0) (w_61_0 : mxxMat1) (w_63_0 : mxxMat1) (w_65_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_34.constraints_3 (w_41_0 : mxxMat0) (w_47_0 : mxxMat0) (w_59_0 : mxxMat0) (w_62_0 : mxxMat1) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_34.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat1) (w_18_9 : mxxMat1) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_50_0 : mxxMat2) (w_51_0 : mxxMat4) (w_52_0 : mxxMat4) (w_53_0 : mxxMat0) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_56_0 : mxxMat2) (w_57_0 : mxxMat4) (w_58_0 : mxxMat4) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_34.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_6 : mxxMat1) (w_18_8 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat1) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_36_0 : mxxMat0) (w_36_concat_1 : mxxMat2) (w_37_0 : mxxMat3) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat2) (w_45_0 : mxxMat4) (w_46_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_34.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat1) (w_3_0 : Fin 8 → mxxMat1) (w_5_0 : Fin 8 → mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_0_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_18_0 : mxxMat1) (w_18_1 : mxxMat1) (w_18_2 : mxxMat1) (w_18_3 : mxxMat1) (w_18_4 : mxxMat1) (w_18_5 : mxxMat1) (w_18_6 : mxxMat1) (w_18_7 : mxxMat1) (w_18_8 : mxxMat1) (w_18_9 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : Int) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat0) (w_23_concat_1 : mxxMat2) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat0) (w_27_concat_1 : mxxMat2) (w_28_0 : mxxMat3) (w_29_0 : mxxMat1) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_32_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_34_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
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

abbrev parallel_generatedRoot_34.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_34.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 8 → mxxMat1 := inputs.1
  let w_3_0 : Fin 8 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 8 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 8 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.1
  let w_10_0 : ByteArray := inputs.2.2.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
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
  let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_28_0 w_58_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_3 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_4 w_62_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_5 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 : mxxMat1 := MxxRuntime.matrixAdd w_70_0 w_71_0
  let w_73_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_1 w_72_0
  parallel_generatedRoot_34.constraints_0 backend hashModel params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_34.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_34.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_34.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_34.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_34 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_34.Witness,
    parallel_generatedRoot_34.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_35.constraints_0 (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_14_0 : mxxMat1) (w_17_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  outputs = (w_2_0, (w_5_0, (w_8_0, (w_11_0, (w_14_0, (w_17_0, ()))))))

def parallel_generatedRoot_35 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_3_0 : mxxMat1 := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_0_0 w_1_0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  let w_7_0 : Int := (Int.ofNat i_0)
  let w_8_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_6_0 w_7_0
  let w_10_0 : Int := (Int.ofNat i_0)
  let w_11_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_9_0 w_10_0
  let w_13_0 : Int := (Int.ofNat i_0)
  let w_14_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_12_0 w_13_0
  let w_16_0 : Int := (Int.ofNat i_0)
  let w_17_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_15_0 w_16_0
  parallel_generatedRoot_35.constraints_0 w_2_0 w_5_0 w_8_0 w_11_0 w_14_0 w_17_0 outputs

abbrev sequential_generatedRoot_54.constraints_0 (w_1_0 : Fin 8 → mxxMat1) (w_6_0 : Fin 8 → mxxMat1) (w_10_0 : Fin 8 → mxxMat1) (w_14_0 : Fin 8 → mxxMat1) (w_18_0 : Fin 8 → mxxMat1) (w_22_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_generatedRoot_54 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_generatedRoot_54.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev sequential_generatedRoot_315.constraints_4 (w_75_0 : Fin 4 → mxxMat1) (w_79_0 : Fin 4 → mxxMat1) (w_81_0 : Fin 4 → mxxMat1) (w_83_0 : Fin 4 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_315.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_315.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_315.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_315.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 4 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 4 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_generatedRoot_315 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 4 → mxxMat1) × ByteArray × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_generatedRoot_315.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_315.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_315.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_315.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_315.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_675.constraints_4 (w_75_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_675.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_675.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_675.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_675.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_generatedRoot_675 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × ByteArray × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_generatedRoot_675.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_675.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_675.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_675.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_675.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_731.constraints_4 (w_75_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_731.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_731.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_731.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_731.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_generatedRoot_731 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × ByteArray × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_generatedRoot_731.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_731.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_731.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_731.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_731.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_787.constraints_4 (w_75_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_787.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_787.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_787.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_787.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_generatedRoot_787 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × ByteArray × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(7)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_generatedRoot_787.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_787.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_787.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_787.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_787.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_843.constraints_4 (w_75_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_843.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_843.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_843.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_843.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_generatedRoot_843 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × ByteArray × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(9)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_generatedRoot_843.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_843.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_843.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_843.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_843.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_899.constraints_4 (w_75_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_899.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_899.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_899.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_899.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_generatedRoot_899 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × ByteArray × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(11)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_generatedRoot_899.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_899.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_899.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_899.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_899.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_955.constraints_4 (w_75_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_955.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_955.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_955.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_955.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_generatedRoot_955 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × ByteArray × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(13)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_generatedRoot_955.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_955.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_955.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_955.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_955.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev sequential_generatedRoot_1011.constraints_4 (w_75_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_1011.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_1011.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_1011.constraints_1 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_generatedRoot_1011.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 1 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_generatedRoot_1011 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × ByteArray × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_generatedRoot_1011.constraints_0 hashModel i_0 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_generatedRoot_1011.constraints_1 backend i_0 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_generatedRoot_1011.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_generatedRoot_1011.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_generatedRoot_1011.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

abbrev parallel_generatedRoot_1076.constraints_3 (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_56_0 : mxxMat1) (w_57_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_1076.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_39_0 : ByteArray) (w_41_0 : mxxMat1) (w_48_0 : mxxMat1) (w_24_0 : mxxMat0) (w_30_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_37_concat_1 : mxxMat2) (w_38_0 : mxxMat3) (w_40_0 : mxxMat1) (w_42_0 : mxxMat2) (w_43_0 : mxxMat4) (w_44_0 : mxxMat4) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1076.constraints_1 (backend : MxxRuntime.BackendContext) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_16_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat0) (w_24_concat_1 : mxxMat2) (w_27_0 : mxxMat2) (w_28_0 : mxxMat4) (w_29_0 : mxxMat4) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_35_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1076.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat0) (w_5_concat_1 : mxxMat2) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_9_concat_1 : mxxMat2) (w_10_0 : mxxMat3) (w_13_0 : mxxMat2) (w_14_0 : mxxMat4) (w_15_0 : mxxMat4) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) : Prop :=
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

def parallel_generatedRoot_1076 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat1 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.1
  let w_11_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_18_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_19_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_5_0 : mxxMat0),
        ∃ (w_5_concat_1 : mxxMat2),
          ∃ (w_7_0 : mxxMat1),
            ∃ (w_8_0 : mxxMat1),
              ∃ (w_9_0 : mxxMat0),
                ∃ (w_9_concat_1 : mxxMat2),
                  ∃ (w_10_0 : mxxMat3),
                    ∃ (w_13_0 : mxxMat2),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_15_0 : mxxMat4),
                          let w_16_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_15_0
                          ∃ (w_17_0 : mxxMat1),
                            let w_20_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_18_0 w_19_0
                            let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_20_0
                            ∃ (w_22_0 : mxxMat1),
                              ∃ (w_23_0 : mxxMat1),
                                ∃ (w_24_0 : mxxMat0),
                                  ∃ (w_24_concat_1 : mxxMat2),
                                    ∃ (w_27_0 : mxxMat2),
                                      ∃ (w_28_0 : mxxMat4),
                                        ∃ (w_29_0 : mxxMat4),
                                          let w_30_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_29_0
                                          ∃ (w_31_0 : mxxMat1),
                                            let w_33_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_32_0 w_19_0
                                            let w_34_0 : mxxMat1 := MxxRuntime.matrixAdd w_31_0 w_33_0
                                            ∃ (w_35_0 : mxxMat1),
                                              ∃ (w_36_0 : mxxMat1),
                                                ∃ (w_37_0 : mxxMat0),
                                                  ∃ (w_37_concat_1 : mxxMat2),
                                                    ∃ (w_38_0 : mxxMat3),
                                                      ∃ (w_40_0 : mxxMat1),
                                                        ∃ (w_42_0 : mxxMat2),
                                                          ∃ (w_43_0 : mxxMat4),
                                                            ∃ (w_44_0 : mxxMat4),
                                                              let w_45_0 : mxxMat0 := MxxRuntime.matrixMul w_38_0 w_44_0
                                                              ∃ (w_46_0 : mxxMat1),
                                                                ∃ (w_47_0 : mxxMat1),
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_38_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat1),
                                                                              ∃ (w_56_0 : mxxMat1),
                                                                                ∃ (w_57_0 : mxxMat1),
                                                                                  parallel_generatedRoot_1076.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_12_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 ∧
    parallel_generatedRoot_1076.constraints_1 backend w_25_0 w_26_0 w_16_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_concat_1 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 ∧
    parallel_generatedRoot_1076.constraints_2 backend hashModel i_0 w_39_0 w_41_0 w_48_0 w_24_0 w_30_0 w_34_0 w_35_0 w_36_0 w_37_0 w_37_concat_1 w_38_0 w_40_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 ∧
    parallel_generatedRoot_1076.constraints_3 w_45_0 w_46_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 outputs

abbrev parallel_parallel_generatedRoot_1077_6.constraints_0 (w_0_0 : Fin 14400 → mxxMat1) (w_3_0 : Fin 14400 → mxxMat1) (w_5_0 : Fin 14400 → mxxMat1) (w_7_0 : Fin 14400 → mxxMat1) (w_9_0 : Fin 14400 → mxxMat1) (w_11_0 : Fin 14400 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_10_0 : mxxMat1) (w_12_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_1077_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat1 := inputs.1
  let w_3_0 : Fin 14400 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 14400 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.1
  let w_11_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          ∃ (w_10_0 : mxxMat1),
            ∃ (w_12_0 : mxxMat1),
              parallel_parallel_generatedRoot_1077_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_1077_25.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_18_0 : Fin 2 → mxxMat1) (w_22_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_1077_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_parallel_generatedRoot_1077_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1077.Witness where
  w_6_0 : Fin 2 → mxxMat1
  w_6_1 : Fin 2 → mxxMat1
  w_6_2 : Fin 2 → mxxMat1
  w_6_3 : Fin 2 → mxxMat1
  w_6_4 : Fin 2 → mxxMat1
  w_6_5 : Fin 2 → mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat1
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_16_0 : mxxMat1
  w_17_0 : mxxMat1
  w_19_0 : mxxMat1
  w_20_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat1
  w_25_0 : mxxMat1
  w_25_1 : mxxMat1
  w_25_2 : mxxMat1
  w_25_3 : mxxMat1
  w_25_4 : mxxMat1
  w_25_5 : mxxMat1

abbrev parallel_generatedRoot_1077.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 14400 → mxxMat1) (w_1_0 : Fin 14400 → mxxMat1) (w_2_0 : Fin 14400 → mxxMat1) (w_3_0 : Fin 14400 → mxxMat1) (w_4_0 : Fin 14400 → mxxMat1) (w_5_0 : Fin 14400 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_6_1 : Fin 2 → mxxMat1) (w_6_2 : Fin 2 → mxxMat1) (w_6_3 : Fin 2 → mxxMat1) (w_6_4 : Fin 2 → mxxMat1) (w_6_5 : Fin 2 → mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_25_1 : mxxMat1) (w_25_2 : mxxMat1) (w_25_3 : mxxMat1) (w_25_4 : mxxMat1) (w_25_5 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_1077_6 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_1077_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_1077.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_1077.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat1 := inputs.1
  let w_1_0 : Fin 14400 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 14400 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.1
  let w_5_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_6_4 := witness.w_6_4
  let w_6_5 := witness.w_6_5
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat1 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_25_2 := witness.w_25_2
  let w_25_3 := witness.w_25_3
  let w_25_4 := witness.w_25_4
  let w_25_5 := witness.w_25_5
  parallel_generatedRoot_1077.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_1077 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_1077.Witness,
    parallel_generatedRoot_1077.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_1080.constraints_4 (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) (w_61_0 : mxxMat1) (w_62_0 : mxxMat1) (w_63_0 : mxxMat1) (w_64_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_1080.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_46_0 : ByteArray) (w_55_0 : mxxMat1) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_56_0 : mxxMat2) (w_57_0 : mxxMat4) (w_58_0 : mxxMat4) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) (w_61_0 : mxxMat1) (w_62_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1080.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_38_0 : Fin 7200 → mxxMat1) (w_46_0 : ByteArray) (w_48_0 : mxxMat1) (w_12_0 : Int) (w_28_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat1) (w_39_0 : mxxMat1) (w_41_0 : mxxMat1) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat0) (w_44_concat_1 : mxxMat2) (w_45_0 : mxxMat3) (w_47_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_1080.constraints_1 (backend : MxxRuntime.BackendContext) (w_21_0 : Fin 7200 → mxxMat1) (w_29_0 : Fin 7200 → mxxMat1) (w_31_0 : Fin 7200 → mxxMat1) (w_12_0 : Int) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_30_0 : mxxMat1) (w_32_0 : mxxMat1) (w_33_0 : mxxMat2) (w_34_0 : mxxMat4) (w_35_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_1080.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_11_0 : Fin 7200 → mxxMat1) (w_14_0 : Fin 7200 → mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat0) (w_5_concat_1 : mxxMat2) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_9_concat_1 : mxxMat2) (w_10_0 : mxxMat3) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat2) (w_17_0 : mxxMat4) (w_18_0 : mxxMat4) : Prop :=
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

def parallel_generatedRoot_1080 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat1 × ByteArray × mxxMat1 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × mxxMat5 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × ByteArray × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat1 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.1
  let w_11_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.1
  let w_14_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_21_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_23_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_5_0 : mxxMat0),
        ∃ (w_5_concat_1 : mxxMat2),
          ∃ (w_7_0 : mxxMat1),
            ∃ (w_8_0 : mxxMat1),
              ∃ (w_9_0 : mxxMat0),
                ∃ (w_9_concat_1 : mxxMat2),
                  ∃ (w_10_0 : mxxMat3),
                    let w_12_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (2))
                    ∃ (w_13_0 : mxxMat1),
                      ∃ (w_15_0 : mxxMat1),
                        ∃ (w_16_0 : mxxMat2),
                          ∃ (w_17_0 : mxxMat4),
                            ∃ (w_18_0 : mxxMat4),
                              let w_19_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_18_0
                              ∃ (w_20_0 : mxxMat1),
                                ∃ (w_22_0 : mxxMat1),
                                  let w_24_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_22_0 w_23_0
                                  let w_25_0 : mxxMat1 := MxxRuntime.matrixAdd w_20_0 w_24_0
                                  ∃ (w_26_0 : mxxMat1),
                                    ∃ (w_27_0 : mxxMat1),
                                      ∃ (w_28_0 : mxxMat0),
                                        ∃ (w_28_concat_1 : mxxMat2),
                                          ∃ (w_30_0 : mxxMat1),
                                            ∃ (w_32_0 : mxxMat1),
                                              ∃ (w_33_0 : mxxMat2),
                                                ∃ (w_34_0 : mxxMat4),
                                                  ∃ (w_35_0 : mxxMat4),
                                                    let w_36_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_35_0
                                                    ∃ (w_37_0 : mxxMat1),
                                                      ∃ (w_39_0 : mxxMat1),
                                                        let w_40_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_39_0 w_23_0
                                                        let w_41_0 : mxxMat1 := MxxRuntime.matrixAdd w_37_0 w_40_0
                                                        ∃ (w_42_0 : mxxMat1),
                                                          ∃ (w_43_0 : mxxMat1),
                                                            ∃ (w_44_0 : mxxMat0),
                                                              ∃ (w_44_concat_1 : mxxMat2),
                                                                ∃ (w_45_0 : mxxMat3),
                                                                  ∃ (w_47_0 : mxxMat1),
                                                                    ∃ (w_49_0 : mxxMat2),
                                                                      ∃ (w_50_0 : mxxMat4),
                                                                        ∃ (w_51_0 : mxxMat4),
                                                                          let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_45_0 w_51_0
                                                                          ∃ (w_53_0 : mxxMat1),
                                                                            ∃ (w_54_0 : mxxMat1),
                                                                              ∃ (w_56_0 : mxxMat2),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  ∃ (w_58_0 : mxxMat4),
                                                                                    let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_45_0 w_58_0
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      ∃ (w_61_0 : mxxMat1),
                                                                                        ∃ (w_62_0 : mxxMat1),
                                                                                          ∃ (w_63_0 : mxxMat1),
                                                                                            ∃ (w_64_0 : mxxMat1),
                                                                                              parallel_generatedRoot_1080.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_14_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_1080.constraints_1 backend w_21_0 w_29_0 w_31_0 w_12_0 w_19_0 w_20_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_concat_1 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 ∧
    parallel_generatedRoot_1080.constraints_2 backend hashModel i_0 w_38_0 w_46_0 w_48_0 w_12_0 w_28_0 w_36_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_concat_1 w_45_0 w_47_0 w_49_0 w_50_0 w_51_0 ∧
    parallel_generatedRoot_1080.constraints_3 backend hashModel i_0 w_46_0 w_55_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 ∧
    parallel_generatedRoot_1080.constraints_4 w_52_0 w_53_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_parallel_generatedRoot_1081_6.constraints_0 (w_0_0 : Fin 14400 → mxxMat1) (w_3_0 : Fin 14400 → mxxMat1) (w_5_0 : Fin 14400 → mxxMat1) (w_7_0 : Fin 14400 → mxxMat1) (w_9_0 : Fin 14400 → mxxMat1) (w_11_0 : Fin 14400 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_10_0 : mxxMat1) (w_12_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_1081_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat1 := inputs.1
  let w_3_0 : Fin 14400 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 14400 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.1
  let w_11_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          ∃ (w_10_0 : mxxMat1),
            ∃ (w_12_0 : mxxMat1),
              parallel_parallel_generatedRoot_1081_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_1081_25.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_18_0 : Fin 2 → mxxMat1) (w_22_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_1081_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_parallel_generatedRoot_1081_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1081.Witness where
  w_6_0 : Fin 2 → mxxMat1
  w_6_1 : Fin 2 → mxxMat1
  w_6_2 : Fin 2 → mxxMat1
  w_6_3 : Fin 2 → mxxMat1
  w_6_4 : Fin 2 → mxxMat1
  w_6_5 : Fin 2 → mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat1
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_16_0 : mxxMat1
  w_17_0 : mxxMat1
  w_19_0 : mxxMat1
  w_20_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat1
  w_25_0 : mxxMat1
  w_25_1 : mxxMat1
  w_25_2 : mxxMat1
  w_25_3 : mxxMat1
  w_25_4 : mxxMat1
  w_25_5 : mxxMat1

abbrev parallel_generatedRoot_1081.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 14400 → mxxMat1) (w_1_0 : Fin 14400 → mxxMat1) (w_2_0 : Fin 14400 → mxxMat1) (w_3_0 : Fin 14400 → mxxMat1) (w_4_0 : Fin 14400 → mxxMat1) (w_5_0 : Fin 14400 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_6_1 : Fin 2 → mxxMat1) (w_6_2 : Fin 2 → mxxMat1) (w_6_3 : Fin 2 → mxxMat1) (w_6_4 : Fin 2 → mxxMat1) (w_6_5 : Fin 2 → mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_25_1 : mxxMat1) (w_25_2 : mxxMat1) (w_25_3 : mxxMat1) (w_25_4 : mxxMat1) (w_25_5 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_1081_6 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_1081_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_1081.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_1081.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat1 := inputs.1
  let w_1_0 : Fin 14400 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 14400 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.1
  let w_5_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_6_4 := witness.w_6_4
  let w_6_5 := witness.w_6_5
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat1 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_25_2 := witness.w_25_2
  let w_25_3 := witness.w_25_3
  let w_25_4 := witness.w_25_4
  let w_25_5 := witness.w_25_5
  parallel_generatedRoot_1081.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_1081 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_1081.Witness,
    parallel_generatedRoot_1081.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_1082_18.constraints_4 (w_75_0 : Fin 8 → mxxMat1) (w_79_0 : Fin 8 → mxxMat1) (w_81_0 : Fin 8 → mxxMat1) (w_83_0 : Fin 8 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_1082_18.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_parallel_generatedRoot_1082_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_parallel_generatedRoot_1082_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_parallel_generatedRoot_1082_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1082_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_parallel_generatedRoot_1082_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_1082_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_1082_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_1082_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_1082_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1082.Witness where
  w_2_0 : mxxMat1
  w_4_0 : mxxMat1
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_18_0 : mxxMat1
  w_18_1 : mxxMat1
  w_18_2 : mxxMat1
  w_18_3 : mxxMat1
  w_18_4 : mxxMat1
  w_18_5 : mxxMat1
  w_18_6 : mxxMat1
  w_18_7 : mxxMat1
  w_18_8 : mxxMat1
  w_18_9 : mxxMat1
  w_19_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat0
  w_23_concat_1 : mxxMat2
  w_24_0 : mxxMat1
  w_25_0 : mxxMat1
  w_26_0 : mxxMat1
  w_27_0 : mxxMat0
  w_27_concat_1 : mxxMat2
  w_28_0 : mxxMat3
  w_29_0 : mxxMat1
  w_30_0 : mxxMat1
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_32_concat_1 : mxxMat2
  w_33_0 : mxxMat1
  w_34_0 : mxxMat1
  w_35_0 : mxxMat1
  w_36_0 : mxxMat0
  w_36_concat_1 : mxxMat2
  w_37_0 : mxxMat3
  w_38_0 : mxxMat2
  w_39_0 : mxxMat4
  w_40_0 : mxxMat4
  w_42_0 : mxxMat1
  w_43_0 : mxxMat1
  w_44_0 : mxxMat2
  w_45_0 : mxxMat4
  w_46_0 : mxxMat4
  w_48_0 : mxxMat1
  w_50_0 : mxxMat2
  w_51_0 : mxxMat4
  w_52_0 : mxxMat4
  w_54_0 : mxxMat1
  w_55_0 : mxxMat1
  w_56_0 : mxxMat2
  w_57_0 : mxxMat4
  w_58_0 : mxxMat4
  w_60_0 : mxxMat1
  w_62_0 : mxxMat1
  w_64_0 : mxxMat1
  w_66_0 : mxxMat1
  w_67_0 : mxxMat1
  w_70_0 : mxxMat1
  w_71_0 : mxxMat1

abbrev parallel_generatedRoot_1082.constraints_4 (w_49_0 : mxxMat1) (w_53_0 : mxxMat0) (w_59_0 : mxxMat0) (w_61_0 : mxxMat1) (w_63_0 : mxxMat1) (w_65_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_1082.constraints_3 (w_41_0 : mxxMat0) (w_47_0 : mxxMat0) (w_59_0 : mxxMat0) (w_62_0 : mxxMat1) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1082.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat1) (w_18_9 : mxxMat1) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_50_0 : mxxMat2) (w_51_0 : mxxMat4) (w_52_0 : mxxMat4) (w_53_0 : mxxMat0) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_56_0 : mxxMat2) (w_57_0 : mxxMat4) (w_58_0 : mxxMat4) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1082.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_6 : mxxMat1) (w_18_8 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat1) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_36_0 : mxxMat0) (w_36_concat_1 : mxxMat2) (w_37_0 : mxxMat3) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat2) (w_45_0 : mxxMat4) (w_46_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_1082.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat1) (w_3_0 : Fin 8 → mxxMat1) (w_5_0 : Fin 8 → mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_0_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_18_0 : mxxMat1) (w_18_1 : mxxMat1) (w_18_2 : mxxMat1) (w_18_3 : mxxMat1) (w_18_4 : mxxMat1) (w_18_5 : mxxMat1) (w_18_6 : mxxMat1) (w_18_7 : mxxMat1) (w_18_8 : mxxMat1) (w_18_9 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : Int) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat0) (w_23_concat_1 : mxxMat2) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat0) (w_27_concat_1 : mxxMat2) (w_28_0 : mxxMat3) (w_29_0 : mxxMat1) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_32_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_1082_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
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

abbrev parallel_generatedRoot_1082.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_1082.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 8 → mxxMat1 := inputs.1
  let w_3_0 : Fin 8 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 8 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 8 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.1
  let w_10_0 : ByteArray := inputs.2.2.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
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
  let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_28_0 w_58_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_3 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_4 w_62_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_5 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 : mxxMat1 := MxxRuntime.matrixAdd w_70_0 w_71_0
  let w_73_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_1 w_72_0
  parallel_generatedRoot_1082.constraints_0 backend hashModel params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_1082.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_1082.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_1082.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_1082.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_1082 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_1082.Witness,
    parallel_generatedRoot_1082.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_1083.constraints_0 (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_14_0 : mxxMat1) (w_17_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  outputs = (w_2_0, (w_5_0, (w_8_0, (w_11_0, (w_14_0, (w_17_0, ()))))))

def parallel_generatedRoot_1083 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_3_0 : mxxMat1 := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_2_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_0_0 w_1_0
  let w_4_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  let w_7_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_8_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_6_0 w_7_0
  let w_10_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_11_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_9_0 w_10_0
  let w_13_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_14_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_12_0 w_13_0
  let w_16_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_17_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_15_0 w_16_0
  parallel_generatedRoot_1083.constraints_0 w_2_0 w_5_0 w_8_0 w_11_0 w_14_0 w_17_0 outputs

abbrev sequential_generatedRoot_1102.constraints_0 (w_1_0 : Fin 7200 → mxxMat1) (w_6_0 : Fin 7200 → mxxMat1) (w_10_0 : Fin 7200 → mxxMat1) (w_14_0 : Fin 7200 → mxxMat1) (w_18_0 : Fin 7200 → mxxMat1) (w_22_0 : Fin 7200 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_generatedRoot_1102 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_generatedRoot_1102.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev sequential_sequential_generatedRoot_1114_28.constraints_0 (w_1_0 : Fin 7200 → mxxMat1) (w_6_0 : Fin 7200 → mxxMat1) (w_10_0 : Fin 7200 → mxxMat1) (w_14_0 : Fin 7200 → mxxMat1) (w_18_0 : Fin 7200 → mxxMat1) (w_22_0 : Fin 7200 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_sequential_generatedRoot_1114_28 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := ((10 + (8 * (Int.ofNat i_0))) + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_sequential_generatedRoot_1114_28.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_1114.Witness where
  w_4_0 : mxxMat1
  w_6_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat1
  w_25_0 : mxxMat1
  w_26_0 : mxxMat1
  w_28_0 : mxxMat1
  w_28_1 : mxxMat1
  w_28_2 : mxxMat1
  w_28_3 : mxxMat1
  w_28_4 : mxxMat1
  w_28_5 : mxxMat1
  w_29_0 : mxxMat6
  w_33_0 : mxxMat6
  w_37_0 : mxxMat6
  w_41_0 : mxxMat6
  w_45_0 : mxxMat6
  w_49_0 : mxxMat6

abbrev sequential_generatedRoot_1114.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_2_0 : Fin 7200 → mxxMat1) (w_8_0 : Fin 7200 → mxxMat1) (w_12_0 : Fin 7200 → mxxMat1) (w_16_0 : Fin 7200 → mxxMat1) (w_20_0 : Fin 7200 → mxxMat1) (w_24_0 : Fin 7200 → mxxMat1) (w_1_0 : mxxMat7) (w_3_0 : Int) (w_5_0 : Int) (w_7_0 : mxxMat1) (w_11_0 : mxxMat1) (w_15_0 : mxxMat1) (w_19_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (w_28_1 : mxxMat1) (w_28_2 : mxxMat1) (w_28_3 : mxxMat1) (w_28_4 : mxxMat1) (w_28_5 : mxxMat1) (w_29_0 : mxxMat6) (w_30_0 : mxxMat6) (w_32_0 : mxxMat7) (w_33_0 : mxxMat6) (w_34_0 : mxxMat6) (w_36_0 : mxxMat7) (w_37_0 : mxxMat6) (w_38_0 : mxxMat6) (w_40_0 : mxxMat7) (w_41_0 : mxxMat6) (w_42_0 : mxxMat6) (w_44_0 : mxxMat7) (w_45_0 : mxxMat6) (w_46_0 : mxxMat6) (w_48_0 : mxxMat7) (w_49_0 : mxxMat6) (w_50_0 : mxxMat6) (outputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × Unit) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_3_0 w_25_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_5_0 w_26_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_sequential_generatedRoot_1114_28 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_2_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ())))))))))))) next) (Int.toNat (6)) (w_7_0, (w_11_0, (w_15_0, (w_19_0, (w_23_0, (w_27_0, ())))))) (w_28_0, (w_28_1, (w_28_2, (w_28_3, (w_28_4, (w_28_5, ())))))) ∧
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

abbrev sequential_generatedRoot_1114.constraints_0 (i_0 : Nat) (w_2_0 : Fin 7200 → mxxMat1) (w_8_0 : Fin 7200 → mxxMat1) (w_12_0 : Fin 7200 → mxxMat1) (w_16_0 : Fin 7200 → mxxMat1) (w_20_0 : Fin 7200 → mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_1114.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit) (outputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × Unit) (witness : sequential_generatedRoot_1114.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat6 := inputs.1
  let w_31_0 : mxxMat6 := inputs.2.1
  let w_35_0 : mxxMat6 := inputs.2.2.1
  let w_39_0 : mxxMat6 := inputs.2.2.2.1
  let w_43_0 : mxxMat6 := inputs.2.2.2.2.1
  let w_47_0 : mxxMat6 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_8_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_12_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_3_0 : Int := (8 + (8 * (Int.ofNat i_0)))
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := (9 + (8 * (Int.ofNat i_0)))
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_6_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_10_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_18_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_22_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_1 := witness.w_28_1
  let w_28_2 := witness.w_28_2
  let w_28_3 := witness.w_28_3
  let w_28_4 := witness.w_28_4
  let w_28_5 := witness.w_28_5
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat6 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat6 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_37_0 := witness.w_37_0
  let w_38_0 : mxxMat6 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat6 := MxxRuntime.matrixAdd w_39_0 w_41_0
  let w_44_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat6 := MxxRuntime.matrixAdd w_43_0 w_45_0
  let w_48_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_49_0 := witness.w_49_0
  let w_50_0 : mxxMat6 := MxxRuntime.matrixAdd w_47_0 w_49_0
  sequential_generatedRoot_1114.constraints_0 i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_20_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_17_0 w_18_0 w_21_0 w_22_0 ∧
    sequential_generatedRoot_1114.constraints_1 backend hashModel params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_20_0 w_24_0 w_1_0 w_3_0 w_5_0 w_7_0 w_11_0 w_15_0 w_19_0 w_23_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_1 w_28_2 w_28_3 w_28_4 w_28_5 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_36_0 w_37_0 w_38_0 w_40_0 w_41_0 w_42_0 w_44_0 w_45_0 w_46_0 w_48_0 w_49_0 w_50_0 outputs

def sequential_generatedRoot_1114 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit) (outputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_1114.Witness,
    sequential_generatedRoot_1114.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_1130.constraints_3 (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_56_0 : mxxMat1) (w_57_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_1130.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_39_0 : ByteArray) (w_41_0 : mxxMat1) (w_48_0 : mxxMat1) (w_24_0 : mxxMat0) (w_30_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_37_concat_1 : mxxMat2) (w_38_0 : mxxMat3) (w_40_0 : mxxMat1) (w_42_0 : mxxMat2) (w_43_0 : mxxMat4) (w_44_0 : mxxMat4) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1130.constraints_1 (backend : MxxRuntime.BackendContext) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_16_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat0) (w_24_concat_1 : mxxMat2) (w_27_0 : mxxMat2) (w_28_0 : mxxMat4) (w_29_0 : mxxMat4) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_35_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1130.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat0) (w_5_concat_1 : mxxMat2) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_9_concat_1 : mxxMat2) (w_10_0 : mxxMat3) (w_13_0 : mxxMat2) (w_14_0 : mxxMat4) (w_15_0 : mxxMat4) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) : Prop :=
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

def parallel_generatedRoot_1130 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat1 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.1
  let w_11_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_18_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_19_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_39_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_41_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_5_0 : mxxMat0),
        ∃ (w_5_concat_1 : mxxMat2),
          ∃ (w_7_0 : mxxMat1),
            ∃ (w_8_0 : mxxMat1),
              ∃ (w_9_0 : mxxMat0),
                ∃ (w_9_concat_1 : mxxMat2),
                  ∃ (w_10_0 : mxxMat3),
                    ∃ (w_13_0 : mxxMat2),
                      ∃ (w_14_0 : mxxMat4),
                        ∃ (w_15_0 : mxxMat4),
                          let w_16_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_15_0
                          ∃ (w_17_0 : mxxMat1),
                            let w_20_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_18_0 w_19_0
                            let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_20_0
                            ∃ (w_22_0 : mxxMat1),
                              ∃ (w_23_0 : mxxMat1),
                                ∃ (w_24_0 : mxxMat0),
                                  ∃ (w_24_concat_1 : mxxMat2),
                                    ∃ (w_27_0 : mxxMat2),
                                      ∃ (w_28_0 : mxxMat4),
                                        ∃ (w_29_0 : mxxMat4),
                                          let w_30_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_29_0
                                          ∃ (w_31_0 : mxxMat1),
                                            let w_33_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_32_0 w_19_0
                                            let w_34_0 : mxxMat1 := MxxRuntime.matrixAdd w_31_0 w_33_0
                                            ∃ (w_35_0 : mxxMat1),
                                              ∃ (w_36_0 : mxxMat1),
                                                ∃ (w_37_0 : mxxMat0),
                                                  ∃ (w_37_concat_1 : mxxMat2),
                                                    ∃ (w_38_0 : mxxMat3),
                                                      ∃ (w_40_0 : mxxMat1),
                                                        ∃ (w_42_0 : mxxMat2),
                                                          ∃ (w_43_0 : mxxMat4),
                                                            ∃ (w_44_0 : mxxMat4),
                                                              let w_45_0 : mxxMat0 := MxxRuntime.matrixMul w_38_0 w_44_0
                                                              ∃ (w_46_0 : mxxMat1),
                                                                ∃ (w_47_0 : mxxMat1),
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_38_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat1),
                                                                              ∃ (w_56_0 : mxxMat1),
                                                                                ∃ (w_57_0 : mxxMat1),
                                                                                  parallel_generatedRoot_1130.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_12_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 ∧
    parallel_generatedRoot_1130.constraints_1 backend w_25_0 w_26_0 w_16_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_concat_1 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 ∧
    parallel_generatedRoot_1130.constraints_2 backend hashModel i_0 w_39_0 w_41_0 w_48_0 w_24_0 w_30_0 w_34_0 w_35_0 w_36_0 w_37_0 w_37_concat_1 w_38_0 w_40_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 ∧
    parallel_generatedRoot_1130.constraints_3 w_45_0 w_46_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 outputs

abbrev parallel_parallel_generatedRoot_1131_6.constraints_0 (w_0_0 : Fin 72000 → mxxMat1) (w_3_0 : Fin 72000 → mxxMat1) (w_5_0 : Fin 72000 → mxxMat1) (w_7_0 : Fin 72000 → mxxMat1) (w_9_0 : Fin 72000 → mxxMat1) (w_11_0 : Fin 72000 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_10_0 : mxxMat1) (w_12_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_1131_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat1 := inputs.1
  let w_3_0 : Fin 72000 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 72000 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.1
  let w_11_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          ∃ (w_10_0 : mxxMat1),
            ∃ (w_12_0 : mxxMat1),
              parallel_parallel_generatedRoot_1131_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_1131_25.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_18_0 : Fin 2 → mxxMat1) (w_22_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_1131_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_parallel_generatedRoot_1131_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1131.Witness where
  w_6_0 : Fin 2 → mxxMat1
  w_6_1 : Fin 2 → mxxMat1
  w_6_2 : Fin 2 → mxxMat1
  w_6_3 : Fin 2 → mxxMat1
  w_6_4 : Fin 2 → mxxMat1
  w_6_5 : Fin 2 → mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat1
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_16_0 : mxxMat1
  w_17_0 : mxxMat1
  w_19_0 : mxxMat1
  w_20_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat1
  w_25_0 : mxxMat1
  w_25_1 : mxxMat1
  w_25_2 : mxxMat1
  w_25_3 : mxxMat1
  w_25_4 : mxxMat1
  w_25_5 : mxxMat1

abbrev parallel_generatedRoot_1131.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 72000 → mxxMat1) (w_1_0 : Fin 72000 → mxxMat1) (w_2_0 : Fin 72000 → mxxMat1) (w_3_0 : Fin 72000 → mxxMat1) (w_4_0 : Fin 72000 → mxxMat1) (w_5_0 : Fin 72000 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_6_1 : Fin 2 → mxxMat1) (w_6_2 : Fin 2 → mxxMat1) (w_6_3 : Fin 2 → mxxMat1) (w_6_4 : Fin 2 → mxxMat1) (w_6_5 : Fin 2 → mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_25_1 : mxxMat1) (w_25_2 : mxxMat1) (w_25_3 : mxxMat1) (w_25_4 : mxxMat1) (w_25_5 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_1131_6 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_1131_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_1131.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_1131.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat1 := inputs.1
  let w_1_0 : Fin 72000 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 72000 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.1
  let w_5_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_6_4 := witness.w_6_4
  let w_6_5 := witness.w_6_5
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat1 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_25_2 := witness.w_25_2
  let w_25_3 := witness.w_25_3
  let w_25_4 := witness.w_25_4
  let w_25_5 := witness.w_25_5
  parallel_generatedRoot_1131.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_1131 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_1131.Witness,
    parallel_generatedRoot_1131.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_1134.constraints_4 (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) (w_61_0 : mxxMat1) (w_62_0 : mxxMat1) (w_63_0 : mxxMat1) (w_64_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_1134.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_46_0 : ByteArray) (w_55_0 : mxxMat1) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_56_0 : mxxMat2) (w_57_0 : mxxMat4) (w_58_0 : mxxMat4) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) (w_61_0 : mxxMat1) (w_62_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1134.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_38_0 : Fin 36000 → mxxMat1) (w_46_0 : ByteArray) (w_48_0 : mxxMat1) (w_12_0 : Int) (w_28_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat1) (w_39_0 : mxxMat1) (w_41_0 : mxxMat1) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat0) (w_44_concat_1 : mxxMat2) (w_45_0 : mxxMat3) (w_47_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_1134.constraints_1 (backend : MxxRuntime.BackendContext) (w_21_0 : Fin 36000 → mxxMat1) (w_29_0 : Fin 36000 → mxxMat1) (w_31_0 : Fin 36000 → mxxMat1) (w_12_0 : Int) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_30_0 : mxxMat1) (w_32_0 : mxxMat1) (w_33_0 : mxxMat2) (w_34_0 : mxxMat4) (w_35_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_1134.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Int) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_11_0 : Fin 36000 → mxxMat1) (w_14_0 : Fin 36000 → mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat0) (w_5_concat_1 : mxxMat2) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_9_concat_1 : mxxMat2) (w_10_0 : mxxMat3) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat2) (w_17_0 : mxxMat4) (w_18_0 : mxxMat4) : Prop :=
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

def parallel_generatedRoot_1134 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat1 × ByteArray × mxxMat1 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × mxxMat5 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × ByteArray × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Int := inputs.1
  let w_0_0 : mxxMat1 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.1
  let w_11_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.1
  let w_14_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_21_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_23_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_31_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_5_0 : mxxMat0),
        ∃ (w_5_concat_1 : mxxMat2),
          ∃ (w_7_0 : mxxMat1),
            ∃ (w_8_0 : mxxMat1),
              ∃ (w_9_0 : mxxMat0),
                ∃ (w_9_concat_1 : mxxMat2),
                  ∃ (w_10_0 : mxxMat3),
                    let w_12_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (2))
                    ∃ (w_13_0 : mxxMat1),
                      ∃ (w_15_0 : mxxMat1),
                        ∃ (w_16_0 : mxxMat2),
                          ∃ (w_17_0 : mxxMat4),
                            ∃ (w_18_0 : mxxMat4),
                              let w_19_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_18_0
                              ∃ (w_20_0 : mxxMat1),
                                ∃ (w_22_0 : mxxMat1),
                                  let w_24_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_22_0 w_23_0
                                  let w_25_0 : mxxMat1 := MxxRuntime.matrixAdd w_20_0 w_24_0
                                  ∃ (w_26_0 : mxxMat1),
                                    ∃ (w_27_0 : mxxMat1),
                                      ∃ (w_28_0 : mxxMat0),
                                        ∃ (w_28_concat_1 : mxxMat2),
                                          ∃ (w_30_0 : mxxMat1),
                                            ∃ (w_32_0 : mxxMat1),
                                              ∃ (w_33_0 : mxxMat2),
                                                ∃ (w_34_0 : mxxMat4),
                                                  ∃ (w_35_0 : mxxMat4),
                                                    let w_36_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_35_0
                                                    ∃ (w_37_0 : mxxMat1),
                                                      ∃ (w_39_0 : mxxMat1),
                                                        let w_40_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_39_0 w_23_0
                                                        let w_41_0 : mxxMat1 := MxxRuntime.matrixAdd w_37_0 w_40_0
                                                        ∃ (w_42_0 : mxxMat1),
                                                          ∃ (w_43_0 : mxxMat1),
                                                            ∃ (w_44_0 : mxxMat0),
                                                              ∃ (w_44_concat_1 : mxxMat2),
                                                                ∃ (w_45_0 : mxxMat3),
                                                                  ∃ (w_47_0 : mxxMat1),
                                                                    ∃ (w_49_0 : mxxMat2),
                                                                      ∃ (w_50_0 : mxxMat4),
                                                                        ∃ (w_51_0 : mxxMat4),
                                                                          let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_45_0 w_51_0
                                                                          ∃ (w_53_0 : mxxMat1),
                                                                            ∃ (w_54_0 : mxxMat1),
                                                                              ∃ (w_56_0 : mxxMat2),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  ∃ (w_58_0 : mxxMat4),
                                                                                    let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_45_0 w_58_0
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      ∃ (w_61_0 : mxxMat1),
                                                                                        ∃ (w_62_0 : mxxMat1),
                                                                                          ∃ (w_63_0 : mxxMat1),
                                                                                            ∃ (w_64_0 : mxxMat1),
                                                                                              parallel_generatedRoot_1134.constraints_0 backend hashModel w_2_0 w_0_0 w_1_0 w_6_0 w_11_0 w_14_0 w_3_0 w_4_0 w_5_0 w_5_concat_1 w_7_0 w_8_0 w_9_0 w_9_concat_1 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_1134.constraints_1 backend w_21_0 w_29_0 w_31_0 w_12_0 w_19_0 w_20_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_concat_1 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 ∧
    parallel_generatedRoot_1134.constraints_2 backend hashModel i_0 w_38_0 w_46_0 w_48_0 w_12_0 w_28_0 w_36_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_concat_1 w_45_0 w_47_0 w_49_0 w_50_0 w_51_0 ∧
    parallel_generatedRoot_1134.constraints_3 backend hashModel i_0 w_46_0 w_55_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 ∧
    parallel_generatedRoot_1134.constraints_4 w_52_0 w_53_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_parallel_generatedRoot_1135_6.constraints_0 (w_0_0 : Fin 72000 → mxxMat1) (w_3_0 : Fin 72000 → mxxMat1) (w_5_0 : Fin 72000 → mxxMat1) (w_7_0 : Fin 72000 → mxxMat1) (w_9_0 : Fin 72000 → mxxMat1) (w_11_0 : Fin 72000 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_10_0 : mxxMat1) (w_12_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_1135_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat1 := inputs.1
  let w_3_0 : Fin 72000 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 72000 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.1
  let w_11_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          ∃ (w_10_0 : mxxMat1),
            ∃ (w_12_0 : mxxMat1),
              parallel_parallel_generatedRoot_1135_6.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 outputs

abbrev sequential_parallel_generatedRoot_1135_25.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_18_0 : Fin 2 → mxxMat1) (w_22_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_1135_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_parallel_generatedRoot_1135_25.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1135.Witness where
  w_6_0 : Fin 2 → mxxMat1
  w_6_1 : Fin 2 → mxxMat1
  w_6_2 : Fin 2 → mxxMat1
  w_6_3 : Fin 2 → mxxMat1
  w_6_4 : Fin 2 → mxxMat1
  w_6_5 : Fin 2 → mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat1
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_16_0 : mxxMat1
  w_17_0 : mxxMat1
  w_19_0 : mxxMat1
  w_20_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat1
  w_25_0 : mxxMat1
  w_25_1 : mxxMat1
  w_25_2 : mxxMat1
  w_25_3 : mxxMat1
  w_25_4 : mxxMat1
  w_25_5 : mxxMat1

abbrev parallel_generatedRoot_1135.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 72000 → mxxMat1) (w_1_0 : Fin 72000 → mxxMat1) (w_2_0 : Fin 72000 → mxxMat1) (w_3_0 : Fin 72000 → mxxMat1) (w_4_0 : Fin 72000 → mxxMat1) (w_5_0 : Fin 72000 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_6_1 : Fin 2 → mxxMat1) (w_6_2 : Fin 2 → mxxMat1) (w_6_3 : Fin 2 → mxxMat1) (w_6_4 : Fin 2 → mxxMat1) (w_6_5 : Fin 2 → mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_25_1 : mxxMat1) (w_25_2 : mxxMat1) (w_25_3 : mxxMat1) (w_25_4 : mxxMat1) (w_25_5 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_1135_6 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, ())))))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ((w_6_4 i), ((w_6_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_1135_25 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_6_4, (w_6_5, ())))))))))))) next) (Int.toNat (0)) (w_9_0, (w_12_0, (w_15_0, (w_18_0, (w_21_0, (w_24_0, ())))))) (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ())))))) ∧
  outputs = (w_25_0, (w_25_1, (w_25_2, (w_25_3, (w_25_4, (w_25_5, ()))))))

abbrev parallel_generatedRoot_1135.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_1135.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat1 := inputs.1
  let w_1_0 : Fin 72000 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 72000 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.1
  let w_5_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_6_4 := witness.w_6_4
  let w_6_5 := witness.w_6_5
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat1 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_25_1 := witness.w_25_1
  let w_25_2 := witness.w_25_2
  let w_25_3 := witness.w_25_3
  let w_25_4 := witness.w_25_4
  let w_25_5 := witness.w_25_5
  parallel_generatedRoot_1135.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs

def parallel_generatedRoot_1135 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_1135.Witness,
    parallel_generatedRoot_1135.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_1136_18.constraints_4 (w_75_0 : Fin 8 → mxxMat1) (w_79_0 : Fin 8 → mxxMat1) (w_81_0 : Fin 8 → mxxMat1) (w_83_0 : Fin 8 → mxxMat1) (w_41_0 : mxxMat0) (w_45_0 : mxxMat1) (w_58_0 : mxxMat0) (w_62_0 : mxxMat1) (w_65_0 : mxxMat1) (w_68_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : Int) (w_78_0 : mxxMat1) (w_80_0 : mxxMat1) (w_82_0 : mxxMat1) (w_84_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_1136_18.constraints_3 (w_41_0 : mxxMat0) (w_52_0 : mxxMat0) (w_58_0 : mxxMat0) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_64_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev sequential_parallel_generatedRoot_1136_18.constraints_2 (backend : MxxRuntime.BackendContext) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_35_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_49_0 : mxxMat2) (w_50_0 : mxxMat4) (w_51_0 : mxxMat4) (w_52_0 : mxxMat0) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat2) (w_56_0 : mxxMat4) (w_57_0 : mxxMat4) : Prop :=
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

abbrev sequential_parallel_generatedRoot_1136_18.constraints_1 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat1) (w_21_0 : mxxMat0) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat0) (w_28_concat_1 : mxxMat2) (w_29_0 : mxxMat3) (w_32_0 : mxxMat2) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) : Prop :=
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

abbrev sequential_parallel_generatedRoot_1136_18.constraints_0 (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : Fin 8 → mxxMat1) (w_4_0 : ByteArray) (w_9_0 : Fin 8 → mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat0) (w_8_concat_1 : mxxMat2) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat0) (w_13_concat_1 : mxxMat2) (w_14_0 : mxxMat3) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_21_concat_1 : mxxMat2) (w_23_0 : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1136_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_46_0 : mxxMat1 := inputs.2.1
  let w_63_0 : mxxMat1 := inputs.2.2.1
  let w_66_0 : mxxMat1 := inputs.2.2.2.1
  let w_69_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_47_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_31_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_48_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_75_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_79_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_83_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])
    ∃ (w_6_0 : mxxMat1),
      ∃ (w_7_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat0),
          ∃ (w_8_concat_1 : mxxMat2),
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat1),
                  ∃ (w_13_0 : mxxMat0),
                    ∃ (w_13_concat_1 : mxxMat2),
                      ∃ (w_14_0 : mxxMat3),
                        ∃ (w_16_0 : mxxMat1),
                          ∃ (w_18_0 : mxxMat1),
                            ∃ (w_20_0 : mxxMat1),
                              ∃ (w_21_0 : mxxMat0),
                                ∃ (w_21_concat_1 : mxxMat2),
                                  ∃ (w_23_0 : mxxMat1),
                                    ∃ (w_25_0 : mxxMat1),
                                      ∃ (w_27_0 : mxxMat1),
                                        ∃ (w_28_0 : mxxMat0),
                                          ∃ (w_28_concat_1 : mxxMat2),
                                            ∃ (w_29_0 : mxxMat3),
                                              ∃ (w_32_0 : mxxMat2),
                                                ∃ (w_33_0 : mxxMat4),
                                                  ∃ (w_34_0 : mxxMat4),
                                                    let w_35_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_34_0
                                                    ∃ (w_36_0 : mxxMat1),
                                                      ∃ (w_37_0 : mxxMat1),
                                                        ∃ (w_38_0 : mxxMat2),
                                                          ∃ (w_39_0 : mxxMat4),
                                                            ∃ (w_40_0 : mxxMat4),
                                                              let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_40_0
                                                              ∃ (w_42_0 : mxxMat1),
                                                                ∃ (w_43_0 : mxxMat1),
                                                                  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                                                  let w_45_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_44_0
                                                                  ∃ (w_49_0 : mxxMat2),
                                                                    ∃ (w_50_0 : mxxMat4),
                                                                      ∃ (w_51_0 : mxxMat4),
                                                                        let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_29_0 w_51_0
                                                                        ∃ (w_53_0 : mxxMat1),
                                                                          ∃ (w_54_0 : mxxMat1),
                                                                            ∃ (w_55_0 : mxxMat2),
                                                                              ∃ (w_56_0 : mxxMat4),
                                                                                ∃ (w_57_0 : mxxMat4),
                                                                                  let w_58_0 : mxxMat0 := MxxRuntime.matrixMul w_14_0 w_57_0
                                                                                  ∃ (w_59_0 : mxxMat1),
                                                                                    ∃ (w_60_0 : mxxMat1),
                                                                                      let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_59_0 w_60_0
                                                                                      let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_46_0 w_61_0
                                                                                      ∃ (w_64_0 : mxxMat1),
                                                                                        let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_63_0 w_64_0
                                                                                        ∃ (w_67_0 : mxxMat1),
                                                                                          let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
                                                                                          ∃ (w_70_0 : mxxMat1),
                                                                                            let w_71_0 : mxxMat1 := MxxRuntime.matrixAdd w_69_0 w_70_0
                                                                                            ∃ (w_73_0 : mxxMat1),
                                                                                              let w_74_0 : mxxMat1 := MxxRuntime.matrixAdd w_72_0 w_73_0
                                                                                              let w_76_0 : Int := 1
                                                                                              let w_77_0 : Int := (w_2_0 + w_76_0)
                                                                                              ∃ (w_78_0 : mxxMat1),
                                                                                                ∃ (w_80_0 : mxxMat1),
                                                                                                  ∃ (w_82_0 : mxxMat1),
                                                                                                    ∃ (w_84_0 : mxxMat1),
                                                                                                      sequential_parallel_generatedRoot_1136_18.constraints_0 hashModel i_1 w_1_0 w_4_0 w_9_0 w_15_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_10_0 w_11_0 w_12_0 w_13_0 w_13_concat_1 w_14_0 w_16_0 w_18_0 w_20_0 w_21_0 w_21_concat_1 w_23_0 ∧
    sequential_parallel_generatedRoot_1136_18.constraints_1 backend i_1 w_30_0 w_31_0 w_24_0 w_26_0 w_21_0 w_23_0 w_25_0 w_27_0 w_28_0 w_28_concat_1 w_29_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 ∧
    sequential_parallel_generatedRoot_1136_18.constraints_2 backend w_47_0 w_48_0 w_35_0 w_41_0 w_42_0 w_43_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 ∧
    sequential_parallel_generatedRoot_1136_18.constraints_3 w_41_0 w_52_0 w_58_0 w_59_0 w_60_0 w_64_0 w_67_0 ∧
    sequential_parallel_generatedRoot_1136_18.constraints_4 w_75_0 w_79_0 w_81_0 w_83_0 w_41_0 w_45_0 w_58_0 w_62_0 w_65_0 w_68_0 w_70_0 w_71_0 w_73_0 w_74_0 w_77_0 w_78_0 w_80_0 w_82_0 w_84_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1136.Witness where
  w_2_0 : mxxMat1
  w_4_0 : mxxMat1
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_18_0 : mxxMat1
  w_18_1 : mxxMat1
  w_18_2 : mxxMat1
  w_18_3 : mxxMat1
  w_18_4 : mxxMat1
  w_18_5 : mxxMat1
  w_18_6 : mxxMat1
  w_18_7 : mxxMat1
  w_18_8 : mxxMat1
  w_18_9 : mxxMat1
  w_19_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat1
  w_23_0 : mxxMat0
  w_23_concat_1 : mxxMat2
  w_24_0 : mxxMat1
  w_25_0 : mxxMat1
  w_26_0 : mxxMat1
  w_27_0 : mxxMat0
  w_27_concat_1 : mxxMat2
  w_28_0 : mxxMat3
  w_29_0 : mxxMat1
  w_30_0 : mxxMat1
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_32_concat_1 : mxxMat2
  w_33_0 : mxxMat1
  w_34_0 : mxxMat1
  w_35_0 : mxxMat1
  w_36_0 : mxxMat0
  w_36_concat_1 : mxxMat2
  w_37_0 : mxxMat3
  w_38_0 : mxxMat2
  w_39_0 : mxxMat4
  w_40_0 : mxxMat4
  w_42_0 : mxxMat1
  w_43_0 : mxxMat1
  w_44_0 : mxxMat2
  w_45_0 : mxxMat4
  w_46_0 : mxxMat4
  w_48_0 : mxxMat1
  w_50_0 : mxxMat2
  w_51_0 : mxxMat4
  w_52_0 : mxxMat4
  w_54_0 : mxxMat1
  w_55_0 : mxxMat1
  w_56_0 : mxxMat2
  w_57_0 : mxxMat4
  w_58_0 : mxxMat4
  w_60_0 : mxxMat1
  w_62_0 : mxxMat1
  w_64_0 : mxxMat1
  w_66_0 : mxxMat1
  w_67_0 : mxxMat1
  w_70_0 : mxxMat1
  w_71_0 : mxxMat1

abbrev parallel_generatedRoot_1136.constraints_4 (w_49_0 : mxxMat1) (w_53_0 : mxxMat0) (w_59_0 : mxxMat0) (w_61_0 : mxxMat1) (w_63_0 : mxxMat1) (w_65_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_73_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_1136.constraints_3 (w_41_0 : mxxMat0) (w_47_0 : mxxMat0) (w_59_0 : mxxMat0) (w_62_0 : mxxMat1) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1136.constraints_2 (backend : MxxRuntime.BackendContext) (w_18_7 : mxxMat1) (w_18_9 : mxxMat1) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_50_0 : mxxMat2) (w_51_0 : mxxMat4) (w_52_0 : mxxMat4) (w_53_0 : mxxMat0) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_56_0 : mxxMat2) (w_57_0 : mxxMat4) (w_58_0 : mxxMat4) (w_59_0 : mxxMat0) (w_60_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_1136.constraints_1 (backend : MxxRuntime.BackendContext) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_6 : mxxMat1) (w_18_8 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat1) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_36_0 : mxxMat0) (w_36_concat_1 : mxxMat2) (w_37_0 : mxxMat3) (w_38_0 : mxxMat2) (w_39_0 : mxxMat4) (w_40_0 : mxxMat4) (w_41_0 : mxxMat0) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat2) (w_45_0 : mxxMat4) (w_46_0 : mxxMat4) : Prop :=
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

abbrev parallel_generatedRoot_1136.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : Fin 8 → mxxMat1) (w_3_0 : Fin 8 → mxxMat1) (w_5_0 : Fin 8 → mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_10_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_0_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_18_0 : mxxMat1) (w_18_1 : mxxMat1) (w_18_2 : mxxMat1) (w_18_3 : mxxMat1) (w_18_4 : mxxMat1) (w_18_5 : mxxMat1) (w_18_6 : mxxMat1) (w_18_7 : mxxMat1) (w_18_8 : mxxMat1) (w_18_9 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : Int) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat0) (w_23_concat_1 : mxxMat2) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat0) (w_27_concat_1 : mxxMat2) (w_28_0 : mxxMat3) (w_29_0 : mxxMat1) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_32_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.familyGetStatic w_5_0 (0) w_6_0 ∧
  MxxRuntime.familyGetStatic w_7_0 (0) w_8_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_1136_18 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_1_0, (w_3_0, (w_5_0, (w_7_0, ()))))))))))))))))))))))) next) (Int.toNat (7)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_0_0, (w_2_0, (w_4_0, (w_6_0, (w_8_0, ())))))))))) (w_18_0, (w_18_1, (w_18_2, (w_18_3, (w_18_4, (w_18_5, (w_18_6, (w_18_7, (w_18_8, (w_18_9, ())))))))))) ∧
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

abbrev parallel_generatedRoot_1136.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_1136.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 8 → mxxMat1 := inputs.1
  let w_3_0 : Fin 8 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 8 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 8 → mxxMat1 := inputs.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.1
  let w_10_0 : ByteArray := inputs.2.2.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
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
  let w_41_0 : mxxMat0 := MxxRuntime.matrixMul w_37_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixMul w_28_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_2 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixMul w_37_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_28_0 w_58_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_3 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_4 w_62_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_5 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat1 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 : mxxMat1 := MxxRuntime.matrixAdd w_70_0 w_71_0
  let w_73_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_1 w_72_0
  parallel_generatedRoot_1136.constraints_0 backend hashModel params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 ∧
    parallel_generatedRoot_1136.constraints_1 backend w_15_0 w_16_0 w_17_0 w_18_6 w_18_8 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_36_concat_1 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 ∧
    parallel_generatedRoot_1136.constraints_2 backend w_18_7 w_18_9 w_47_0 w_48_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 ∧
    parallel_generatedRoot_1136.constraints_3 w_41_0 w_47_0 w_59_0 w_62_0 w_64_0 w_66_0 w_67_0 ∧
    parallel_generatedRoot_1136.constraints_4 w_49_0 w_53_0 w_59_0 w_61_0 w_63_0 w_65_0 w_69_0 w_70_0 w_71_0 w_73_0 outputs

def parallel_generatedRoot_1136 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_1136.Witness,
    parallel_generatedRoot_1136.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_1137.constraints_0 (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_14_0 : mxxMat1) (w_17_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  outputs = (w_2_0, (w_5_0, (w_8_0, (w_11_0, (w_14_0, (w_17_0, ()))))))

def parallel_generatedRoot_1137 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_3_0 : mxxMat1 := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_12_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_2_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_0_0 w_1_0
  let w_4_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  let w_7_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_8_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_6_0 w_7_0
  let w_10_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_11_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_9_0 w_10_0
  let w_13_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_14_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_12_0 w_13_0
  let w_16_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_17_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_15_0 w_16_0
  parallel_generatedRoot_1137.constraints_0 w_2_0 w_5_0 w_8_0 w_11_0 w_14_0 w_17_0 outputs

abbrev sequential_generatedRoot_1156.constraints_0 (w_1_0 : Fin 36000 → mxxMat1) (w_6_0 : Fin 36000 → mxxMat1) (w_10_0 : Fin 36000 → mxxMat1) (w_14_0 : Fin 36000 → mxxMat1) (w_18_0 : Fin 36000 → mxxMat1) (w_22_0 : Fin 36000 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_generatedRoot_1156 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_generatedRoot_1156.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

abbrev sequential_sequential_generatedRoot_1168_28.constraints_0 (w_1_0 : Fin 36000 → mxxMat1) (w_6_0 : Fin 36000 → mxxMat1) (w_10_0 : Fin 36000 → mxxMat1) (w_14_0 : Fin 36000 → mxxMat1) (w_18_0 : Fin 36000 → mxxMat1) (w_22_0 : Fin 36000 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_sequential_generatedRoot_1168_28 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := ((10 + (8 * (Int.ofNat i_0))) + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_19_0
            ∃ (w_23_0 : mxxMat1),
              let w_24_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_23_0
              sequential_sequential_generatedRoot_1168_28.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_18_0 w_22_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_1168.Witness where
  w_4_0 : mxxMat1
  w_6_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat1
  w_25_0 : mxxMat1
  w_26_0 : mxxMat1
  w_28_0 : mxxMat1
  w_28_1 : mxxMat1
  w_28_2 : mxxMat1
  w_28_3 : mxxMat1
  w_28_4 : mxxMat1
  w_28_5 : mxxMat1
  w_29_0 : mxxMat8
  w_33_0 : mxxMat8
  w_37_0 : mxxMat8
  w_41_0 : mxxMat8
  w_45_0 : mxxMat8
  w_49_0 : mxxMat8

abbrev sequential_generatedRoot_1168.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_2_0 : Fin 36000 → mxxMat1) (w_8_0 : Fin 36000 → mxxMat1) (w_12_0 : Fin 36000 → mxxMat1) (w_16_0 : Fin 36000 → mxxMat1) (w_20_0 : Fin 36000 → mxxMat1) (w_24_0 : Fin 36000 → mxxMat1) (w_1_0 : mxxMat9) (w_3_0 : Int) (w_5_0 : Int) (w_7_0 : mxxMat1) (w_11_0 : mxxMat1) (w_15_0 : mxxMat1) (w_19_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (w_28_1 : mxxMat1) (w_28_2 : mxxMat1) (w_28_3 : mxxMat1) (w_28_4 : mxxMat1) (w_28_5 : mxxMat1) (w_29_0 : mxxMat8) (w_30_0 : mxxMat8) (w_32_0 : mxxMat9) (w_33_0 : mxxMat8) (w_34_0 : mxxMat8) (w_36_0 : mxxMat9) (w_37_0 : mxxMat8) (w_38_0 : mxxMat8) (w_40_0 : mxxMat9) (w_41_0 : mxxMat8) (w_42_0 : mxxMat8) (w_44_0 : mxxMat9) (w_45_0 : mxxMat8) (w_46_0 : mxxMat8) (w_48_0 : mxxMat9) (w_49_0 : mxxMat8) (w_50_0 : mxxMat8) (outputs : mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × Unit) : Prop :=
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_3_0 w_25_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_24_0 w_5_0 w_26_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_sequential_generatedRoot_1168_28 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_2_0, (w_8_0, (w_12_0, (w_16_0, (w_20_0, (w_24_0, ())))))))))))) next) (Int.toNat (6)) (w_7_0, (w_11_0, (w_15_0, (w_19_0, (w_23_0, (w_27_0, ())))))) (w_28_0, (w_28_1, (w_28_2, (w_28_3, (w_28_4, (w_28_5, ())))))) ∧
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

abbrev sequential_generatedRoot_1168.constraints_0 (i_0 : Nat) (w_2_0 : Fin 36000 → mxxMat1) (w_8_0 : Fin 36000 → mxxMat1) (w_12_0 : Fin 36000 → mxxMat1) (w_16_0 : Fin 36000 → mxxMat1) (w_20_0 : Fin 36000 → mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_1168.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit) (outputs : mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × Unit) (witness : sequential_generatedRoot_1168.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat8 := inputs.1
  let w_31_0 : mxxMat8 := inputs.2.1
  let w_35_0 : mxxMat8 := inputs.2.2.1
  let w_39_0 : mxxMat8 := inputs.2.2.2.1
  let w_43_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_47_0 : mxxMat8 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_8_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_12_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat9 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat9)
  let w_3_0 : Int := (8 + (8 * (Int.ofNat i_0)))
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := (9 + (8 * (Int.ofNat i_0)))
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_6_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_10_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_18_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixAdd w_21_0 w_22_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_1 := witness.w_28_1
  let w_28_2 := witness.w_28_2
  let w_28_3 := witness.w_28_3
  let w_28_4 := witness.w_28_4
  let w_28_5 := witness.w_28_5
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat8 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 : mxxMat9 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat9)
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat8 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 : mxxMat9 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat9)
  let w_37_0 := witness.w_37_0
  let w_38_0 : mxxMat8 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 : mxxMat9 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat9)
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat8 := MxxRuntime.matrixAdd w_39_0 w_41_0
  let w_44_0 : mxxMat9 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat9)
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat8 := MxxRuntime.matrixAdd w_43_0 w_45_0
  let w_48_0 : mxxMat9 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat9)
  let w_49_0 := witness.w_49_0
  let w_50_0 : mxxMat8 := MxxRuntime.matrixAdd w_47_0 w_49_0
  sequential_generatedRoot_1168.constraints_0 i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_20_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_17_0 w_18_0 w_21_0 w_22_0 ∧
    sequential_generatedRoot_1168.constraints_1 backend hashModel params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_20_0 w_24_0 w_1_0 w_3_0 w_5_0 w_7_0 w_11_0 w_15_0 w_19_0 w_23_0 w_25_0 w_26_0 w_27_0 w_28_0 w_28_1 w_28_2 w_28_3 w_28_4 w_28_5 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_36_0 w_37_0 w_38_0 w_40_0 w_41_0 w_42_0 w_44_0 w_45_0 w_46_0 w_48_0 w_49_0 w_50_0 outputs

def sequential_generatedRoot_1168 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit) (outputs : mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_1168.Witness,
    sequential_generatedRoot_1168.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_1267_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1267_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1267_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_1267_93.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1267_93 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1267_93.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1267.Witness where
  w_2_0 : mxxMat12
  w_6_0 : mxxMat4
  w_7_0 : mxxMat4
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat13
  w_17_0 : mxxMat1
  w_20_0 : mxxMat14
  w_21_0 : mxxMat14
  w_26_0 : mxxMat10
  w_28_0 : mxxMat15
  w_28_concat_1 : mxxMat16
  w_28_concat_2 : mxxMat17
  w_28_concat_3 : mxxMat18
  w_28_concat_4 : mxxMat19
  w_28_concat_5 : mxxMat20
  w_28_concat_6 : mxxMat21
  w_28_concat_7 : mxxMat22
  w_28_concat_8 : mxxMat23
  w_28_concat_9 : mxxMat24
  w_28_concat_10 : mxxMat25
  w_28_concat_11 : mxxMat26
  w_28_concat_12 : mxxMat27
  w_28_concat_13 : mxxMat28
  w_28_concat_14 : mxxMat29
  w_28_concat_15 : mxxMat30
  w_28_concat_16 : mxxMat31
  w_28_concat_17 : mxxMat32
  w_28_concat_18 : mxxMat33
  w_28_concat_19 : mxxMat34
  w_28_concat_20 : mxxMat35
  w_28_concat_21 : mxxMat36
  w_28_concat_22 : mxxMat37
  w_28_concat_23 : mxxMat38
  w_29_0 : mxxMat15
  w_32_0 : mxxMat39
  w_33_0 : mxxMat39
  w_35_0 : mxxMat1
  w_37_0 : mxxMat11
  w_38_0 : mxxMat11
  w_40_0 : mxxMat1
  w_41_0 : mxxMat1
  w_44_0 : mxxMat10
  w_46_0 : mxxMat14
  w_46_concat_1 : mxxMat40
  w_46_concat_2 : mxxMat41
  w_46_concat_3 : mxxMat42
  w_46_concat_4 : mxxMat43
  w_46_concat_5 : mxxMat44
  w_46_concat_6 : mxxMat45
  w_46_concat_7 : mxxMat46
  w_46_concat_8 : mxxMat47
  w_46_concat_9 : mxxMat48
  w_46_concat_10 : mxxMat49
  w_46_concat_11 : mxxMat50
  w_46_concat_12 : mxxMat51
  w_46_concat_13 : mxxMat52
  w_46_concat_14 : mxxMat53
  w_46_concat_15 : mxxMat54
  w_46_concat_16 : mxxMat55
  w_46_concat_17 : mxxMat56
  w_46_concat_18 : mxxMat57
  w_46_concat_19 : mxxMat58
  w_46_concat_20 : mxxMat59
  w_46_concat_21 : mxxMat60
  w_46_concat_22 : mxxMat61
  w_46_concat_23 : mxxMat62
  w_47_0 : mxxMat14
  w_50_0 : mxxMat63
  w_51_0 : mxxMat63
  w_56_0 : mxxMat12
  w_59_0 : mxxMat4
  w_60_0 : mxxMat4
  w_63_0 : mxxMat1
  w_64_0 : mxxMat1
  w_65_0 : mxxMat13
  w_70_0 : mxxMat1
  w_73_0 : mxxMat14
  w_74_0 : mxxMat14
  w_79_0 : mxxMat10
  w_80_0 : mxxMat15
  w_80_concat_1 : mxxMat16
  w_80_concat_2 : mxxMat17
  w_80_concat_3 : mxxMat18
  w_80_concat_4 : mxxMat19
  w_80_concat_5 : mxxMat20
  w_80_concat_6 : mxxMat21
  w_80_concat_7 : mxxMat22
  w_80_concat_8 : mxxMat23
  w_80_concat_9 : mxxMat24
  w_80_concat_10 : mxxMat25
  w_80_concat_11 : mxxMat26
  w_80_concat_12 : mxxMat27
  w_80_concat_13 : mxxMat28
  w_80_concat_14 : mxxMat29
  w_80_concat_15 : mxxMat30
  w_80_concat_16 : mxxMat31
  w_80_concat_17 : mxxMat32
  w_80_concat_18 : mxxMat33
  w_80_concat_19 : mxxMat34
  w_80_concat_20 : mxxMat35
  w_80_concat_21 : mxxMat36
  w_80_concat_22 : mxxMat37
  w_80_concat_23 : mxxMat38
  w_81_0 : mxxMat15
  w_84_0 : mxxMat39
  w_85_0 : mxxMat39
  w_87_0 : mxxMat1
  w_89_0 : mxxMat11
  w_90_0 : mxxMat11
  w_92_0 : mxxMat1
  w_93_0 : mxxMat1
  w_96_0 : mxxMat10
  w_97_0 : mxxMat14
  w_97_concat_1 : mxxMat40
  w_97_concat_2 : mxxMat41
  w_97_concat_3 : mxxMat42
  w_97_concat_4 : mxxMat43
  w_97_concat_5 : mxxMat44
  w_97_concat_6 : mxxMat45
  w_97_concat_7 : mxxMat46
  w_97_concat_8 : mxxMat47
  w_97_concat_9 : mxxMat48
  w_97_concat_10 : mxxMat49
  w_97_concat_11 : mxxMat50
  w_97_concat_12 : mxxMat51
  w_97_concat_13 : mxxMat52
  w_97_concat_14 : mxxMat53
  w_97_concat_15 : mxxMat54
  w_97_concat_16 : mxxMat55
  w_97_concat_17 : mxxMat56
  w_97_concat_18 : mxxMat57
  w_97_concat_19 : mxxMat58
  w_97_concat_20 : mxxMat59
  w_97_concat_21 : mxxMat60
  w_97_concat_22 : mxxMat61
  w_97_concat_23 : mxxMat62
  w_98_0 : mxxMat14
  w_101_0 : mxxMat63
  w_102_0 : mxxMat63
  w_105_0 : mxxMat2

abbrev parallel_generatedRoot_1267.constraints_5 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_53_0 : mxxMat1) (w_97_0 : mxxMat14) (w_97_concat_5 : mxxMat44) (w_97_concat_6 : mxxMat45) (w_97_concat_7 : mxxMat46) (w_97_concat_8 : mxxMat47) (w_97_concat_9 : mxxMat48) (w_97_concat_10 : mxxMat49) (w_97_concat_11 : mxxMat50) (w_97_concat_12 : mxxMat51) (w_97_concat_13 : mxxMat52) (w_97_concat_14 : mxxMat53) (w_97_concat_15 : mxxMat54) (w_97_concat_16 : mxxMat55) (w_97_concat_17 : mxxMat56) (w_97_concat_18 : mxxMat57) (w_97_concat_19 : mxxMat58) (w_97_concat_20 : mxxMat59) (w_97_concat_21 : mxxMat60) (w_97_concat_22 : mxxMat61) (w_97_concat_23 : mxxMat62) (w_98_0 : mxxMat14) (w_100_0 : mxxMat14) (w_101_0 : mxxMat63) (w_102_0 : mxxMat63) (w_104_0 : mxxMat1) (w_105_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  MxxRuntime.concatDiagonal w_97_concat_5 w_45_0 w_97_concat_6 ∧
  MxxRuntime.concatDiagonal w_97_concat_6 w_45_0 w_97_concat_7 ∧
  MxxRuntime.concatDiagonal w_97_concat_7 w_45_0 w_97_concat_8 ∧
  MxxRuntime.concatDiagonal w_97_concat_8 w_45_0 w_97_concat_9 ∧
  MxxRuntime.concatDiagonal w_97_concat_9 w_45_0 w_97_concat_10 ∧
  MxxRuntime.concatDiagonal w_97_concat_10 w_45_0 w_97_concat_11 ∧
  MxxRuntime.concatDiagonal w_97_concat_11 w_45_0 w_97_concat_12 ∧
  MxxRuntime.concatDiagonal w_97_concat_12 w_45_0 w_97_concat_13 ∧
  MxxRuntime.concatDiagonal w_97_concat_13 w_45_0 w_97_concat_14 ∧
  MxxRuntime.concatDiagonal w_97_concat_14 w_45_0 w_97_concat_15 ∧
  MxxRuntime.concatDiagonal w_97_concat_15 w_45_0 w_97_concat_16 ∧
  MxxRuntime.concatDiagonal w_97_concat_16 w_45_0 w_97_concat_17 ∧
  MxxRuntime.concatDiagonal w_97_concat_17 w_45_0 w_97_concat_18 ∧
  MxxRuntime.concatDiagonal w_97_concat_18 w_45_0 w_97_concat_19 ∧
  MxxRuntime.concatDiagonal w_97_concat_19 w_45_0 w_97_concat_20 ∧
  MxxRuntime.concatDiagonal w_97_concat_20 w_45_0 w_97_concat_21 ∧
  MxxRuntime.concatDiagonal w_97_concat_21 w_45_0 w_97_concat_22 ∧
  MxxRuntime.concatDiagonal w_97_concat_22 w_45_0 w_97_concat_23 ∧
  MxxRuntime.concatDiagonal w_97_concat_23 w_45_0 w_97_0 ∧
  MxxRuntime.centeredRebaseRuns w_97_0 w_98_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_100_0 w_101_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_101_0 w_102_0 ∧
  MxxRuntime.concatRows w_53_0 w_104_0 w_105_0 ∧
  outputs = w_105_0

abbrev parallel_generatedRoot_1267.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_45_0 : mxxMat5) (w_95_0 : mxxMat8) (w_77_0 : mxxMat1) (w_80_0 : mxxMat15) (w_80_concat_19 : mxxMat34) (w_80_concat_20 : mxxMat35) (w_80_concat_21 : mxxMat36) (w_80_concat_22 : mxxMat37) (w_80_concat_23 : mxxMat38) (w_81_0 : mxxMat15) (w_83_0 : mxxMat15) (w_84_0 : mxxMat39) (w_85_0 : mxxMat39) (w_86_0 : mxxMat10) (w_87_0 : mxxMat1) (w_88_0 : mxxMat5) (w_89_0 : mxxMat11) (w_90_0 : mxxMat11) (w_91_0 : mxxMat5) (w_92_0 : mxxMat1) (w_93_0 : mxxMat1) (w_96_0 : mxxMat10) (w_97_concat_1 : mxxMat40) (w_97_concat_2 : mxxMat41) (w_97_concat_3 : mxxMat42) (w_97_concat_4 : mxxMat43) (w_97_concat_5 : mxxMat44) : Prop :=
  MxxRuntime.concatDiagonal w_80_concat_19 w_27_0 w_80_concat_20 ∧
  MxxRuntime.concatDiagonal w_80_concat_20 w_27_0 w_80_concat_21 ∧
  MxxRuntime.concatDiagonal w_80_concat_21 w_27_0 w_80_concat_22 ∧
  MxxRuntime.concatDiagonal w_80_concat_22 w_27_0 w_80_concat_23 ∧
  MxxRuntime.concatDiagonal w_80_concat_23 w_27_0 w_80_0 ∧
  MxxRuntime.centeredRebaseRuns w_80_0 w_81_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_83_0 w_84_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_84_0 w_85_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_86_0 0 1 0 25 w_87_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_88_0 w_89_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_89_0 w_90_0 ∧
  MxxRuntime.tensorRuns w_77_0 w_91_0 w_92_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1267_93 backend hashModel params i_0 i (current, (w_86_0, (w_90_0, ()))) next) (Int.toNat (24)) w_92_0 w_93_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (3125 * (Int.ofNat i_0)) ∧
  (3125 * (Int.ofNat i_0)) < (625 + (3125 * (Int.ofNat i_0))) ∧
  (625 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_95_0 0 1 (3125 * (Int.ofNat i_0)) (625 + (3125 * (Int.ofNat i_0))) w_96_0 ∧
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_97_concat_1 ∧
  MxxRuntime.concatDiagonal w_97_concat_1 w_45_0 w_97_concat_2 ∧
  MxxRuntime.concatDiagonal w_97_concat_2 w_45_0 w_97_concat_3 ∧
  MxxRuntime.concatDiagonal w_97_concat_3 w_45_0 w_97_concat_4 ∧
  MxxRuntime.concatDiagonal w_97_concat_4 w_45_0 w_97_concat_5

abbrev parallel_generatedRoot_1267.constraints_3 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_27_0 : mxxMat1) (w_78_0 : mxxMat6) (w_64_0 : mxxMat1) (w_65_0 : mxxMat13) (w_69_0 : mxxMat13) (w_70_0 : mxxMat1) (w_72_0 : mxxMat1) (w_73_0 : mxxMat14) (w_74_0 : mxxMat14) (w_79_0 : mxxMat10) (w_80_concat_1 : mxxMat16) (w_80_concat_2 : mxxMat17) (w_80_concat_3 : mxxMat18) (w_80_concat_4 : mxxMat19) (w_80_concat_5 : mxxMat20) (w_80_concat_6 : mxxMat21) (w_80_concat_7 : mxxMat22) (w_80_concat_8 : mxxMat23) (w_80_concat_9 : mxxMat24) (w_80_concat_10 : mxxMat25) (w_80_concat_11 : mxxMat26) (w_80_concat_12 : mxxMat27) (w_80_concat_13 : mxxMat28) (w_80_concat_14 : mxxMat29) (w_80_concat_15 : mxxMat30) (w_80_concat_16 : mxxMat31) (w_80_concat_17 : mxxMat32) (w_80_concat_18 : mxxMat33) (w_80_concat_19 : mxxMat34) : Prop :=
  MxxRuntime.modulusReduceRuns w_64_0 w_65_0 ∧
  MxxRuntime.centeredRebaseRuns w_69_0 w_70_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_72_0 w_73_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_73_0 w_74_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_79_0 ∧
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_80_concat_1 ∧
  MxxRuntime.concatDiagonal w_80_concat_1 w_27_0 w_80_concat_2 ∧
  MxxRuntime.concatDiagonal w_80_concat_2 w_27_0 w_80_concat_3 ∧
  MxxRuntime.concatDiagonal w_80_concat_3 w_27_0 w_80_concat_4 ∧
  MxxRuntime.concatDiagonal w_80_concat_4 w_27_0 w_80_concat_5 ∧
  MxxRuntime.concatDiagonal w_80_concat_5 w_27_0 w_80_concat_6 ∧
  MxxRuntime.concatDiagonal w_80_concat_6 w_27_0 w_80_concat_7 ∧
  MxxRuntime.concatDiagonal w_80_concat_7 w_27_0 w_80_concat_8 ∧
  MxxRuntime.concatDiagonal w_80_concat_8 w_27_0 w_80_concat_9 ∧
  MxxRuntime.concatDiagonal w_80_concat_9 w_27_0 w_80_concat_10 ∧
  MxxRuntime.concatDiagonal w_80_concat_10 w_27_0 w_80_concat_11 ∧
  MxxRuntime.concatDiagonal w_80_concat_11 w_27_0 w_80_concat_12 ∧
  MxxRuntime.concatDiagonal w_80_concat_12 w_27_0 w_80_concat_13 ∧
  MxxRuntime.concatDiagonal w_80_concat_13 w_27_0 w_80_concat_14 ∧
  MxxRuntime.concatDiagonal w_80_concat_14 w_27_0 w_80_concat_15 ∧
  MxxRuntime.concatDiagonal w_80_concat_15 w_27_0 w_80_concat_16 ∧
  MxxRuntime.concatDiagonal w_80_concat_16 w_27_0 w_80_concat_17 ∧
  MxxRuntime.concatDiagonal w_80_concat_17 w_27_0 w_80_concat_18 ∧
  MxxRuntime.concatDiagonal w_80_concat_18 w_27_0 w_80_concat_19

abbrev parallel_generatedRoot_1267.constraints_2 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_46_0 : mxxMat14) (w_46_concat_1 : mxxMat40) (w_46_concat_2 : mxxMat41) (w_46_concat_3 : mxxMat42) (w_46_concat_4 : mxxMat43) (w_46_concat_5 : mxxMat44) (w_46_concat_6 : mxxMat45) (w_46_concat_7 : mxxMat46) (w_46_concat_8 : mxxMat47) (w_46_concat_9 : mxxMat48) (w_46_concat_10 : mxxMat49) (w_46_concat_11 : mxxMat50) (w_46_concat_12 : mxxMat51) (w_46_concat_13 : mxxMat52) (w_46_concat_14 : mxxMat53) (w_46_concat_15 : mxxMat54) (w_46_concat_16 : mxxMat55) (w_46_concat_17 : mxxMat56) (w_46_concat_18 : mxxMat57) (w_46_concat_19 : mxxMat58) (w_46_concat_20 : mxxMat59) (w_46_concat_21 : mxxMat60) (w_46_concat_22 : mxxMat61) (w_46_concat_23 : mxxMat62) (w_47_0 : mxxMat14) (w_49_0 : mxxMat14) (w_50_0 : mxxMat63) (w_51_0 : mxxMat63) (w_56_0 : mxxMat12) (w_58_0 : mxxMat2) (w_59_0 : mxxMat4) (w_60_0 : mxxMat4) (w_63_0 : mxxMat1) (w_64_0 : mxxMat1) : Prop :=
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_46_concat_1 ∧
  MxxRuntime.concatDiagonal w_46_concat_1 w_45_0 w_46_concat_2 ∧
  MxxRuntime.concatDiagonal w_46_concat_2 w_45_0 w_46_concat_3 ∧
  MxxRuntime.concatDiagonal w_46_concat_3 w_45_0 w_46_concat_4 ∧
  MxxRuntime.concatDiagonal w_46_concat_4 w_45_0 w_46_concat_5 ∧
  MxxRuntime.concatDiagonal w_46_concat_5 w_45_0 w_46_concat_6 ∧
  MxxRuntime.concatDiagonal w_46_concat_6 w_45_0 w_46_concat_7 ∧
  MxxRuntime.concatDiagonal w_46_concat_7 w_45_0 w_46_concat_8 ∧
  MxxRuntime.concatDiagonal w_46_concat_8 w_45_0 w_46_concat_9 ∧
  MxxRuntime.concatDiagonal w_46_concat_9 w_45_0 w_46_concat_10 ∧
  MxxRuntime.concatDiagonal w_46_concat_10 w_45_0 w_46_concat_11 ∧
  MxxRuntime.concatDiagonal w_46_concat_11 w_45_0 w_46_concat_12 ∧
  MxxRuntime.concatDiagonal w_46_concat_12 w_45_0 w_46_concat_13 ∧
  MxxRuntime.concatDiagonal w_46_concat_13 w_45_0 w_46_concat_14 ∧
  MxxRuntime.concatDiagonal w_46_concat_14 w_45_0 w_46_concat_15 ∧
  MxxRuntime.concatDiagonal w_46_concat_15 w_45_0 w_46_concat_16 ∧
  MxxRuntime.concatDiagonal w_46_concat_16 w_45_0 w_46_concat_17 ∧
  MxxRuntime.concatDiagonal w_46_concat_17 w_45_0 w_46_concat_18 ∧
  MxxRuntime.concatDiagonal w_46_concat_18 w_45_0 w_46_concat_19 ∧
  MxxRuntime.concatDiagonal w_46_concat_19 w_45_0 w_46_concat_20 ∧
  MxxRuntime.concatDiagonal w_46_concat_20 w_45_0 w_46_concat_21 ∧
  MxxRuntime.concatDiagonal w_46_concat_21 w_45_0 w_46_concat_22 ∧
  MxxRuntime.concatDiagonal w_46_concat_22 w_45_0 w_46_concat_23 ∧
  MxxRuntime.concatDiagonal w_46_concat_23 w_45_0 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  MxxRuntime.concatColumns w_54_0 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_63_0 ∧
  MxxRuntime.centeredRebaseRuns w_63_0 w_64_0

abbrev parallel_generatedRoot_1267.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_43_0 : mxxMat8) (w_24_0 : mxxMat1) (w_28_0 : mxxMat15) (w_28_concat_14 : mxxMat29) (w_28_concat_15 : mxxMat30) (w_28_concat_16 : mxxMat31) (w_28_concat_17 : mxxMat32) (w_28_concat_18 : mxxMat33) (w_28_concat_19 : mxxMat34) (w_28_concat_20 : mxxMat35) (w_28_concat_21 : mxxMat36) (w_28_concat_22 : mxxMat37) (w_28_concat_23 : mxxMat38) (w_29_0 : mxxMat15) (w_31_0 : mxxMat15) (w_32_0 : mxxMat39) (w_33_0 : mxxMat39) (w_34_0 : mxxMat10) (w_35_0 : mxxMat1) (w_36_0 : mxxMat5) (w_37_0 : mxxMat11) (w_38_0 : mxxMat11) (w_39_0 : mxxMat5) (w_40_0 : mxxMat1) (w_41_0 : mxxMat1) (w_44_0 : mxxMat10) : Prop :=
  MxxRuntime.concatDiagonal w_28_concat_14 w_27_0 w_28_concat_15 ∧
  MxxRuntime.concatDiagonal w_28_concat_15 w_27_0 w_28_concat_16 ∧
  MxxRuntime.concatDiagonal w_28_concat_16 w_27_0 w_28_concat_17 ∧
  MxxRuntime.concatDiagonal w_28_concat_17 w_27_0 w_28_concat_18 ∧
  MxxRuntime.concatDiagonal w_28_concat_18 w_27_0 w_28_concat_19 ∧
  MxxRuntime.concatDiagonal w_28_concat_19 w_27_0 w_28_concat_20 ∧
  MxxRuntime.concatDiagonal w_28_concat_20 w_27_0 w_28_concat_21 ∧
  MxxRuntime.concatDiagonal w_28_concat_21 w_27_0 w_28_concat_22 ∧
  MxxRuntime.concatDiagonal w_28_concat_22 w_27_0 w_28_concat_23 ∧
  MxxRuntime.concatDiagonal w_28_concat_23 w_27_0 w_28_0 ∧
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
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1267_41 backend hashModel params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (3125 * (Int.ofNat i_0)) ∧
  (3125 * (Int.ofNat i_0)) < (625 + (3125 * (Int.ofNat i_0))) ∧
  (625 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_43_0 0 1 (3125 * (Int.ofNat i_0)) (625 + (3125 * (Int.ofNat i_0))) w_44_0

abbrev parallel_generatedRoot_1267.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat1) (w_1_0 : mxxMat1) (w_25_0 : mxxMat6) (w_27_0 : mxxMat1) (w_2_0 : mxxMat12) (w_5_0 : mxxMat2) (w_6_0 : mxxMat4) (w_7_0 : mxxMat4) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat13) (w_16_0 : mxxMat13) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat14) (w_21_0 : mxxMat14) (w_26_0 : mxxMat10) (w_28_concat_1 : mxxMat16) (w_28_concat_2 : mxxMat17) (w_28_concat_3 : mxxMat18) (w_28_concat_4 : mxxMat19) (w_28_concat_5 : mxxMat20) (w_28_concat_6 : mxxMat21) (w_28_concat_7 : mxxMat22) (w_28_concat_8 : mxxMat23) (w_28_concat_9 : mxxMat24) (w_28_concat_10 : mxxMat25) (w_28_concat_11 : mxxMat26) (w_28_concat_12 : mxxMat27) (w_28_concat_13 : mxxMat28) (w_28_concat_14 : mxxMat29) : Prop :=
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
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_28_concat_1 ∧
  MxxRuntime.concatDiagonal w_28_concat_1 w_27_0 w_28_concat_2 ∧
  MxxRuntime.concatDiagonal w_28_concat_2 w_27_0 w_28_concat_3 ∧
  MxxRuntime.concatDiagonal w_28_concat_3 w_27_0 w_28_concat_4 ∧
  MxxRuntime.concatDiagonal w_28_concat_4 w_27_0 w_28_concat_5 ∧
  MxxRuntime.concatDiagonal w_28_concat_5 w_27_0 w_28_concat_6 ∧
  MxxRuntime.concatDiagonal w_28_concat_6 w_27_0 w_28_concat_7 ∧
  MxxRuntime.concatDiagonal w_28_concat_7 w_27_0 w_28_concat_8 ∧
  MxxRuntime.concatDiagonal w_28_concat_8 w_27_0 w_28_concat_9 ∧
  MxxRuntime.concatDiagonal w_28_concat_9 w_27_0 w_28_concat_10 ∧
  MxxRuntime.concatDiagonal w_28_concat_10 w_27_0 w_28_concat_11 ∧
  MxxRuntime.concatDiagonal w_28_concat_11 w_27_0 w_28_concat_12 ∧
  MxxRuntime.concatDiagonal w_28_concat_12 w_27_0 w_28_concat_13 ∧
  MxxRuntime.concatDiagonal w_28_concat_13 w_27_0 w_28_concat_14

abbrev parallel_generatedRoot_1267.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) (witness : parallel_generatedRoot_1267.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_4_0 : mxxMat2 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_25_0 : mxxMat6 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_43_0 : mxxMat8 := inputs.2.2.2.2.2.2.1
  let w_45_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_54_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_62_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_78_0 : mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_95_0 : mxxMat8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)
  let w_5_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat64 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat64)
  let w_14_0 : mxxMat13 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat64 := (MxxRuntime.matrixPolynomial [(11560550400)] : mxxMat64)
  let w_16_0 : mxxMat13 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat5)
  let w_19_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat1 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_concat_1 := witness.w_28_concat_1
  let w_28_concat_2 := witness.w_28_concat_2
  let w_28_concat_3 := witness.w_28_concat_3
  let w_28_concat_4 := witness.w_28_concat_4
  let w_28_concat_5 := witness.w_28_concat_5
  let w_28_concat_6 := witness.w_28_concat_6
  let w_28_concat_7 := witness.w_28_concat_7
  let w_28_concat_8 := witness.w_28_concat_8
  let w_28_concat_9 := witness.w_28_concat_9
  let w_28_concat_10 := witness.w_28_concat_10
  let w_28_concat_11 := witness.w_28_concat_11
  let w_28_concat_12 := witness.w_28_concat_12
  let w_28_concat_13 := witness.w_28_concat_13
  let w_28_concat_14 := witness.w_28_concat_14
  let w_28_concat_15 := witness.w_28_concat_15
  let w_28_concat_16 := witness.w_28_concat_16
  let w_28_concat_17 := witness.w_28_concat_17
  let w_28_concat_18 := witness.w_28_concat_18
  let w_28_concat_19 := witness.w_28_concat_19
  let w_28_concat_20 := witness.w_28_concat_20
  let w_28_concat_21 := witness.w_28_concat_21
  let w_28_concat_22 := witness.w_28_concat_22
  let w_28_concat_23 := witness.w_28_concat_23
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_31_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat10 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat5 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_44_0 := witness.w_44_0
  let w_46_0 := witness.w_46_0
  let w_46_concat_1 := witness.w_46_concat_1
  let w_46_concat_2 := witness.w_46_concat_2
  let w_46_concat_3 := witness.w_46_concat_3
  let w_46_concat_4 := witness.w_46_concat_4
  let w_46_concat_5 := witness.w_46_concat_5
  let w_46_concat_6 := witness.w_46_concat_6
  let w_46_concat_7 := witness.w_46_concat_7
  let w_46_concat_8 := witness.w_46_concat_8
  let w_46_concat_9 := witness.w_46_concat_9
  let w_46_concat_10 := witness.w_46_concat_10
  let w_46_concat_11 := witness.w_46_concat_11
  let w_46_concat_12 := witness.w_46_concat_12
  let w_46_concat_13 := witness.w_46_concat_13
  let w_46_concat_14 := witness.w_46_concat_14
  let w_46_concat_15 := witness.w_46_concat_15
  let w_46_concat_16 := witness.w_46_concat_16
  let w_46_concat_17 := witness.w_46_concat_17
  let w_46_concat_18 := witness.w_46_concat_18
  let w_46_concat_19 := witness.w_46_concat_19
  let w_46_concat_20 := witness.w_46_concat_20
  let w_46_concat_21 := witness.w_46_concat_21
  let w_46_concat_22 := witness.w_46_concat_22
  let w_46_concat_23 := witness.w_46_concat_23
  let w_47_0 := witness.w_47_0
  let w_48_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_49_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_47_0 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_44_0 w_51_0
  let w_53_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_52_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_57_0 w_4_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixMul w_56_0 w_60_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : mxxMat64 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat64)
  let w_67_0 : mxxMat13 := MxxRuntime.matrixMulScalarRight w_65_0 w_66_0
  let w_68_0 : mxxMat64 := (MxxRuntime.matrixPolynomial [(11560550400)] : mxxMat64)
  let w_69_0 : mxxMat13 := MxxRuntime.matrixMulScalarRight w_67_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat3)] : mxxMat5)
  let w_72_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_70_0 w_71_0
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 : mxxMat1 := MxxRuntime.matrixMul w_62_0 w_74_0
  let w_76_0 : mxxMat1 := MxxRuntime.matrixSub w_61_0 w_75_0
  let w_77_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_80_concat_1 := witness.w_80_concat_1
  let w_80_concat_2 := witness.w_80_concat_2
  let w_80_concat_3 := witness.w_80_concat_3
  let w_80_concat_4 := witness.w_80_concat_4
  let w_80_concat_5 := witness.w_80_concat_5
  let w_80_concat_6 := witness.w_80_concat_6
  let w_80_concat_7 := witness.w_80_concat_7
  let w_80_concat_8 := witness.w_80_concat_8
  let w_80_concat_9 := witness.w_80_concat_9
  let w_80_concat_10 := witness.w_80_concat_10
  let w_80_concat_11 := witness.w_80_concat_11
  let w_80_concat_12 := witness.w_80_concat_12
  let w_80_concat_13 := witness.w_80_concat_13
  let w_80_concat_14 := witness.w_80_concat_14
  let w_80_concat_15 := witness.w_80_concat_15
  let w_80_concat_16 := witness.w_80_concat_16
  let w_80_concat_17 := witness.w_80_concat_17
  let w_80_concat_18 := witness.w_80_concat_18
  let w_80_concat_19 := witness.w_80_concat_19
  let w_80_concat_20 := witness.w_80_concat_20
  let w_80_concat_21 := witness.w_80_concat_21
  let w_80_concat_22 := witness.w_80_concat_22
  let w_80_concat_23 := witness.w_80_concat_23
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_83_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_81_0 w_82_0
  let w_84_0 := witness.w_84_0
  let w_85_0 := witness.w_85_0
  let w_86_0 : mxxMat10 := MxxRuntime.matrixMul w_79_0 w_85_0
  let w_87_0 := witness.w_87_0
  let w_88_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254428791916078459060479968194539524630727034176641)] : mxxMat5)
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 : mxxMat5 := MxxRuntime.matrixMul w_87_0 w_90_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 : mxxMat1 := MxxRuntime.matrixAdd w_76_0 w_93_0
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_97_concat_1 := witness.w_97_concat_1
  let w_97_concat_2 := witness.w_97_concat_2
  let w_97_concat_3 := witness.w_97_concat_3
  let w_97_concat_4 := witness.w_97_concat_4
  let w_97_concat_5 := witness.w_97_concat_5
  let w_97_concat_6 := witness.w_97_concat_6
  let w_97_concat_7 := witness.w_97_concat_7
  let w_97_concat_8 := witness.w_97_concat_8
  let w_97_concat_9 := witness.w_97_concat_9
  let w_97_concat_10 := witness.w_97_concat_10
  let w_97_concat_11 := witness.w_97_concat_11
  let w_97_concat_12 := witness.w_97_concat_12
  let w_97_concat_13 := witness.w_97_concat_13
  let w_97_concat_14 := witness.w_97_concat_14
  let w_97_concat_15 := witness.w_97_concat_15
  let w_97_concat_16 := witness.w_97_concat_16
  let w_97_concat_17 := witness.w_97_concat_17
  let w_97_concat_18 := witness.w_97_concat_18
  let w_97_concat_19 := witness.w_97_concat_19
  let w_97_concat_20 := witness.w_97_concat_20
  let w_97_concat_21 := witness.w_97_concat_21
  let w_97_concat_22 := witness.w_97_concat_22
  let w_97_concat_23 := witness.w_97_concat_23
  let w_98_0 := witness.w_98_0
  let w_99_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_100_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_98_0 w_99_0
  let w_101_0 := witness.w_101_0
  let w_102_0 := witness.w_102_0
  let w_103_0 : mxxMat1 := MxxRuntime.matrixMul w_96_0 w_102_0
  let w_104_0 : mxxMat1 := MxxRuntime.matrixAdd w_94_0 w_103_0
  let w_105_0 := witness.w_105_0
  parallel_generatedRoot_1267.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_26_0 w_28_concat_1 w_28_concat_2 w_28_concat_3 w_28_concat_4 w_28_concat_5 w_28_concat_6 w_28_concat_7 w_28_concat_8 w_28_concat_9 w_28_concat_10 w_28_concat_11 w_28_concat_12 w_28_concat_13 w_28_concat_14 ∧
    parallel_generatedRoot_1267.constraints_1 backend hashModel params i_0 w_27_0 w_43_0 w_24_0 w_28_0 w_28_concat_14 w_28_concat_15 w_28_concat_16 w_28_concat_17 w_28_concat_18 w_28_concat_19 w_28_concat_20 w_28_concat_21 w_28_concat_22 w_28_concat_23 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_44_0 ∧
    parallel_generatedRoot_1267.constraints_2 backend w_45_0 w_54_0 w_55_0 w_46_0 w_46_concat_1 w_46_concat_2 w_46_concat_3 w_46_concat_4 w_46_concat_5 w_46_concat_6 w_46_concat_7 w_46_concat_8 w_46_concat_9 w_46_concat_10 w_46_concat_11 w_46_concat_12 w_46_concat_13 w_46_concat_14 w_46_concat_15 w_46_concat_16 w_46_concat_17 w_46_concat_18 w_46_concat_19 w_46_concat_20 w_46_concat_21 w_46_concat_22 w_46_concat_23 w_47_0 w_49_0 w_50_0 w_51_0 w_56_0 w_58_0 w_59_0 w_60_0 w_63_0 w_64_0 ∧
    parallel_generatedRoot_1267.constraints_3 backend i_0 w_27_0 w_78_0 w_64_0 w_65_0 w_69_0 w_70_0 w_72_0 w_73_0 w_74_0 w_79_0 w_80_concat_1 w_80_concat_2 w_80_concat_3 w_80_concat_4 w_80_concat_5 w_80_concat_6 w_80_concat_7 w_80_concat_8 w_80_concat_9 w_80_concat_10 w_80_concat_11 w_80_concat_12 w_80_concat_13 w_80_concat_14 w_80_concat_15 w_80_concat_16 w_80_concat_17 w_80_concat_18 w_80_concat_19 ∧
    parallel_generatedRoot_1267.constraints_4 backend hashModel params i_0 w_27_0 w_45_0 w_95_0 w_77_0 w_80_0 w_80_concat_19 w_80_concat_20 w_80_concat_21 w_80_concat_22 w_80_concat_23 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_96_0 w_97_concat_1 w_97_concat_2 w_97_concat_3 w_97_concat_4 w_97_concat_5 ∧
    parallel_generatedRoot_1267.constraints_5 backend w_45_0 w_53_0 w_97_0 w_97_concat_5 w_97_concat_6 w_97_concat_7 w_97_concat_8 w_97_concat_9 w_97_concat_10 w_97_concat_11 w_97_concat_12 w_97_concat_13 w_97_concat_14 w_97_concat_15 w_97_concat_16 w_97_concat_17 w_97_concat_18 w_97_concat_19 w_97_concat_20 w_97_concat_21 w_97_concat_22 w_97_concat_23 w_98_0 w_100_0 w_101_0 w_102_0 w_104_0 w_105_0 outputs

def parallel_generatedRoot_1267 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) : Prop :=
  ∃ witness : parallel_generatedRoot_1267.Witness,
    parallel_generatedRoot_1267.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_1268_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1268_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1268_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_1268_93.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1268_93 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1268_93.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1268.Witness where
  w_2_0 : mxxMat12
  w_6_0 : mxxMat4
  w_7_0 : mxxMat4
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat65
  w_17_0 : mxxMat1
  w_20_0 : mxxMat14
  w_21_0 : mxxMat14
  w_26_0 : mxxMat10
  w_28_0 : mxxMat15
  w_28_concat_1 : mxxMat16
  w_28_concat_2 : mxxMat17
  w_28_concat_3 : mxxMat18
  w_28_concat_4 : mxxMat19
  w_28_concat_5 : mxxMat20
  w_28_concat_6 : mxxMat21
  w_28_concat_7 : mxxMat22
  w_28_concat_8 : mxxMat23
  w_28_concat_9 : mxxMat24
  w_28_concat_10 : mxxMat25
  w_28_concat_11 : mxxMat26
  w_28_concat_12 : mxxMat27
  w_28_concat_13 : mxxMat28
  w_28_concat_14 : mxxMat29
  w_28_concat_15 : mxxMat30
  w_28_concat_16 : mxxMat31
  w_28_concat_17 : mxxMat32
  w_28_concat_18 : mxxMat33
  w_28_concat_19 : mxxMat34
  w_28_concat_20 : mxxMat35
  w_28_concat_21 : mxxMat36
  w_28_concat_22 : mxxMat37
  w_28_concat_23 : mxxMat38
  w_29_0 : mxxMat15
  w_32_0 : mxxMat39
  w_33_0 : mxxMat39
  w_35_0 : mxxMat1
  w_37_0 : mxxMat11
  w_38_0 : mxxMat11
  w_40_0 : mxxMat1
  w_41_0 : mxxMat1
  w_44_0 : mxxMat10
  w_46_0 : mxxMat14
  w_46_concat_1 : mxxMat40
  w_46_concat_2 : mxxMat41
  w_46_concat_3 : mxxMat42
  w_46_concat_4 : mxxMat43
  w_46_concat_5 : mxxMat44
  w_46_concat_6 : mxxMat45
  w_46_concat_7 : mxxMat46
  w_46_concat_8 : mxxMat47
  w_46_concat_9 : mxxMat48
  w_46_concat_10 : mxxMat49
  w_46_concat_11 : mxxMat50
  w_46_concat_12 : mxxMat51
  w_46_concat_13 : mxxMat52
  w_46_concat_14 : mxxMat53
  w_46_concat_15 : mxxMat54
  w_46_concat_16 : mxxMat55
  w_46_concat_17 : mxxMat56
  w_46_concat_18 : mxxMat57
  w_46_concat_19 : mxxMat58
  w_46_concat_20 : mxxMat59
  w_46_concat_21 : mxxMat60
  w_46_concat_22 : mxxMat61
  w_46_concat_23 : mxxMat62
  w_47_0 : mxxMat14
  w_50_0 : mxxMat63
  w_51_0 : mxxMat63
  w_56_0 : mxxMat12
  w_59_0 : mxxMat4
  w_60_0 : mxxMat4
  w_63_0 : mxxMat1
  w_64_0 : mxxMat1
  w_65_0 : mxxMat65
  w_70_0 : mxxMat1
  w_73_0 : mxxMat14
  w_74_0 : mxxMat14
  w_79_0 : mxxMat10
  w_80_0 : mxxMat15
  w_80_concat_1 : mxxMat16
  w_80_concat_2 : mxxMat17
  w_80_concat_3 : mxxMat18
  w_80_concat_4 : mxxMat19
  w_80_concat_5 : mxxMat20
  w_80_concat_6 : mxxMat21
  w_80_concat_7 : mxxMat22
  w_80_concat_8 : mxxMat23
  w_80_concat_9 : mxxMat24
  w_80_concat_10 : mxxMat25
  w_80_concat_11 : mxxMat26
  w_80_concat_12 : mxxMat27
  w_80_concat_13 : mxxMat28
  w_80_concat_14 : mxxMat29
  w_80_concat_15 : mxxMat30
  w_80_concat_16 : mxxMat31
  w_80_concat_17 : mxxMat32
  w_80_concat_18 : mxxMat33
  w_80_concat_19 : mxxMat34
  w_80_concat_20 : mxxMat35
  w_80_concat_21 : mxxMat36
  w_80_concat_22 : mxxMat37
  w_80_concat_23 : mxxMat38
  w_81_0 : mxxMat15
  w_84_0 : mxxMat39
  w_85_0 : mxxMat39
  w_87_0 : mxxMat1
  w_89_0 : mxxMat11
  w_90_0 : mxxMat11
  w_92_0 : mxxMat1
  w_93_0 : mxxMat1
  w_96_0 : mxxMat10
  w_97_0 : mxxMat14
  w_97_concat_1 : mxxMat40
  w_97_concat_2 : mxxMat41
  w_97_concat_3 : mxxMat42
  w_97_concat_4 : mxxMat43
  w_97_concat_5 : mxxMat44
  w_97_concat_6 : mxxMat45
  w_97_concat_7 : mxxMat46
  w_97_concat_8 : mxxMat47
  w_97_concat_9 : mxxMat48
  w_97_concat_10 : mxxMat49
  w_97_concat_11 : mxxMat50
  w_97_concat_12 : mxxMat51
  w_97_concat_13 : mxxMat52
  w_97_concat_14 : mxxMat53
  w_97_concat_15 : mxxMat54
  w_97_concat_16 : mxxMat55
  w_97_concat_17 : mxxMat56
  w_97_concat_18 : mxxMat57
  w_97_concat_19 : mxxMat58
  w_97_concat_20 : mxxMat59
  w_97_concat_21 : mxxMat60
  w_97_concat_22 : mxxMat61
  w_97_concat_23 : mxxMat62
  w_98_0 : mxxMat14
  w_101_0 : mxxMat63
  w_102_0 : mxxMat63
  w_105_0 : mxxMat2

abbrev parallel_generatedRoot_1268.constraints_5 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_53_0 : mxxMat1) (w_97_0 : mxxMat14) (w_97_concat_5 : mxxMat44) (w_97_concat_6 : mxxMat45) (w_97_concat_7 : mxxMat46) (w_97_concat_8 : mxxMat47) (w_97_concat_9 : mxxMat48) (w_97_concat_10 : mxxMat49) (w_97_concat_11 : mxxMat50) (w_97_concat_12 : mxxMat51) (w_97_concat_13 : mxxMat52) (w_97_concat_14 : mxxMat53) (w_97_concat_15 : mxxMat54) (w_97_concat_16 : mxxMat55) (w_97_concat_17 : mxxMat56) (w_97_concat_18 : mxxMat57) (w_97_concat_19 : mxxMat58) (w_97_concat_20 : mxxMat59) (w_97_concat_21 : mxxMat60) (w_97_concat_22 : mxxMat61) (w_97_concat_23 : mxxMat62) (w_98_0 : mxxMat14) (w_100_0 : mxxMat14) (w_101_0 : mxxMat63) (w_102_0 : mxxMat63) (w_104_0 : mxxMat1) (w_105_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  MxxRuntime.concatDiagonal w_97_concat_5 w_45_0 w_97_concat_6 ∧
  MxxRuntime.concatDiagonal w_97_concat_6 w_45_0 w_97_concat_7 ∧
  MxxRuntime.concatDiagonal w_97_concat_7 w_45_0 w_97_concat_8 ∧
  MxxRuntime.concatDiagonal w_97_concat_8 w_45_0 w_97_concat_9 ∧
  MxxRuntime.concatDiagonal w_97_concat_9 w_45_0 w_97_concat_10 ∧
  MxxRuntime.concatDiagonal w_97_concat_10 w_45_0 w_97_concat_11 ∧
  MxxRuntime.concatDiagonal w_97_concat_11 w_45_0 w_97_concat_12 ∧
  MxxRuntime.concatDiagonal w_97_concat_12 w_45_0 w_97_concat_13 ∧
  MxxRuntime.concatDiagonal w_97_concat_13 w_45_0 w_97_concat_14 ∧
  MxxRuntime.concatDiagonal w_97_concat_14 w_45_0 w_97_concat_15 ∧
  MxxRuntime.concatDiagonal w_97_concat_15 w_45_0 w_97_concat_16 ∧
  MxxRuntime.concatDiagonal w_97_concat_16 w_45_0 w_97_concat_17 ∧
  MxxRuntime.concatDiagonal w_97_concat_17 w_45_0 w_97_concat_18 ∧
  MxxRuntime.concatDiagonal w_97_concat_18 w_45_0 w_97_concat_19 ∧
  MxxRuntime.concatDiagonal w_97_concat_19 w_45_0 w_97_concat_20 ∧
  MxxRuntime.concatDiagonal w_97_concat_20 w_45_0 w_97_concat_21 ∧
  MxxRuntime.concatDiagonal w_97_concat_21 w_45_0 w_97_concat_22 ∧
  MxxRuntime.concatDiagonal w_97_concat_22 w_45_0 w_97_concat_23 ∧
  MxxRuntime.concatDiagonal w_97_concat_23 w_45_0 w_97_0 ∧
  MxxRuntime.centeredRebaseRuns w_97_0 w_98_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_100_0 w_101_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_101_0 w_102_0 ∧
  MxxRuntime.concatRows w_53_0 w_104_0 w_105_0 ∧
  outputs = w_105_0

abbrev parallel_generatedRoot_1268.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_45_0 : mxxMat5) (w_95_0 : mxxMat8) (w_77_0 : mxxMat1) (w_80_0 : mxxMat15) (w_80_concat_19 : mxxMat34) (w_80_concat_20 : mxxMat35) (w_80_concat_21 : mxxMat36) (w_80_concat_22 : mxxMat37) (w_80_concat_23 : mxxMat38) (w_81_0 : mxxMat15) (w_83_0 : mxxMat15) (w_84_0 : mxxMat39) (w_85_0 : mxxMat39) (w_86_0 : mxxMat10) (w_87_0 : mxxMat1) (w_88_0 : mxxMat5) (w_89_0 : mxxMat11) (w_90_0 : mxxMat11) (w_91_0 : mxxMat5) (w_92_0 : mxxMat1) (w_93_0 : mxxMat1) (w_96_0 : mxxMat10) (w_97_concat_1 : mxxMat40) (w_97_concat_2 : mxxMat41) (w_97_concat_3 : mxxMat42) (w_97_concat_4 : mxxMat43) (w_97_concat_5 : mxxMat44) : Prop :=
  MxxRuntime.concatDiagonal w_80_concat_19 w_27_0 w_80_concat_20 ∧
  MxxRuntime.concatDiagonal w_80_concat_20 w_27_0 w_80_concat_21 ∧
  MxxRuntime.concatDiagonal w_80_concat_21 w_27_0 w_80_concat_22 ∧
  MxxRuntime.concatDiagonal w_80_concat_22 w_27_0 w_80_concat_23 ∧
  MxxRuntime.concatDiagonal w_80_concat_23 w_27_0 w_80_0 ∧
  MxxRuntime.centeredRebaseRuns w_80_0 w_81_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_83_0 w_84_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_84_0 w_85_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_86_0 0 1 0 25 w_87_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_88_0 w_89_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_89_0 w_90_0 ∧
  MxxRuntime.tensorRuns w_77_0 w_91_0 w_92_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1268_93 backend hashModel params i_0 i (current, (w_86_0, (w_90_0, ()))) next) (Int.toNat (24)) w_92_0 w_93_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 + (3125 * (Int.ofNat i_0))) ∧
  (625 + (3125 * (Int.ofNat i_0))) < (1250 + (3125 * (Int.ofNat i_0))) ∧
  (1250 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_95_0 0 1 (625 + (3125 * (Int.ofNat i_0))) (1250 + (3125 * (Int.ofNat i_0))) w_96_0 ∧
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_97_concat_1 ∧
  MxxRuntime.concatDiagonal w_97_concat_1 w_45_0 w_97_concat_2 ∧
  MxxRuntime.concatDiagonal w_97_concat_2 w_45_0 w_97_concat_3 ∧
  MxxRuntime.concatDiagonal w_97_concat_3 w_45_0 w_97_concat_4 ∧
  MxxRuntime.concatDiagonal w_97_concat_4 w_45_0 w_97_concat_5

abbrev parallel_generatedRoot_1268.constraints_3 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_27_0 : mxxMat1) (w_78_0 : mxxMat6) (w_64_0 : mxxMat1) (w_65_0 : mxxMat65) (w_69_0 : mxxMat65) (w_70_0 : mxxMat1) (w_72_0 : mxxMat1) (w_73_0 : mxxMat14) (w_74_0 : mxxMat14) (w_79_0 : mxxMat10) (w_80_concat_1 : mxxMat16) (w_80_concat_2 : mxxMat17) (w_80_concat_3 : mxxMat18) (w_80_concat_4 : mxxMat19) (w_80_concat_5 : mxxMat20) (w_80_concat_6 : mxxMat21) (w_80_concat_7 : mxxMat22) (w_80_concat_8 : mxxMat23) (w_80_concat_9 : mxxMat24) (w_80_concat_10 : mxxMat25) (w_80_concat_11 : mxxMat26) (w_80_concat_12 : mxxMat27) (w_80_concat_13 : mxxMat28) (w_80_concat_14 : mxxMat29) (w_80_concat_15 : mxxMat30) (w_80_concat_16 : mxxMat31) (w_80_concat_17 : mxxMat32) (w_80_concat_18 : mxxMat33) (w_80_concat_19 : mxxMat34) : Prop :=
  MxxRuntime.modulusReduceRuns w_64_0 w_65_0 ∧
  MxxRuntime.centeredRebaseRuns w_69_0 w_70_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_72_0 w_73_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_73_0 w_74_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_79_0 ∧
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_80_concat_1 ∧
  MxxRuntime.concatDiagonal w_80_concat_1 w_27_0 w_80_concat_2 ∧
  MxxRuntime.concatDiagonal w_80_concat_2 w_27_0 w_80_concat_3 ∧
  MxxRuntime.concatDiagonal w_80_concat_3 w_27_0 w_80_concat_4 ∧
  MxxRuntime.concatDiagonal w_80_concat_4 w_27_0 w_80_concat_5 ∧
  MxxRuntime.concatDiagonal w_80_concat_5 w_27_0 w_80_concat_6 ∧
  MxxRuntime.concatDiagonal w_80_concat_6 w_27_0 w_80_concat_7 ∧
  MxxRuntime.concatDiagonal w_80_concat_7 w_27_0 w_80_concat_8 ∧
  MxxRuntime.concatDiagonal w_80_concat_8 w_27_0 w_80_concat_9 ∧
  MxxRuntime.concatDiagonal w_80_concat_9 w_27_0 w_80_concat_10 ∧
  MxxRuntime.concatDiagonal w_80_concat_10 w_27_0 w_80_concat_11 ∧
  MxxRuntime.concatDiagonal w_80_concat_11 w_27_0 w_80_concat_12 ∧
  MxxRuntime.concatDiagonal w_80_concat_12 w_27_0 w_80_concat_13 ∧
  MxxRuntime.concatDiagonal w_80_concat_13 w_27_0 w_80_concat_14 ∧
  MxxRuntime.concatDiagonal w_80_concat_14 w_27_0 w_80_concat_15 ∧
  MxxRuntime.concatDiagonal w_80_concat_15 w_27_0 w_80_concat_16 ∧
  MxxRuntime.concatDiagonal w_80_concat_16 w_27_0 w_80_concat_17 ∧
  MxxRuntime.concatDiagonal w_80_concat_17 w_27_0 w_80_concat_18 ∧
  MxxRuntime.concatDiagonal w_80_concat_18 w_27_0 w_80_concat_19

abbrev parallel_generatedRoot_1268.constraints_2 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_46_0 : mxxMat14) (w_46_concat_1 : mxxMat40) (w_46_concat_2 : mxxMat41) (w_46_concat_3 : mxxMat42) (w_46_concat_4 : mxxMat43) (w_46_concat_5 : mxxMat44) (w_46_concat_6 : mxxMat45) (w_46_concat_7 : mxxMat46) (w_46_concat_8 : mxxMat47) (w_46_concat_9 : mxxMat48) (w_46_concat_10 : mxxMat49) (w_46_concat_11 : mxxMat50) (w_46_concat_12 : mxxMat51) (w_46_concat_13 : mxxMat52) (w_46_concat_14 : mxxMat53) (w_46_concat_15 : mxxMat54) (w_46_concat_16 : mxxMat55) (w_46_concat_17 : mxxMat56) (w_46_concat_18 : mxxMat57) (w_46_concat_19 : mxxMat58) (w_46_concat_20 : mxxMat59) (w_46_concat_21 : mxxMat60) (w_46_concat_22 : mxxMat61) (w_46_concat_23 : mxxMat62) (w_47_0 : mxxMat14) (w_49_0 : mxxMat14) (w_50_0 : mxxMat63) (w_51_0 : mxxMat63) (w_56_0 : mxxMat12) (w_58_0 : mxxMat2) (w_59_0 : mxxMat4) (w_60_0 : mxxMat4) (w_63_0 : mxxMat1) (w_64_0 : mxxMat1) : Prop :=
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_46_concat_1 ∧
  MxxRuntime.concatDiagonal w_46_concat_1 w_45_0 w_46_concat_2 ∧
  MxxRuntime.concatDiagonal w_46_concat_2 w_45_0 w_46_concat_3 ∧
  MxxRuntime.concatDiagonal w_46_concat_3 w_45_0 w_46_concat_4 ∧
  MxxRuntime.concatDiagonal w_46_concat_4 w_45_0 w_46_concat_5 ∧
  MxxRuntime.concatDiagonal w_46_concat_5 w_45_0 w_46_concat_6 ∧
  MxxRuntime.concatDiagonal w_46_concat_6 w_45_0 w_46_concat_7 ∧
  MxxRuntime.concatDiagonal w_46_concat_7 w_45_0 w_46_concat_8 ∧
  MxxRuntime.concatDiagonal w_46_concat_8 w_45_0 w_46_concat_9 ∧
  MxxRuntime.concatDiagonal w_46_concat_9 w_45_0 w_46_concat_10 ∧
  MxxRuntime.concatDiagonal w_46_concat_10 w_45_0 w_46_concat_11 ∧
  MxxRuntime.concatDiagonal w_46_concat_11 w_45_0 w_46_concat_12 ∧
  MxxRuntime.concatDiagonal w_46_concat_12 w_45_0 w_46_concat_13 ∧
  MxxRuntime.concatDiagonal w_46_concat_13 w_45_0 w_46_concat_14 ∧
  MxxRuntime.concatDiagonal w_46_concat_14 w_45_0 w_46_concat_15 ∧
  MxxRuntime.concatDiagonal w_46_concat_15 w_45_0 w_46_concat_16 ∧
  MxxRuntime.concatDiagonal w_46_concat_16 w_45_0 w_46_concat_17 ∧
  MxxRuntime.concatDiagonal w_46_concat_17 w_45_0 w_46_concat_18 ∧
  MxxRuntime.concatDiagonal w_46_concat_18 w_45_0 w_46_concat_19 ∧
  MxxRuntime.concatDiagonal w_46_concat_19 w_45_0 w_46_concat_20 ∧
  MxxRuntime.concatDiagonal w_46_concat_20 w_45_0 w_46_concat_21 ∧
  MxxRuntime.concatDiagonal w_46_concat_21 w_45_0 w_46_concat_22 ∧
  MxxRuntime.concatDiagonal w_46_concat_22 w_45_0 w_46_concat_23 ∧
  MxxRuntime.concatDiagonal w_46_concat_23 w_45_0 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  MxxRuntime.concatColumns w_54_0 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_63_0 ∧
  MxxRuntime.centeredRebaseRuns w_63_0 w_64_0

abbrev parallel_generatedRoot_1268.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_43_0 : mxxMat8) (w_24_0 : mxxMat1) (w_28_0 : mxxMat15) (w_28_concat_14 : mxxMat29) (w_28_concat_15 : mxxMat30) (w_28_concat_16 : mxxMat31) (w_28_concat_17 : mxxMat32) (w_28_concat_18 : mxxMat33) (w_28_concat_19 : mxxMat34) (w_28_concat_20 : mxxMat35) (w_28_concat_21 : mxxMat36) (w_28_concat_22 : mxxMat37) (w_28_concat_23 : mxxMat38) (w_29_0 : mxxMat15) (w_31_0 : mxxMat15) (w_32_0 : mxxMat39) (w_33_0 : mxxMat39) (w_34_0 : mxxMat10) (w_35_0 : mxxMat1) (w_36_0 : mxxMat5) (w_37_0 : mxxMat11) (w_38_0 : mxxMat11) (w_39_0 : mxxMat5) (w_40_0 : mxxMat1) (w_41_0 : mxxMat1) (w_44_0 : mxxMat10) : Prop :=
  MxxRuntime.concatDiagonal w_28_concat_14 w_27_0 w_28_concat_15 ∧
  MxxRuntime.concatDiagonal w_28_concat_15 w_27_0 w_28_concat_16 ∧
  MxxRuntime.concatDiagonal w_28_concat_16 w_27_0 w_28_concat_17 ∧
  MxxRuntime.concatDiagonal w_28_concat_17 w_27_0 w_28_concat_18 ∧
  MxxRuntime.concatDiagonal w_28_concat_18 w_27_0 w_28_concat_19 ∧
  MxxRuntime.concatDiagonal w_28_concat_19 w_27_0 w_28_concat_20 ∧
  MxxRuntime.concatDiagonal w_28_concat_20 w_27_0 w_28_concat_21 ∧
  MxxRuntime.concatDiagonal w_28_concat_21 w_27_0 w_28_concat_22 ∧
  MxxRuntime.concatDiagonal w_28_concat_22 w_27_0 w_28_concat_23 ∧
  MxxRuntime.concatDiagonal w_28_concat_23 w_27_0 w_28_0 ∧
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
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1268_41 backend hashModel params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 + (3125 * (Int.ofNat i_0))) ∧
  (625 + (3125 * (Int.ofNat i_0))) < (1250 + (3125 * (Int.ofNat i_0))) ∧
  (1250 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_43_0 0 1 (625 + (3125 * (Int.ofNat i_0))) (1250 + (3125 * (Int.ofNat i_0))) w_44_0

abbrev parallel_generatedRoot_1268.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat1) (w_1_0 : mxxMat1) (w_25_0 : mxxMat6) (w_27_0 : mxxMat1) (w_2_0 : mxxMat12) (w_5_0 : mxxMat2) (w_6_0 : mxxMat4) (w_7_0 : mxxMat4) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat65) (w_16_0 : mxxMat65) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat14) (w_21_0 : mxxMat14) (w_26_0 : mxxMat10) (w_28_concat_1 : mxxMat16) (w_28_concat_2 : mxxMat17) (w_28_concat_3 : mxxMat18) (w_28_concat_4 : mxxMat19) (w_28_concat_5 : mxxMat20) (w_28_concat_6 : mxxMat21) (w_28_concat_7 : mxxMat22) (w_28_concat_8 : mxxMat23) (w_28_concat_9 : mxxMat24) (w_28_concat_10 : mxxMat25) (w_28_concat_11 : mxxMat26) (w_28_concat_12 : mxxMat27) (w_28_concat_13 : mxxMat28) (w_28_concat_14 : mxxMat29) : Prop :=
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
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_28_concat_1 ∧
  MxxRuntime.concatDiagonal w_28_concat_1 w_27_0 w_28_concat_2 ∧
  MxxRuntime.concatDiagonal w_28_concat_2 w_27_0 w_28_concat_3 ∧
  MxxRuntime.concatDiagonal w_28_concat_3 w_27_0 w_28_concat_4 ∧
  MxxRuntime.concatDiagonal w_28_concat_4 w_27_0 w_28_concat_5 ∧
  MxxRuntime.concatDiagonal w_28_concat_5 w_27_0 w_28_concat_6 ∧
  MxxRuntime.concatDiagonal w_28_concat_6 w_27_0 w_28_concat_7 ∧
  MxxRuntime.concatDiagonal w_28_concat_7 w_27_0 w_28_concat_8 ∧
  MxxRuntime.concatDiagonal w_28_concat_8 w_27_0 w_28_concat_9 ∧
  MxxRuntime.concatDiagonal w_28_concat_9 w_27_0 w_28_concat_10 ∧
  MxxRuntime.concatDiagonal w_28_concat_10 w_27_0 w_28_concat_11 ∧
  MxxRuntime.concatDiagonal w_28_concat_11 w_27_0 w_28_concat_12 ∧
  MxxRuntime.concatDiagonal w_28_concat_12 w_27_0 w_28_concat_13 ∧
  MxxRuntime.concatDiagonal w_28_concat_13 w_27_0 w_28_concat_14

abbrev parallel_generatedRoot_1268.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) (witness : parallel_generatedRoot_1268.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_4_0 : mxxMat2 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_25_0 : mxxMat6 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_43_0 : mxxMat8 := inputs.2.2.2.2.2.2.1
  let w_45_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_54_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_62_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_78_0 : mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_95_0 : mxxMat8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)
  let w_5_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat66 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat66)
  let w_14_0 : mxxMat65 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat66 := (MxxRuntime.matrixPolynomial [(1125894775373377)] : mxxMat66)
  let w_16_0 : mxxMat65 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat4)] : mxxMat5)
  let w_19_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat1 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_concat_1 := witness.w_28_concat_1
  let w_28_concat_2 := witness.w_28_concat_2
  let w_28_concat_3 := witness.w_28_concat_3
  let w_28_concat_4 := witness.w_28_concat_4
  let w_28_concat_5 := witness.w_28_concat_5
  let w_28_concat_6 := witness.w_28_concat_6
  let w_28_concat_7 := witness.w_28_concat_7
  let w_28_concat_8 := witness.w_28_concat_8
  let w_28_concat_9 := witness.w_28_concat_9
  let w_28_concat_10 := witness.w_28_concat_10
  let w_28_concat_11 := witness.w_28_concat_11
  let w_28_concat_12 := witness.w_28_concat_12
  let w_28_concat_13 := witness.w_28_concat_13
  let w_28_concat_14 := witness.w_28_concat_14
  let w_28_concat_15 := witness.w_28_concat_15
  let w_28_concat_16 := witness.w_28_concat_16
  let w_28_concat_17 := witness.w_28_concat_17
  let w_28_concat_18 := witness.w_28_concat_18
  let w_28_concat_19 := witness.w_28_concat_19
  let w_28_concat_20 := witness.w_28_concat_20
  let w_28_concat_21 := witness.w_28_concat_21
  let w_28_concat_22 := witness.w_28_concat_22
  let w_28_concat_23 := witness.w_28_concat_23
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_31_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat10 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat5 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_44_0 := witness.w_44_0
  let w_46_0 := witness.w_46_0
  let w_46_concat_1 := witness.w_46_concat_1
  let w_46_concat_2 := witness.w_46_concat_2
  let w_46_concat_3 := witness.w_46_concat_3
  let w_46_concat_4 := witness.w_46_concat_4
  let w_46_concat_5 := witness.w_46_concat_5
  let w_46_concat_6 := witness.w_46_concat_6
  let w_46_concat_7 := witness.w_46_concat_7
  let w_46_concat_8 := witness.w_46_concat_8
  let w_46_concat_9 := witness.w_46_concat_9
  let w_46_concat_10 := witness.w_46_concat_10
  let w_46_concat_11 := witness.w_46_concat_11
  let w_46_concat_12 := witness.w_46_concat_12
  let w_46_concat_13 := witness.w_46_concat_13
  let w_46_concat_14 := witness.w_46_concat_14
  let w_46_concat_15 := witness.w_46_concat_15
  let w_46_concat_16 := witness.w_46_concat_16
  let w_46_concat_17 := witness.w_46_concat_17
  let w_46_concat_18 := witness.w_46_concat_18
  let w_46_concat_19 := witness.w_46_concat_19
  let w_46_concat_20 := witness.w_46_concat_20
  let w_46_concat_21 := witness.w_46_concat_21
  let w_46_concat_22 := witness.w_46_concat_22
  let w_46_concat_23 := witness.w_46_concat_23
  let w_47_0 := witness.w_47_0
  let w_48_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_49_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_47_0 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_44_0 w_51_0
  let w_53_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_52_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_57_0 w_4_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixMul w_56_0 w_60_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : mxxMat66 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat66)
  let w_67_0 : mxxMat65 := MxxRuntime.matrixMulScalarRight w_65_0 w_66_0
  let w_68_0 : mxxMat66 := (MxxRuntime.matrixPolynomial [(1125894775373377)] : mxxMat66)
  let w_69_0 : mxxMat65 := MxxRuntime.matrixMulScalarRight w_67_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat4)] : mxxMat5)
  let w_72_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_70_0 w_71_0
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 : mxxMat1 := MxxRuntime.matrixMul w_62_0 w_74_0
  let w_76_0 : mxxMat1 := MxxRuntime.matrixSub w_61_0 w_75_0
  let w_77_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_80_concat_1 := witness.w_80_concat_1
  let w_80_concat_2 := witness.w_80_concat_2
  let w_80_concat_3 := witness.w_80_concat_3
  let w_80_concat_4 := witness.w_80_concat_4
  let w_80_concat_5 := witness.w_80_concat_5
  let w_80_concat_6 := witness.w_80_concat_6
  let w_80_concat_7 := witness.w_80_concat_7
  let w_80_concat_8 := witness.w_80_concat_8
  let w_80_concat_9 := witness.w_80_concat_9
  let w_80_concat_10 := witness.w_80_concat_10
  let w_80_concat_11 := witness.w_80_concat_11
  let w_80_concat_12 := witness.w_80_concat_12
  let w_80_concat_13 := witness.w_80_concat_13
  let w_80_concat_14 := witness.w_80_concat_14
  let w_80_concat_15 := witness.w_80_concat_15
  let w_80_concat_16 := witness.w_80_concat_16
  let w_80_concat_17 := witness.w_80_concat_17
  let w_80_concat_18 := witness.w_80_concat_18
  let w_80_concat_19 := witness.w_80_concat_19
  let w_80_concat_20 := witness.w_80_concat_20
  let w_80_concat_21 := witness.w_80_concat_21
  let w_80_concat_22 := witness.w_80_concat_22
  let w_80_concat_23 := witness.w_80_concat_23
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_83_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_81_0 w_82_0
  let w_84_0 := witness.w_84_0
  let w_85_0 := witness.w_85_0
  let w_86_0 : mxxMat10 := MxxRuntime.matrixMul w_79_0 w_85_0
  let w_87_0 := witness.w_87_0
  let w_88_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254565807694577896670997561277593686540426466090209)] : mxxMat5)
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 : mxxMat5 := MxxRuntime.matrixMul w_87_0 w_90_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 : mxxMat1 := MxxRuntime.matrixAdd w_76_0 w_93_0
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_97_concat_1 := witness.w_97_concat_1
  let w_97_concat_2 := witness.w_97_concat_2
  let w_97_concat_3 := witness.w_97_concat_3
  let w_97_concat_4 := witness.w_97_concat_4
  let w_97_concat_5 := witness.w_97_concat_5
  let w_97_concat_6 := witness.w_97_concat_6
  let w_97_concat_7 := witness.w_97_concat_7
  let w_97_concat_8 := witness.w_97_concat_8
  let w_97_concat_9 := witness.w_97_concat_9
  let w_97_concat_10 := witness.w_97_concat_10
  let w_97_concat_11 := witness.w_97_concat_11
  let w_97_concat_12 := witness.w_97_concat_12
  let w_97_concat_13 := witness.w_97_concat_13
  let w_97_concat_14 := witness.w_97_concat_14
  let w_97_concat_15 := witness.w_97_concat_15
  let w_97_concat_16 := witness.w_97_concat_16
  let w_97_concat_17 := witness.w_97_concat_17
  let w_97_concat_18 := witness.w_97_concat_18
  let w_97_concat_19 := witness.w_97_concat_19
  let w_97_concat_20 := witness.w_97_concat_20
  let w_97_concat_21 := witness.w_97_concat_21
  let w_97_concat_22 := witness.w_97_concat_22
  let w_97_concat_23 := witness.w_97_concat_23
  let w_98_0 := witness.w_98_0
  let w_99_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_100_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_98_0 w_99_0
  let w_101_0 := witness.w_101_0
  let w_102_0 := witness.w_102_0
  let w_103_0 : mxxMat1 := MxxRuntime.matrixMul w_96_0 w_102_0
  let w_104_0 : mxxMat1 := MxxRuntime.matrixAdd w_94_0 w_103_0
  let w_105_0 := witness.w_105_0
  parallel_generatedRoot_1268.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_26_0 w_28_concat_1 w_28_concat_2 w_28_concat_3 w_28_concat_4 w_28_concat_5 w_28_concat_6 w_28_concat_7 w_28_concat_8 w_28_concat_9 w_28_concat_10 w_28_concat_11 w_28_concat_12 w_28_concat_13 w_28_concat_14 ∧
    parallel_generatedRoot_1268.constraints_1 backend hashModel params i_0 w_27_0 w_43_0 w_24_0 w_28_0 w_28_concat_14 w_28_concat_15 w_28_concat_16 w_28_concat_17 w_28_concat_18 w_28_concat_19 w_28_concat_20 w_28_concat_21 w_28_concat_22 w_28_concat_23 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_44_0 ∧
    parallel_generatedRoot_1268.constraints_2 backend w_45_0 w_54_0 w_55_0 w_46_0 w_46_concat_1 w_46_concat_2 w_46_concat_3 w_46_concat_4 w_46_concat_5 w_46_concat_6 w_46_concat_7 w_46_concat_8 w_46_concat_9 w_46_concat_10 w_46_concat_11 w_46_concat_12 w_46_concat_13 w_46_concat_14 w_46_concat_15 w_46_concat_16 w_46_concat_17 w_46_concat_18 w_46_concat_19 w_46_concat_20 w_46_concat_21 w_46_concat_22 w_46_concat_23 w_47_0 w_49_0 w_50_0 w_51_0 w_56_0 w_58_0 w_59_0 w_60_0 w_63_0 w_64_0 ∧
    parallel_generatedRoot_1268.constraints_3 backend i_0 w_27_0 w_78_0 w_64_0 w_65_0 w_69_0 w_70_0 w_72_0 w_73_0 w_74_0 w_79_0 w_80_concat_1 w_80_concat_2 w_80_concat_3 w_80_concat_4 w_80_concat_5 w_80_concat_6 w_80_concat_7 w_80_concat_8 w_80_concat_9 w_80_concat_10 w_80_concat_11 w_80_concat_12 w_80_concat_13 w_80_concat_14 w_80_concat_15 w_80_concat_16 w_80_concat_17 w_80_concat_18 w_80_concat_19 ∧
    parallel_generatedRoot_1268.constraints_4 backend hashModel params i_0 w_27_0 w_45_0 w_95_0 w_77_0 w_80_0 w_80_concat_19 w_80_concat_20 w_80_concat_21 w_80_concat_22 w_80_concat_23 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_96_0 w_97_concat_1 w_97_concat_2 w_97_concat_3 w_97_concat_4 w_97_concat_5 ∧
    parallel_generatedRoot_1268.constraints_5 backend w_45_0 w_53_0 w_97_0 w_97_concat_5 w_97_concat_6 w_97_concat_7 w_97_concat_8 w_97_concat_9 w_97_concat_10 w_97_concat_11 w_97_concat_12 w_97_concat_13 w_97_concat_14 w_97_concat_15 w_97_concat_16 w_97_concat_17 w_97_concat_18 w_97_concat_19 w_97_concat_20 w_97_concat_21 w_97_concat_22 w_97_concat_23 w_98_0 w_100_0 w_101_0 w_102_0 w_104_0 w_105_0 outputs

def parallel_generatedRoot_1268 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) : Prop :=
  ∃ witness : parallel_generatedRoot_1268.Witness,
    parallel_generatedRoot_1268.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_1269_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1269_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1269_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_1269_93.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1269_93 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1269_93.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1269.Witness where
  w_2_0 : mxxMat12
  w_6_0 : mxxMat4
  w_7_0 : mxxMat4
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat67
  w_17_0 : mxxMat1
  w_20_0 : mxxMat14
  w_21_0 : mxxMat14
  w_26_0 : mxxMat10
  w_28_0 : mxxMat15
  w_28_concat_1 : mxxMat16
  w_28_concat_2 : mxxMat17
  w_28_concat_3 : mxxMat18
  w_28_concat_4 : mxxMat19
  w_28_concat_5 : mxxMat20
  w_28_concat_6 : mxxMat21
  w_28_concat_7 : mxxMat22
  w_28_concat_8 : mxxMat23
  w_28_concat_9 : mxxMat24
  w_28_concat_10 : mxxMat25
  w_28_concat_11 : mxxMat26
  w_28_concat_12 : mxxMat27
  w_28_concat_13 : mxxMat28
  w_28_concat_14 : mxxMat29
  w_28_concat_15 : mxxMat30
  w_28_concat_16 : mxxMat31
  w_28_concat_17 : mxxMat32
  w_28_concat_18 : mxxMat33
  w_28_concat_19 : mxxMat34
  w_28_concat_20 : mxxMat35
  w_28_concat_21 : mxxMat36
  w_28_concat_22 : mxxMat37
  w_28_concat_23 : mxxMat38
  w_29_0 : mxxMat15
  w_32_0 : mxxMat39
  w_33_0 : mxxMat39
  w_35_0 : mxxMat1
  w_37_0 : mxxMat11
  w_38_0 : mxxMat11
  w_40_0 : mxxMat1
  w_41_0 : mxxMat1
  w_44_0 : mxxMat10
  w_46_0 : mxxMat14
  w_46_concat_1 : mxxMat40
  w_46_concat_2 : mxxMat41
  w_46_concat_3 : mxxMat42
  w_46_concat_4 : mxxMat43
  w_46_concat_5 : mxxMat44
  w_46_concat_6 : mxxMat45
  w_46_concat_7 : mxxMat46
  w_46_concat_8 : mxxMat47
  w_46_concat_9 : mxxMat48
  w_46_concat_10 : mxxMat49
  w_46_concat_11 : mxxMat50
  w_46_concat_12 : mxxMat51
  w_46_concat_13 : mxxMat52
  w_46_concat_14 : mxxMat53
  w_46_concat_15 : mxxMat54
  w_46_concat_16 : mxxMat55
  w_46_concat_17 : mxxMat56
  w_46_concat_18 : mxxMat57
  w_46_concat_19 : mxxMat58
  w_46_concat_20 : mxxMat59
  w_46_concat_21 : mxxMat60
  w_46_concat_22 : mxxMat61
  w_46_concat_23 : mxxMat62
  w_47_0 : mxxMat14
  w_50_0 : mxxMat63
  w_51_0 : mxxMat63
  w_56_0 : mxxMat12
  w_59_0 : mxxMat4
  w_60_0 : mxxMat4
  w_63_0 : mxxMat1
  w_64_0 : mxxMat1
  w_65_0 : mxxMat67
  w_70_0 : mxxMat1
  w_73_0 : mxxMat14
  w_74_0 : mxxMat14
  w_79_0 : mxxMat10
  w_80_0 : mxxMat15
  w_80_concat_1 : mxxMat16
  w_80_concat_2 : mxxMat17
  w_80_concat_3 : mxxMat18
  w_80_concat_4 : mxxMat19
  w_80_concat_5 : mxxMat20
  w_80_concat_6 : mxxMat21
  w_80_concat_7 : mxxMat22
  w_80_concat_8 : mxxMat23
  w_80_concat_9 : mxxMat24
  w_80_concat_10 : mxxMat25
  w_80_concat_11 : mxxMat26
  w_80_concat_12 : mxxMat27
  w_80_concat_13 : mxxMat28
  w_80_concat_14 : mxxMat29
  w_80_concat_15 : mxxMat30
  w_80_concat_16 : mxxMat31
  w_80_concat_17 : mxxMat32
  w_80_concat_18 : mxxMat33
  w_80_concat_19 : mxxMat34
  w_80_concat_20 : mxxMat35
  w_80_concat_21 : mxxMat36
  w_80_concat_22 : mxxMat37
  w_80_concat_23 : mxxMat38
  w_81_0 : mxxMat15
  w_84_0 : mxxMat39
  w_85_0 : mxxMat39
  w_87_0 : mxxMat1
  w_89_0 : mxxMat11
  w_90_0 : mxxMat11
  w_92_0 : mxxMat1
  w_93_0 : mxxMat1
  w_96_0 : mxxMat10
  w_97_0 : mxxMat14
  w_97_concat_1 : mxxMat40
  w_97_concat_2 : mxxMat41
  w_97_concat_3 : mxxMat42
  w_97_concat_4 : mxxMat43
  w_97_concat_5 : mxxMat44
  w_97_concat_6 : mxxMat45
  w_97_concat_7 : mxxMat46
  w_97_concat_8 : mxxMat47
  w_97_concat_9 : mxxMat48
  w_97_concat_10 : mxxMat49
  w_97_concat_11 : mxxMat50
  w_97_concat_12 : mxxMat51
  w_97_concat_13 : mxxMat52
  w_97_concat_14 : mxxMat53
  w_97_concat_15 : mxxMat54
  w_97_concat_16 : mxxMat55
  w_97_concat_17 : mxxMat56
  w_97_concat_18 : mxxMat57
  w_97_concat_19 : mxxMat58
  w_97_concat_20 : mxxMat59
  w_97_concat_21 : mxxMat60
  w_97_concat_22 : mxxMat61
  w_97_concat_23 : mxxMat62
  w_98_0 : mxxMat14
  w_101_0 : mxxMat63
  w_102_0 : mxxMat63
  w_105_0 : mxxMat2

abbrev parallel_generatedRoot_1269.constraints_5 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_53_0 : mxxMat1) (w_97_0 : mxxMat14) (w_97_concat_5 : mxxMat44) (w_97_concat_6 : mxxMat45) (w_97_concat_7 : mxxMat46) (w_97_concat_8 : mxxMat47) (w_97_concat_9 : mxxMat48) (w_97_concat_10 : mxxMat49) (w_97_concat_11 : mxxMat50) (w_97_concat_12 : mxxMat51) (w_97_concat_13 : mxxMat52) (w_97_concat_14 : mxxMat53) (w_97_concat_15 : mxxMat54) (w_97_concat_16 : mxxMat55) (w_97_concat_17 : mxxMat56) (w_97_concat_18 : mxxMat57) (w_97_concat_19 : mxxMat58) (w_97_concat_20 : mxxMat59) (w_97_concat_21 : mxxMat60) (w_97_concat_22 : mxxMat61) (w_97_concat_23 : mxxMat62) (w_98_0 : mxxMat14) (w_100_0 : mxxMat14) (w_101_0 : mxxMat63) (w_102_0 : mxxMat63) (w_104_0 : mxxMat1) (w_105_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  MxxRuntime.concatDiagonal w_97_concat_5 w_45_0 w_97_concat_6 ∧
  MxxRuntime.concatDiagonal w_97_concat_6 w_45_0 w_97_concat_7 ∧
  MxxRuntime.concatDiagonal w_97_concat_7 w_45_0 w_97_concat_8 ∧
  MxxRuntime.concatDiagonal w_97_concat_8 w_45_0 w_97_concat_9 ∧
  MxxRuntime.concatDiagonal w_97_concat_9 w_45_0 w_97_concat_10 ∧
  MxxRuntime.concatDiagonal w_97_concat_10 w_45_0 w_97_concat_11 ∧
  MxxRuntime.concatDiagonal w_97_concat_11 w_45_0 w_97_concat_12 ∧
  MxxRuntime.concatDiagonal w_97_concat_12 w_45_0 w_97_concat_13 ∧
  MxxRuntime.concatDiagonal w_97_concat_13 w_45_0 w_97_concat_14 ∧
  MxxRuntime.concatDiagonal w_97_concat_14 w_45_0 w_97_concat_15 ∧
  MxxRuntime.concatDiagonal w_97_concat_15 w_45_0 w_97_concat_16 ∧
  MxxRuntime.concatDiagonal w_97_concat_16 w_45_0 w_97_concat_17 ∧
  MxxRuntime.concatDiagonal w_97_concat_17 w_45_0 w_97_concat_18 ∧
  MxxRuntime.concatDiagonal w_97_concat_18 w_45_0 w_97_concat_19 ∧
  MxxRuntime.concatDiagonal w_97_concat_19 w_45_0 w_97_concat_20 ∧
  MxxRuntime.concatDiagonal w_97_concat_20 w_45_0 w_97_concat_21 ∧
  MxxRuntime.concatDiagonal w_97_concat_21 w_45_0 w_97_concat_22 ∧
  MxxRuntime.concatDiagonal w_97_concat_22 w_45_0 w_97_concat_23 ∧
  MxxRuntime.concatDiagonal w_97_concat_23 w_45_0 w_97_0 ∧
  MxxRuntime.centeredRebaseRuns w_97_0 w_98_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_100_0 w_101_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_101_0 w_102_0 ∧
  MxxRuntime.concatRows w_53_0 w_104_0 w_105_0 ∧
  outputs = w_105_0

abbrev parallel_generatedRoot_1269.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_45_0 : mxxMat5) (w_95_0 : mxxMat8) (w_77_0 : mxxMat1) (w_80_0 : mxxMat15) (w_80_concat_19 : mxxMat34) (w_80_concat_20 : mxxMat35) (w_80_concat_21 : mxxMat36) (w_80_concat_22 : mxxMat37) (w_80_concat_23 : mxxMat38) (w_81_0 : mxxMat15) (w_83_0 : mxxMat15) (w_84_0 : mxxMat39) (w_85_0 : mxxMat39) (w_86_0 : mxxMat10) (w_87_0 : mxxMat1) (w_88_0 : mxxMat5) (w_89_0 : mxxMat11) (w_90_0 : mxxMat11) (w_91_0 : mxxMat5) (w_92_0 : mxxMat1) (w_93_0 : mxxMat1) (w_96_0 : mxxMat10) (w_97_concat_1 : mxxMat40) (w_97_concat_2 : mxxMat41) (w_97_concat_3 : mxxMat42) (w_97_concat_4 : mxxMat43) (w_97_concat_5 : mxxMat44) : Prop :=
  MxxRuntime.concatDiagonal w_80_concat_19 w_27_0 w_80_concat_20 ∧
  MxxRuntime.concatDiagonal w_80_concat_20 w_27_0 w_80_concat_21 ∧
  MxxRuntime.concatDiagonal w_80_concat_21 w_27_0 w_80_concat_22 ∧
  MxxRuntime.concatDiagonal w_80_concat_22 w_27_0 w_80_concat_23 ∧
  MxxRuntime.concatDiagonal w_80_concat_23 w_27_0 w_80_0 ∧
  MxxRuntime.centeredRebaseRuns w_80_0 w_81_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_83_0 w_84_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_84_0 w_85_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_86_0 0 1 0 25 w_87_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_88_0 w_89_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_89_0 w_90_0 ∧
  MxxRuntime.tensorRuns w_77_0 w_91_0 w_92_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1269_93 backend hashModel params i_0 i (current, (w_86_0, (w_90_0, ()))) next) (Int.toNat (24)) w_92_0 w_93_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1250 + (3125 * (Int.ofNat i_0))) ∧
  (1250 + (3125 * (Int.ofNat i_0))) < (1875 + (3125 * (Int.ofNat i_0))) ∧
  (1875 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_95_0 0 1 (1250 + (3125 * (Int.ofNat i_0))) (1875 + (3125 * (Int.ofNat i_0))) w_96_0 ∧
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_97_concat_1 ∧
  MxxRuntime.concatDiagonal w_97_concat_1 w_45_0 w_97_concat_2 ∧
  MxxRuntime.concatDiagonal w_97_concat_2 w_45_0 w_97_concat_3 ∧
  MxxRuntime.concatDiagonal w_97_concat_3 w_45_0 w_97_concat_4 ∧
  MxxRuntime.concatDiagonal w_97_concat_4 w_45_0 w_97_concat_5

abbrev parallel_generatedRoot_1269.constraints_3 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_27_0 : mxxMat1) (w_78_0 : mxxMat6) (w_64_0 : mxxMat1) (w_65_0 : mxxMat67) (w_69_0 : mxxMat67) (w_70_0 : mxxMat1) (w_72_0 : mxxMat1) (w_73_0 : mxxMat14) (w_74_0 : mxxMat14) (w_79_0 : mxxMat10) (w_80_concat_1 : mxxMat16) (w_80_concat_2 : mxxMat17) (w_80_concat_3 : mxxMat18) (w_80_concat_4 : mxxMat19) (w_80_concat_5 : mxxMat20) (w_80_concat_6 : mxxMat21) (w_80_concat_7 : mxxMat22) (w_80_concat_8 : mxxMat23) (w_80_concat_9 : mxxMat24) (w_80_concat_10 : mxxMat25) (w_80_concat_11 : mxxMat26) (w_80_concat_12 : mxxMat27) (w_80_concat_13 : mxxMat28) (w_80_concat_14 : mxxMat29) (w_80_concat_15 : mxxMat30) (w_80_concat_16 : mxxMat31) (w_80_concat_17 : mxxMat32) (w_80_concat_18 : mxxMat33) (w_80_concat_19 : mxxMat34) : Prop :=
  MxxRuntime.modulusReduceRuns w_64_0 w_65_0 ∧
  MxxRuntime.centeredRebaseRuns w_69_0 w_70_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_72_0 w_73_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_73_0 w_74_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_79_0 ∧
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_80_concat_1 ∧
  MxxRuntime.concatDiagonal w_80_concat_1 w_27_0 w_80_concat_2 ∧
  MxxRuntime.concatDiagonal w_80_concat_2 w_27_0 w_80_concat_3 ∧
  MxxRuntime.concatDiagonal w_80_concat_3 w_27_0 w_80_concat_4 ∧
  MxxRuntime.concatDiagonal w_80_concat_4 w_27_0 w_80_concat_5 ∧
  MxxRuntime.concatDiagonal w_80_concat_5 w_27_0 w_80_concat_6 ∧
  MxxRuntime.concatDiagonal w_80_concat_6 w_27_0 w_80_concat_7 ∧
  MxxRuntime.concatDiagonal w_80_concat_7 w_27_0 w_80_concat_8 ∧
  MxxRuntime.concatDiagonal w_80_concat_8 w_27_0 w_80_concat_9 ∧
  MxxRuntime.concatDiagonal w_80_concat_9 w_27_0 w_80_concat_10 ∧
  MxxRuntime.concatDiagonal w_80_concat_10 w_27_0 w_80_concat_11 ∧
  MxxRuntime.concatDiagonal w_80_concat_11 w_27_0 w_80_concat_12 ∧
  MxxRuntime.concatDiagonal w_80_concat_12 w_27_0 w_80_concat_13 ∧
  MxxRuntime.concatDiagonal w_80_concat_13 w_27_0 w_80_concat_14 ∧
  MxxRuntime.concatDiagonal w_80_concat_14 w_27_0 w_80_concat_15 ∧
  MxxRuntime.concatDiagonal w_80_concat_15 w_27_0 w_80_concat_16 ∧
  MxxRuntime.concatDiagonal w_80_concat_16 w_27_0 w_80_concat_17 ∧
  MxxRuntime.concatDiagonal w_80_concat_17 w_27_0 w_80_concat_18 ∧
  MxxRuntime.concatDiagonal w_80_concat_18 w_27_0 w_80_concat_19

abbrev parallel_generatedRoot_1269.constraints_2 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_46_0 : mxxMat14) (w_46_concat_1 : mxxMat40) (w_46_concat_2 : mxxMat41) (w_46_concat_3 : mxxMat42) (w_46_concat_4 : mxxMat43) (w_46_concat_5 : mxxMat44) (w_46_concat_6 : mxxMat45) (w_46_concat_7 : mxxMat46) (w_46_concat_8 : mxxMat47) (w_46_concat_9 : mxxMat48) (w_46_concat_10 : mxxMat49) (w_46_concat_11 : mxxMat50) (w_46_concat_12 : mxxMat51) (w_46_concat_13 : mxxMat52) (w_46_concat_14 : mxxMat53) (w_46_concat_15 : mxxMat54) (w_46_concat_16 : mxxMat55) (w_46_concat_17 : mxxMat56) (w_46_concat_18 : mxxMat57) (w_46_concat_19 : mxxMat58) (w_46_concat_20 : mxxMat59) (w_46_concat_21 : mxxMat60) (w_46_concat_22 : mxxMat61) (w_46_concat_23 : mxxMat62) (w_47_0 : mxxMat14) (w_49_0 : mxxMat14) (w_50_0 : mxxMat63) (w_51_0 : mxxMat63) (w_56_0 : mxxMat12) (w_58_0 : mxxMat2) (w_59_0 : mxxMat4) (w_60_0 : mxxMat4) (w_63_0 : mxxMat1) (w_64_0 : mxxMat1) : Prop :=
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_46_concat_1 ∧
  MxxRuntime.concatDiagonal w_46_concat_1 w_45_0 w_46_concat_2 ∧
  MxxRuntime.concatDiagonal w_46_concat_2 w_45_0 w_46_concat_3 ∧
  MxxRuntime.concatDiagonal w_46_concat_3 w_45_0 w_46_concat_4 ∧
  MxxRuntime.concatDiagonal w_46_concat_4 w_45_0 w_46_concat_5 ∧
  MxxRuntime.concatDiagonal w_46_concat_5 w_45_0 w_46_concat_6 ∧
  MxxRuntime.concatDiagonal w_46_concat_6 w_45_0 w_46_concat_7 ∧
  MxxRuntime.concatDiagonal w_46_concat_7 w_45_0 w_46_concat_8 ∧
  MxxRuntime.concatDiagonal w_46_concat_8 w_45_0 w_46_concat_9 ∧
  MxxRuntime.concatDiagonal w_46_concat_9 w_45_0 w_46_concat_10 ∧
  MxxRuntime.concatDiagonal w_46_concat_10 w_45_0 w_46_concat_11 ∧
  MxxRuntime.concatDiagonal w_46_concat_11 w_45_0 w_46_concat_12 ∧
  MxxRuntime.concatDiagonal w_46_concat_12 w_45_0 w_46_concat_13 ∧
  MxxRuntime.concatDiagonal w_46_concat_13 w_45_0 w_46_concat_14 ∧
  MxxRuntime.concatDiagonal w_46_concat_14 w_45_0 w_46_concat_15 ∧
  MxxRuntime.concatDiagonal w_46_concat_15 w_45_0 w_46_concat_16 ∧
  MxxRuntime.concatDiagonal w_46_concat_16 w_45_0 w_46_concat_17 ∧
  MxxRuntime.concatDiagonal w_46_concat_17 w_45_0 w_46_concat_18 ∧
  MxxRuntime.concatDiagonal w_46_concat_18 w_45_0 w_46_concat_19 ∧
  MxxRuntime.concatDiagonal w_46_concat_19 w_45_0 w_46_concat_20 ∧
  MxxRuntime.concatDiagonal w_46_concat_20 w_45_0 w_46_concat_21 ∧
  MxxRuntime.concatDiagonal w_46_concat_21 w_45_0 w_46_concat_22 ∧
  MxxRuntime.concatDiagonal w_46_concat_22 w_45_0 w_46_concat_23 ∧
  MxxRuntime.concatDiagonal w_46_concat_23 w_45_0 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  MxxRuntime.concatColumns w_54_0 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_63_0 ∧
  MxxRuntime.centeredRebaseRuns w_63_0 w_64_0

abbrev parallel_generatedRoot_1269.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_43_0 : mxxMat8) (w_24_0 : mxxMat1) (w_28_0 : mxxMat15) (w_28_concat_14 : mxxMat29) (w_28_concat_15 : mxxMat30) (w_28_concat_16 : mxxMat31) (w_28_concat_17 : mxxMat32) (w_28_concat_18 : mxxMat33) (w_28_concat_19 : mxxMat34) (w_28_concat_20 : mxxMat35) (w_28_concat_21 : mxxMat36) (w_28_concat_22 : mxxMat37) (w_28_concat_23 : mxxMat38) (w_29_0 : mxxMat15) (w_31_0 : mxxMat15) (w_32_0 : mxxMat39) (w_33_0 : mxxMat39) (w_34_0 : mxxMat10) (w_35_0 : mxxMat1) (w_36_0 : mxxMat5) (w_37_0 : mxxMat11) (w_38_0 : mxxMat11) (w_39_0 : mxxMat5) (w_40_0 : mxxMat1) (w_41_0 : mxxMat1) (w_44_0 : mxxMat10) : Prop :=
  MxxRuntime.concatDiagonal w_28_concat_14 w_27_0 w_28_concat_15 ∧
  MxxRuntime.concatDiagonal w_28_concat_15 w_27_0 w_28_concat_16 ∧
  MxxRuntime.concatDiagonal w_28_concat_16 w_27_0 w_28_concat_17 ∧
  MxxRuntime.concatDiagonal w_28_concat_17 w_27_0 w_28_concat_18 ∧
  MxxRuntime.concatDiagonal w_28_concat_18 w_27_0 w_28_concat_19 ∧
  MxxRuntime.concatDiagonal w_28_concat_19 w_27_0 w_28_concat_20 ∧
  MxxRuntime.concatDiagonal w_28_concat_20 w_27_0 w_28_concat_21 ∧
  MxxRuntime.concatDiagonal w_28_concat_21 w_27_0 w_28_concat_22 ∧
  MxxRuntime.concatDiagonal w_28_concat_22 w_27_0 w_28_concat_23 ∧
  MxxRuntime.concatDiagonal w_28_concat_23 w_27_0 w_28_0 ∧
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
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1269_41 backend hashModel params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1250 + (3125 * (Int.ofNat i_0))) ∧
  (1250 + (3125 * (Int.ofNat i_0))) < (1875 + (3125 * (Int.ofNat i_0))) ∧
  (1875 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_43_0 0 1 (1250 + (3125 * (Int.ofNat i_0))) (1875 + (3125 * (Int.ofNat i_0))) w_44_0

abbrev parallel_generatedRoot_1269.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat1) (w_1_0 : mxxMat1) (w_25_0 : mxxMat6) (w_27_0 : mxxMat1) (w_2_0 : mxxMat12) (w_5_0 : mxxMat2) (w_6_0 : mxxMat4) (w_7_0 : mxxMat4) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat67) (w_16_0 : mxxMat67) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat14) (w_21_0 : mxxMat14) (w_26_0 : mxxMat10) (w_28_concat_1 : mxxMat16) (w_28_concat_2 : mxxMat17) (w_28_concat_3 : mxxMat18) (w_28_concat_4 : mxxMat19) (w_28_concat_5 : mxxMat20) (w_28_concat_6 : mxxMat21) (w_28_concat_7 : mxxMat22) (w_28_concat_8 : mxxMat23) (w_28_concat_9 : mxxMat24) (w_28_concat_10 : mxxMat25) (w_28_concat_11 : mxxMat26) (w_28_concat_12 : mxxMat27) (w_28_concat_13 : mxxMat28) (w_28_concat_14 : mxxMat29) : Prop :=
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
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_28_concat_1 ∧
  MxxRuntime.concatDiagonal w_28_concat_1 w_27_0 w_28_concat_2 ∧
  MxxRuntime.concatDiagonal w_28_concat_2 w_27_0 w_28_concat_3 ∧
  MxxRuntime.concatDiagonal w_28_concat_3 w_27_0 w_28_concat_4 ∧
  MxxRuntime.concatDiagonal w_28_concat_4 w_27_0 w_28_concat_5 ∧
  MxxRuntime.concatDiagonal w_28_concat_5 w_27_0 w_28_concat_6 ∧
  MxxRuntime.concatDiagonal w_28_concat_6 w_27_0 w_28_concat_7 ∧
  MxxRuntime.concatDiagonal w_28_concat_7 w_27_0 w_28_concat_8 ∧
  MxxRuntime.concatDiagonal w_28_concat_8 w_27_0 w_28_concat_9 ∧
  MxxRuntime.concatDiagonal w_28_concat_9 w_27_0 w_28_concat_10 ∧
  MxxRuntime.concatDiagonal w_28_concat_10 w_27_0 w_28_concat_11 ∧
  MxxRuntime.concatDiagonal w_28_concat_11 w_27_0 w_28_concat_12 ∧
  MxxRuntime.concatDiagonal w_28_concat_12 w_27_0 w_28_concat_13 ∧
  MxxRuntime.concatDiagonal w_28_concat_13 w_27_0 w_28_concat_14

abbrev parallel_generatedRoot_1269.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) (witness : parallel_generatedRoot_1269.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_4_0 : mxxMat2 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_25_0 : mxxMat6 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_43_0 : mxxMat8 := inputs.2.2.2.2.2.2.1
  let w_45_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_54_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_62_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_78_0 : mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_95_0 : mxxMat8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)
  let w_5_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat68 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat68)
  let w_14_0 : mxxMat67 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat68 := (MxxRuntime.matrixPolynomial [(7254835200)] : mxxMat68)
  let w_16_0 : mxxMat67 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat5)] : mxxMat5)
  let w_19_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat1 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_concat_1 := witness.w_28_concat_1
  let w_28_concat_2 := witness.w_28_concat_2
  let w_28_concat_3 := witness.w_28_concat_3
  let w_28_concat_4 := witness.w_28_concat_4
  let w_28_concat_5 := witness.w_28_concat_5
  let w_28_concat_6 := witness.w_28_concat_6
  let w_28_concat_7 := witness.w_28_concat_7
  let w_28_concat_8 := witness.w_28_concat_8
  let w_28_concat_9 := witness.w_28_concat_9
  let w_28_concat_10 := witness.w_28_concat_10
  let w_28_concat_11 := witness.w_28_concat_11
  let w_28_concat_12 := witness.w_28_concat_12
  let w_28_concat_13 := witness.w_28_concat_13
  let w_28_concat_14 := witness.w_28_concat_14
  let w_28_concat_15 := witness.w_28_concat_15
  let w_28_concat_16 := witness.w_28_concat_16
  let w_28_concat_17 := witness.w_28_concat_17
  let w_28_concat_18 := witness.w_28_concat_18
  let w_28_concat_19 := witness.w_28_concat_19
  let w_28_concat_20 := witness.w_28_concat_20
  let w_28_concat_21 := witness.w_28_concat_21
  let w_28_concat_22 := witness.w_28_concat_22
  let w_28_concat_23 := witness.w_28_concat_23
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_31_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat10 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat5 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_44_0 := witness.w_44_0
  let w_46_0 := witness.w_46_0
  let w_46_concat_1 := witness.w_46_concat_1
  let w_46_concat_2 := witness.w_46_concat_2
  let w_46_concat_3 := witness.w_46_concat_3
  let w_46_concat_4 := witness.w_46_concat_4
  let w_46_concat_5 := witness.w_46_concat_5
  let w_46_concat_6 := witness.w_46_concat_6
  let w_46_concat_7 := witness.w_46_concat_7
  let w_46_concat_8 := witness.w_46_concat_8
  let w_46_concat_9 := witness.w_46_concat_9
  let w_46_concat_10 := witness.w_46_concat_10
  let w_46_concat_11 := witness.w_46_concat_11
  let w_46_concat_12 := witness.w_46_concat_12
  let w_46_concat_13 := witness.w_46_concat_13
  let w_46_concat_14 := witness.w_46_concat_14
  let w_46_concat_15 := witness.w_46_concat_15
  let w_46_concat_16 := witness.w_46_concat_16
  let w_46_concat_17 := witness.w_46_concat_17
  let w_46_concat_18 := witness.w_46_concat_18
  let w_46_concat_19 := witness.w_46_concat_19
  let w_46_concat_20 := witness.w_46_concat_20
  let w_46_concat_21 := witness.w_46_concat_21
  let w_46_concat_22 := witness.w_46_concat_22
  let w_46_concat_23 := witness.w_46_concat_23
  let w_47_0 := witness.w_47_0
  let w_48_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_49_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_47_0 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_44_0 w_51_0
  let w_53_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_52_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_57_0 w_4_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixMul w_56_0 w_60_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : mxxMat68 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat68)
  let w_67_0 : mxxMat67 := MxxRuntime.matrixMulScalarRight w_65_0 w_66_0
  let w_68_0 : mxxMat68 := (MxxRuntime.matrixPolynomial [(7254835200)] : mxxMat68)
  let w_69_0 : mxxMat67 := MxxRuntime.matrixMulScalarRight w_67_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat5)] : mxxMat5)
  let w_72_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_70_0 w_71_0
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 : mxxMat1 := MxxRuntime.matrixMul w_62_0 w_74_0
  let w_76_0 : mxxMat1 := MxxRuntime.matrixSub w_61_0 w_75_0
  let w_77_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_80_concat_1 := witness.w_80_concat_1
  let w_80_concat_2 := witness.w_80_concat_2
  let w_80_concat_3 := witness.w_80_concat_3
  let w_80_concat_4 := witness.w_80_concat_4
  let w_80_concat_5 := witness.w_80_concat_5
  let w_80_concat_6 := witness.w_80_concat_6
  let w_80_concat_7 := witness.w_80_concat_7
  let w_80_concat_8 := witness.w_80_concat_8
  let w_80_concat_9 := witness.w_80_concat_9
  let w_80_concat_10 := witness.w_80_concat_10
  let w_80_concat_11 := witness.w_80_concat_11
  let w_80_concat_12 := witness.w_80_concat_12
  let w_80_concat_13 := witness.w_80_concat_13
  let w_80_concat_14 := witness.w_80_concat_14
  let w_80_concat_15 := witness.w_80_concat_15
  let w_80_concat_16 := witness.w_80_concat_16
  let w_80_concat_17 := witness.w_80_concat_17
  let w_80_concat_18 := witness.w_80_concat_18
  let w_80_concat_19 := witness.w_80_concat_19
  let w_80_concat_20 := witness.w_80_concat_20
  let w_80_concat_21 := witness.w_80_concat_21
  let w_80_concat_22 := witness.w_80_concat_22
  let w_80_concat_23 := witness.w_80_concat_23
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_83_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_81_0 w_82_0
  let w_84_0 := witness.w_84_0
  let w_85_0 := witness.w_85_0
  let w_86_0 : mxxMat10 := MxxRuntime.matrixMul w_79_0 w_85_0
  let w_87_0 := witness.w_87_0
  let w_88_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044254771331362327096896778694705812889646799470123121)] : mxxMat5)
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 : mxxMat5 := MxxRuntime.matrixMul w_87_0 w_90_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 : mxxMat1 := MxxRuntime.matrixAdd w_76_0 w_93_0
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_97_concat_1 := witness.w_97_concat_1
  let w_97_concat_2 := witness.w_97_concat_2
  let w_97_concat_3 := witness.w_97_concat_3
  let w_97_concat_4 := witness.w_97_concat_4
  let w_97_concat_5 := witness.w_97_concat_5
  let w_97_concat_6 := witness.w_97_concat_6
  let w_97_concat_7 := witness.w_97_concat_7
  let w_97_concat_8 := witness.w_97_concat_8
  let w_97_concat_9 := witness.w_97_concat_9
  let w_97_concat_10 := witness.w_97_concat_10
  let w_97_concat_11 := witness.w_97_concat_11
  let w_97_concat_12 := witness.w_97_concat_12
  let w_97_concat_13 := witness.w_97_concat_13
  let w_97_concat_14 := witness.w_97_concat_14
  let w_97_concat_15 := witness.w_97_concat_15
  let w_97_concat_16 := witness.w_97_concat_16
  let w_97_concat_17 := witness.w_97_concat_17
  let w_97_concat_18 := witness.w_97_concat_18
  let w_97_concat_19 := witness.w_97_concat_19
  let w_97_concat_20 := witness.w_97_concat_20
  let w_97_concat_21 := witness.w_97_concat_21
  let w_97_concat_22 := witness.w_97_concat_22
  let w_97_concat_23 := witness.w_97_concat_23
  let w_98_0 := witness.w_98_0
  let w_99_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_100_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_98_0 w_99_0
  let w_101_0 := witness.w_101_0
  let w_102_0 := witness.w_102_0
  let w_103_0 : mxxMat1 := MxxRuntime.matrixMul w_96_0 w_102_0
  let w_104_0 : mxxMat1 := MxxRuntime.matrixAdd w_94_0 w_103_0
  let w_105_0 := witness.w_105_0
  parallel_generatedRoot_1269.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_26_0 w_28_concat_1 w_28_concat_2 w_28_concat_3 w_28_concat_4 w_28_concat_5 w_28_concat_6 w_28_concat_7 w_28_concat_8 w_28_concat_9 w_28_concat_10 w_28_concat_11 w_28_concat_12 w_28_concat_13 w_28_concat_14 ∧
    parallel_generatedRoot_1269.constraints_1 backend hashModel params i_0 w_27_0 w_43_0 w_24_0 w_28_0 w_28_concat_14 w_28_concat_15 w_28_concat_16 w_28_concat_17 w_28_concat_18 w_28_concat_19 w_28_concat_20 w_28_concat_21 w_28_concat_22 w_28_concat_23 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_44_0 ∧
    parallel_generatedRoot_1269.constraints_2 backend w_45_0 w_54_0 w_55_0 w_46_0 w_46_concat_1 w_46_concat_2 w_46_concat_3 w_46_concat_4 w_46_concat_5 w_46_concat_6 w_46_concat_7 w_46_concat_8 w_46_concat_9 w_46_concat_10 w_46_concat_11 w_46_concat_12 w_46_concat_13 w_46_concat_14 w_46_concat_15 w_46_concat_16 w_46_concat_17 w_46_concat_18 w_46_concat_19 w_46_concat_20 w_46_concat_21 w_46_concat_22 w_46_concat_23 w_47_0 w_49_0 w_50_0 w_51_0 w_56_0 w_58_0 w_59_0 w_60_0 w_63_0 w_64_0 ∧
    parallel_generatedRoot_1269.constraints_3 backend i_0 w_27_0 w_78_0 w_64_0 w_65_0 w_69_0 w_70_0 w_72_0 w_73_0 w_74_0 w_79_0 w_80_concat_1 w_80_concat_2 w_80_concat_3 w_80_concat_4 w_80_concat_5 w_80_concat_6 w_80_concat_7 w_80_concat_8 w_80_concat_9 w_80_concat_10 w_80_concat_11 w_80_concat_12 w_80_concat_13 w_80_concat_14 w_80_concat_15 w_80_concat_16 w_80_concat_17 w_80_concat_18 w_80_concat_19 ∧
    parallel_generatedRoot_1269.constraints_4 backend hashModel params i_0 w_27_0 w_45_0 w_95_0 w_77_0 w_80_0 w_80_concat_19 w_80_concat_20 w_80_concat_21 w_80_concat_22 w_80_concat_23 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_96_0 w_97_concat_1 w_97_concat_2 w_97_concat_3 w_97_concat_4 w_97_concat_5 ∧
    parallel_generatedRoot_1269.constraints_5 backend w_45_0 w_53_0 w_97_0 w_97_concat_5 w_97_concat_6 w_97_concat_7 w_97_concat_8 w_97_concat_9 w_97_concat_10 w_97_concat_11 w_97_concat_12 w_97_concat_13 w_97_concat_14 w_97_concat_15 w_97_concat_16 w_97_concat_17 w_97_concat_18 w_97_concat_19 w_97_concat_20 w_97_concat_21 w_97_concat_22 w_97_concat_23 w_98_0 w_100_0 w_101_0 w_102_0 w_104_0 w_105_0 outputs

def parallel_generatedRoot_1269 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) : Prop :=
  ∃ witness : parallel_generatedRoot_1269.Witness,
    parallel_generatedRoot_1269.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_1270_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1270_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1270_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_1270_93.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1270_93 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1270_93.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1270.Witness where
  w_2_0 : mxxMat12
  w_6_0 : mxxMat4
  w_7_0 : mxxMat4
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat69
  w_17_0 : mxxMat1
  w_20_0 : mxxMat14
  w_21_0 : mxxMat14
  w_26_0 : mxxMat10
  w_28_0 : mxxMat15
  w_28_concat_1 : mxxMat16
  w_28_concat_2 : mxxMat17
  w_28_concat_3 : mxxMat18
  w_28_concat_4 : mxxMat19
  w_28_concat_5 : mxxMat20
  w_28_concat_6 : mxxMat21
  w_28_concat_7 : mxxMat22
  w_28_concat_8 : mxxMat23
  w_28_concat_9 : mxxMat24
  w_28_concat_10 : mxxMat25
  w_28_concat_11 : mxxMat26
  w_28_concat_12 : mxxMat27
  w_28_concat_13 : mxxMat28
  w_28_concat_14 : mxxMat29
  w_28_concat_15 : mxxMat30
  w_28_concat_16 : mxxMat31
  w_28_concat_17 : mxxMat32
  w_28_concat_18 : mxxMat33
  w_28_concat_19 : mxxMat34
  w_28_concat_20 : mxxMat35
  w_28_concat_21 : mxxMat36
  w_28_concat_22 : mxxMat37
  w_28_concat_23 : mxxMat38
  w_29_0 : mxxMat15
  w_32_0 : mxxMat39
  w_33_0 : mxxMat39
  w_35_0 : mxxMat1
  w_37_0 : mxxMat11
  w_38_0 : mxxMat11
  w_40_0 : mxxMat1
  w_41_0 : mxxMat1
  w_44_0 : mxxMat10
  w_46_0 : mxxMat14
  w_46_concat_1 : mxxMat40
  w_46_concat_2 : mxxMat41
  w_46_concat_3 : mxxMat42
  w_46_concat_4 : mxxMat43
  w_46_concat_5 : mxxMat44
  w_46_concat_6 : mxxMat45
  w_46_concat_7 : mxxMat46
  w_46_concat_8 : mxxMat47
  w_46_concat_9 : mxxMat48
  w_46_concat_10 : mxxMat49
  w_46_concat_11 : mxxMat50
  w_46_concat_12 : mxxMat51
  w_46_concat_13 : mxxMat52
  w_46_concat_14 : mxxMat53
  w_46_concat_15 : mxxMat54
  w_46_concat_16 : mxxMat55
  w_46_concat_17 : mxxMat56
  w_46_concat_18 : mxxMat57
  w_46_concat_19 : mxxMat58
  w_46_concat_20 : mxxMat59
  w_46_concat_21 : mxxMat60
  w_46_concat_22 : mxxMat61
  w_46_concat_23 : mxxMat62
  w_47_0 : mxxMat14
  w_50_0 : mxxMat63
  w_51_0 : mxxMat63
  w_56_0 : mxxMat12
  w_59_0 : mxxMat4
  w_60_0 : mxxMat4
  w_63_0 : mxxMat1
  w_64_0 : mxxMat1
  w_65_0 : mxxMat69
  w_70_0 : mxxMat1
  w_73_0 : mxxMat14
  w_74_0 : mxxMat14
  w_79_0 : mxxMat10
  w_80_0 : mxxMat15
  w_80_concat_1 : mxxMat16
  w_80_concat_2 : mxxMat17
  w_80_concat_3 : mxxMat18
  w_80_concat_4 : mxxMat19
  w_80_concat_5 : mxxMat20
  w_80_concat_6 : mxxMat21
  w_80_concat_7 : mxxMat22
  w_80_concat_8 : mxxMat23
  w_80_concat_9 : mxxMat24
  w_80_concat_10 : mxxMat25
  w_80_concat_11 : mxxMat26
  w_80_concat_12 : mxxMat27
  w_80_concat_13 : mxxMat28
  w_80_concat_14 : mxxMat29
  w_80_concat_15 : mxxMat30
  w_80_concat_16 : mxxMat31
  w_80_concat_17 : mxxMat32
  w_80_concat_18 : mxxMat33
  w_80_concat_19 : mxxMat34
  w_80_concat_20 : mxxMat35
  w_80_concat_21 : mxxMat36
  w_80_concat_22 : mxxMat37
  w_80_concat_23 : mxxMat38
  w_81_0 : mxxMat15
  w_84_0 : mxxMat39
  w_85_0 : mxxMat39
  w_87_0 : mxxMat1
  w_89_0 : mxxMat11
  w_90_0 : mxxMat11
  w_92_0 : mxxMat1
  w_93_0 : mxxMat1
  w_96_0 : mxxMat10
  w_97_0 : mxxMat14
  w_97_concat_1 : mxxMat40
  w_97_concat_2 : mxxMat41
  w_97_concat_3 : mxxMat42
  w_97_concat_4 : mxxMat43
  w_97_concat_5 : mxxMat44
  w_97_concat_6 : mxxMat45
  w_97_concat_7 : mxxMat46
  w_97_concat_8 : mxxMat47
  w_97_concat_9 : mxxMat48
  w_97_concat_10 : mxxMat49
  w_97_concat_11 : mxxMat50
  w_97_concat_12 : mxxMat51
  w_97_concat_13 : mxxMat52
  w_97_concat_14 : mxxMat53
  w_97_concat_15 : mxxMat54
  w_97_concat_16 : mxxMat55
  w_97_concat_17 : mxxMat56
  w_97_concat_18 : mxxMat57
  w_97_concat_19 : mxxMat58
  w_97_concat_20 : mxxMat59
  w_97_concat_21 : mxxMat60
  w_97_concat_22 : mxxMat61
  w_97_concat_23 : mxxMat62
  w_98_0 : mxxMat14
  w_101_0 : mxxMat63
  w_102_0 : mxxMat63
  w_105_0 : mxxMat2

abbrev parallel_generatedRoot_1270.constraints_5 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_53_0 : mxxMat1) (w_97_0 : mxxMat14) (w_97_concat_5 : mxxMat44) (w_97_concat_6 : mxxMat45) (w_97_concat_7 : mxxMat46) (w_97_concat_8 : mxxMat47) (w_97_concat_9 : mxxMat48) (w_97_concat_10 : mxxMat49) (w_97_concat_11 : mxxMat50) (w_97_concat_12 : mxxMat51) (w_97_concat_13 : mxxMat52) (w_97_concat_14 : mxxMat53) (w_97_concat_15 : mxxMat54) (w_97_concat_16 : mxxMat55) (w_97_concat_17 : mxxMat56) (w_97_concat_18 : mxxMat57) (w_97_concat_19 : mxxMat58) (w_97_concat_20 : mxxMat59) (w_97_concat_21 : mxxMat60) (w_97_concat_22 : mxxMat61) (w_97_concat_23 : mxxMat62) (w_98_0 : mxxMat14) (w_100_0 : mxxMat14) (w_101_0 : mxxMat63) (w_102_0 : mxxMat63) (w_104_0 : mxxMat1) (w_105_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  MxxRuntime.concatDiagonal w_97_concat_5 w_45_0 w_97_concat_6 ∧
  MxxRuntime.concatDiagonal w_97_concat_6 w_45_0 w_97_concat_7 ∧
  MxxRuntime.concatDiagonal w_97_concat_7 w_45_0 w_97_concat_8 ∧
  MxxRuntime.concatDiagonal w_97_concat_8 w_45_0 w_97_concat_9 ∧
  MxxRuntime.concatDiagonal w_97_concat_9 w_45_0 w_97_concat_10 ∧
  MxxRuntime.concatDiagonal w_97_concat_10 w_45_0 w_97_concat_11 ∧
  MxxRuntime.concatDiagonal w_97_concat_11 w_45_0 w_97_concat_12 ∧
  MxxRuntime.concatDiagonal w_97_concat_12 w_45_0 w_97_concat_13 ∧
  MxxRuntime.concatDiagonal w_97_concat_13 w_45_0 w_97_concat_14 ∧
  MxxRuntime.concatDiagonal w_97_concat_14 w_45_0 w_97_concat_15 ∧
  MxxRuntime.concatDiagonal w_97_concat_15 w_45_0 w_97_concat_16 ∧
  MxxRuntime.concatDiagonal w_97_concat_16 w_45_0 w_97_concat_17 ∧
  MxxRuntime.concatDiagonal w_97_concat_17 w_45_0 w_97_concat_18 ∧
  MxxRuntime.concatDiagonal w_97_concat_18 w_45_0 w_97_concat_19 ∧
  MxxRuntime.concatDiagonal w_97_concat_19 w_45_0 w_97_concat_20 ∧
  MxxRuntime.concatDiagonal w_97_concat_20 w_45_0 w_97_concat_21 ∧
  MxxRuntime.concatDiagonal w_97_concat_21 w_45_0 w_97_concat_22 ∧
  MxxRuntime.concatDiagonal w_97_concat_22 w_45_0 w_97_concat_23 ∧
  MxxRuntime.concatDiagonal w_97_concat_23 w_45_0 w_97_0 ∧
  MxxRuntime.centeredRebaseRuns w_97_0 w_98_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_100_0 w_101_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_101_0 w_102_0 ∧
  MxxRuntime.concatRows w_53_0 w_104_0 w_105_0 ∧
  outputs = w_105_0

abbrev parallel_generatedRoot_1270.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_45_0 : mxxMat5) (w_95_0 : mxxMat8) (w_77_0 : mxxMat1) (w_80_0 : mxxMat15) (w_80_concat_19 : mxxMat34) (w_80_concat_20 : mxxMat35) (w_80_concat_21 : mxxMat36) (w_80_concat_22 : mxxMat37) (w_80_concat_23 : mxxMat38) (w_81_0 : mxxMat15) (w_83_0 : mxxMat15) (w_84_0 : mxxMat39) (w_85_0 : mxxMat39) (w_86_0 : mxxMat10) (w_87_0 : mxxMat1) (w_88_0 : mxxMat5) (w_89_0 : mxxMat11) (w_90_0 : mxxMat11) (w_91_0 : mxxMat5) (w_92_0 : mxxMat1) (w_93_0 : mxxMat1) (w_96_0 : mxxMat10) (w_97_concat_1 : mxxMat40) (w_97_concat_2 : mxxMat41) (w_97_concat_3 : mxxMat42) (w_97_concat_4 : mxxMat43) (w_97_concat_5 : mxxMat44) : Prop :=
  MxxRuntime.concatDiagonal w_80_concat_19 w_27_0 w_80_concat_20 ∧
  MxxRuntime.concatDiagonal w_80_concat_20 w_27_0 w_80_concat_21 ∧
  MxxRuntime.concatDiagonal w_80_concat_21 w_27_0 w_80_concat_22 ∧
  MxxRuntime.concatDiagonal w_80_concat_22 w_27_0 w_80_concat_23 ∧
  MxxRuntime.concatDiagonal w_80_concat_23 w_27_0 w_80_0 ∧
  MxxRuntime.centeredRebaseRuns w_80_0 w_81_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_83_0 w_84_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_84_0 w_85_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_86_0 0 1 0 25 w_87_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_88_0 w_89_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_89_0 w_90_0 ∧
  MxxRuntime.tensorRuns w_77_0 w_91_0 w_92_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1270_93 backend hashModel params i_0 i (current, (w_86_0, (w_90_0, ()))) next) (Int.toNat (24)) w_92_0 w_93_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1875 + (3125 * (Int.ofNat i_0))) ∧
  (1875 + (3125 * (Int.ofNat i_0))) < (2500 + (3125 * (Int.ofNat i_0))) ∧
  (2500 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_95_0 0 1 (1875 + (3125 * (Int.ofNat i_0))) (2500 + (3125 * (Int.ofNat i_0))) w_96_0 ∧
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_97_concat_1 ∧
  MxxRuntime.concatDiagonal w_97_concat_1 w_45_0 w_97_concat_2 ∧
  MxxRuntime.concatDiagonal w_97_concat_2 w_45_0 w_97_concat_3 ∧
  MxxRuntime.concatDiagonal w_97_concat_3 w_45_0 w_97_concat_4 ∧
  MxxRuntime.concatDiagonal w_97_concat_4 w_45_0 w_97_concat_5

abbrev parallel_generatedRoot_1270.constraints_3 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_27_0 : mxxMat1) (w_78_0 : mxxMat6) (w_64_0 : mxxMat1) (w_65_0 : mxxMat69) (w_69_0 : mxxMat69) (w_70_0 : mxxMat1) (w_72_0 : mxxMat1) (w_73_0 : mxxMat14) (w_74_0 : mxxMat14) (w_79_0 : mxxMat10) (w_80_concat_1 : mxxMat16) (w_80_concat_2 : mxxMat17) (w_80_concat_3 : mxxMat18) (w_80_concat_4 : mxxMat19) (w_80_concat_5 : mxxMat20) (w_80_concat_6 : mxxMat21) (w_80_concat_7 : mxxMat22) (w_80_concat_8 : mxxMat23) (w_80_concat_9 : mxxMat24) (w_80_concat_10 : mxxMat25) (w_80_concat_11 : mxxMat26) (w_80_concat_12 : mxxMat27) (w_80_concat_13 : mxxMat28) (w_80_concat_14 : mxxMat29) (w_80_concat_15 : mxxMat30) (w_80_concat_16 : mxxMat31) (w_80_concat_17 : mxxMat32) (w_80_concat_18 : mxxMat33) (w_80_concat_19 : mxxMat34) : Prop :=
  MxxRuntime.modulusReduceRuns w_64_0 w_65_0 ∧
  MxxRuntime.centeredRebaseRuns w_69_0 w_70_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_72_0 w_73_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_73_0 w_74_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_79_0 ∧
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_80_concat_1 ∧
  MxxRuntime.concatDiagonal w_80_concat_1 w_27_0 w_80_concat_2 ∧
  MxxRuntime.concatDiagonal w_80_concat_2 w_27_0 w_80_concat_3 ∧
  MxxRuntime.concatDiagonal w_80_concat_3 w_27_0 w_80_concat_4 ∧
  MxxRuntime.concatDiagonal w_80_concat_4 w_27_0 w_80_concat_5 ∧
  MxxRuntime.concatDiagonal w_80_concat_5 w_27_0 w_80_concat_6 ∧
  MxxRuntime.concatDiagonal w_80_concat_6 w_27_0 w_80_concat_7 ∧
  MxxRuntime.concatDiagonal w_80_concat_7 w_27_0 w_80_concat_8 ∧
  MxxRuntime.concatDiagonal w_80_concat_8 w_27_0 w_80_concat_9 ∧
  MxxRuntime.concatDiagonal w_80_concat_9 w_27_0 w_80_concat_10 ∧
  MxxRuntime.concatDiagonal w_80_concat_10 w_27_0 w_80_concat_11 ∧
  MxxRuntime.concatDiagonal w_80_concat_11 w_27_0 w_80_concat_12 ∧
  MxxRuntime.concatDiagonal w_80_concat_12 w_27_0 w_80_concat_13 ∧
  MxxRuntime.concatDiagonal w_80_concat_13 w_27_0 w_80_concat_14 ∧
  MxxRuntime.concatDiagonal w_80_concat_14 w_27_0 w_80_concat_15 ∧
  MxxRuntime.concatDiagonal w_80_concat_15 w_27_0 w_80_concat_16 ∧
  MxxRuntime.concatDiagonal w_80_concat_16 w_27_0 w_80_concat_17 ∧
  MxxRuntime.concatDiagonal w_80_concat_17 w_27_0 w_80_concat_18 ∧
  MxxRuntime.concatDiagonal w_80_concat_18 w_27_0 w_80_concat_19

abbrev parallel_generatedRoot_1270.constraints_2 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_46_0 : mxxMat14) (w_46_concat_1 : mxxMat40) (w_46_concat_2 : mxxMat41) (w_46_concat_3 : mxxMat42) (w_46_concat_4 : mxxMat43) (w_46_concat_5 : mxxMat44) (w_46_concat_6 : mxxMat45) (w_46_concat_7 : mxxMat46) (w_46_concat_8 : mxxMat47) (w_46_concat_9 : mxxMat48) (w_46_concat_10 : mxxMat49) (w_46_concat_11 : mxxMat50) (w_46_concat_12 : mxxMat51) (w_46_concat_13 : mxxMat52) (w_46_concat_14 : mxxMat53) (w_46_concat_15 : mxxMat54) (w_46_concat_16 : mxxMat55) (w_46_concat_17 : mxxMat56) (w_46_concat_18 : mxxMat57) (w_46_concat_19 : mxxMat58) (w_46_concat_20 : mxxMat59) (w_46_concat_21 : mxxMat60) (w_46_concat_22 : mxxMat61) (w_46_concat_23 : mxxMat62) (w_47_0 : mxxMat14) (w_49_0 : mxxMat14) (w_50_0 : mxxMat63) (w_51_0 : mxxMat63) (w_56_0 : mxxMat12) (w_58_0 : mxxMat2) (w_59_0 : mxxMat4) (w_60_0 : mxxMat4) (w_63_0 : mxxMat1) (w_64_0 : mxxMat1) : Prop :=
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_46_concat_1 ∧
  MxxRuntime.concatDiagonal w_46_concat_1 w_45_0 w_46_concat_2 ∧
  MxxRuntime.concatDiagonal w_46_concat_2 w_45_0 w_46_concat_3 ∧
  MxxRuntime.concatDiagonal w_46_concat_3 w_45_0 w_46_concat_4 ∧
  MxxRuntime.concatDiagonal w_46_concat_4 w_45_0 w_46_concat_5 ∧
  MxxRuntime.concatDiagonal w_46_concat_5 w_45_0 w_46_concat_6 ∧
  MxxRuntime.concatDiagonal w_46_concat_6 w_45_0 w_46_concat_7 ∧
  MxxRuntime.concatDiagonal w_46_concat_7 w_45_0 w_46_concat_8 ∧
  MxxRuntime.concatDiagonal w_46_concat_8 w_45_0 w_46_concat_9 ∧
  MxxRuntime.concatDiagonal w_46_concat_9 w_45_0 w_46_concat_10 ∧
  MxxRuntime.concatDiagonal w_46_concat_10 w_45_0 w_46_concat_11 ∧
  MxxRuntime.concatDiagonal w_46_concat_11 w_45_0 w_46_concat_12 ∧
  MxxRuntime.concatDiagonal w_46_concat_12 w_45_0 w_46_concat_13 ∧
  MxxRuntime.concatDiagonal w_46_concat_13 w_45_0 w_46_concat_14 ∧
  MxxRuntime.concatDiagonal w_46_concat_14 w_45_0 w_46_concat_15 ∧
  MxxRuntime.concatDiagonal w_46_concat_15 w_45_0 w_46_concat_16 ∧
  MxxRuntime.concatDiagonal w_46_concat_16 w_45_0 w_46_concat_17 ∧
  MxxRuntime.concatDiagonal w_46_concat_17 w_45_0 w_46_concat_18 ∧
  MxxRuntime.concatDiagonal w_46_concat_18 w_45_0 w_46_concat_19 ∧
  MxxRuntime.concatDiagonal w_46_concat_19 w_45_0 w_46_concat_20 ∧
  MxxRuntime.concatDiagonal w_46_concat_20 w_45_0 w_46_concat_21 ∧
  MxxRuntime.concatDiagonal w_46_concat_21 w_45_0 w_46_concat_22 ∧
  MxxRuntime.concatDiagonal w_46_concat_22 w_45_0 w_46_concat_23 ∧
  MxxRuntime.concatDiagonal w_46_concat_23 w_45_0 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  MxxRuntime.concatColumns w_54_0 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_63_0 ∧
  MxxRuntime.centeredRebaseRuns w_63_0 w_64_0

abbrev parallel_generatedRoot_1270.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_43_0 : mxxMat8) (w_24_0 : mxxMat1) (w_28_0 : mxxMat15) (w_28_concat_14 : mxxMat29) (w_28_concat_15 : mxxMat30) (w_28_concat_16 : mxxMat31) (w_28_concat_17 : mxxMat32) (w_28_concat_18 : mxxMat33) (w_28_concat_19 : mxxMat34) (w_28_concat_20 : mxxMat35) (w_28_concat_21 : mxxMat36) (w_28_concat_22 : mxxMat37) (w_28_concat_23 : mxxMat38) (w_29_0 : mxxMat15) (w_31_0 : mxxMat15) (w_32_0 : mxxMat39) (w_33_0 : mxxMat39) (w_34_0 : mxxMat10) (w_35_0 : mxxMat1) (w_36_0 : mxxMat5) (w_37_0 : mxxMat11) (w_38_0 : mxxMat11) (w_39_0 : mxxMat5) (w_40_0 : mxxMat1) (w_41_0 : mxxMat1) (w_44_0 : mxxMat10) : Prop :=
  MxxRuntime.concatDiagonal w_28_concat_14 w_27_0 w_28_concat_15 ∧
  MxxRuntime.concatDiagonal w_28_concat_15 w_27_0 w_28_concat_16 ∧
  MxxRuntime.concatDiagonal w_28_concat_16 w_27_0 w_28_concat_17 ∧
  MxxRuntime.concatDiagonal w_28_concat_17 w_27_0 w_28_concat_18 ∧
  MxxRuntime.concatDiagonal w_28_concat_18 w_27_0 w_28_concat_19 ∧
  MxxRuntime.concatDiagonal w_28_concat_19 w_27_0 w_28_concat_20 ∧
  MxxRuntime.concatDiagonal w_28_concat_20 w_27_0 w_28_concat_21 ∧
  MxxRuntime.concatDiagonal w_28_concat_21 w_27_0 w_28_concat_22 ∧
  MxxRuntime.concatDiagonal w_28_concat_22 w_27_0 w_28_concat_23 ∧
  MxxRuntime.concatDiagonal w_28_concat_23 w_27_0 w_28_0 ∧
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
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1270_41 backend hashModel params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1875 + (3125 * (Int.ofNat i_0))) ∧
  (1875 + (3125 * (Int.ofNat i_0))) < (2500 + (3125 * (Int.ofNat i_0))) ∧
  (2500 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_43_0 0 1 (1875 + (3125 * (Int.ofNat i_0))) (2500 + (3125 * (Int.ofNat i_0))) w_44_0

abbrev parallel_generatedRoot_1270.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat1) (w_1_0 : mxxMat1) (w_25_0 : mxxMat6) (w_27_0 : mxxMat1) (w_2_0 : mxxMat12) (w_5_0 : mxxMat2) (w_6_0 : mxxMat4) (w_7_0 : mxxMat4) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat69) (w_16_0 : mxxMat69) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat14) (w_21_0 : mxxMat14) (w_26_0 : mxxMat10) (w_28_concat_1 : mxxMat16) (w_28_concat_2 : mxxMat17) (w_28_concat_3 : mxxMat18) (w_28_concat_4 : mxxMat19) (w_28_concat_5 : mxxMat20) (w_28_concat_6 : mxxMat21) (w_28_concat_7 : mxxMat22) (w_28_concat_8 : mxxMat23) (w_28_concat_9 : mxxMat24) (w_28_concat_10 : mxxMat25) (w_28_concat_11 : mxxMat26) (w_28_concat_12 : mxxMat27) (w_28_concat_13 : mxxMat28) (w_28_concat_14 : mxxMat29) : Prop :=
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
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_28_concat_1 ∧
  MxxRuntime.concatDiagonal w_28_concat_1 w_27_0 w_28_concat_2 ∧
  MxxRuntime.concatDiagonal w_28_concat_2 w_27_0 w_28_concat_3 ∧
  MxxRuntime.concatDiagonal w_28_concat_3 w_27_0 w_28_concat_4 ∧
  MxxRuntime.concatDiagonal w_28_concat_4 w_27_0 w_28_concat_5 ∧
  MxxRuntime.concatDiagonal w_28_concat_5 w_27_0 w_28_concat_6 ∧
  MxxRuntime.concatDiagonal w_28_concat_6 w_27_0 w_28_concat_7 ∧
  MxxRuntime.concatDiagonal w_28_concat_7 w_27_0 w_28_concat_8 ∧
  MxxRuntime.concatDiagonal w_28_concat_8 w_27_0 w_28_concat_9 ∧
  MxxRuntime.concatDiagonal w_28_concat_9 w_27_0 w_28_concat_10 ∧
  MxxRuntime.concatDiagonal w_28_concat_10 w_27_0 w_28_concat_11 ∧
  MxxRuntime.concatDiagonal w_28_concat_11 w_27_0 w_28_concat_12 ∧
  MxxRuntime.concatDiagonal w_28_concat_12 w_27_0 w_28_concat_13 ∧
  MxxRuntime.concatDiagonal w_28_concat_13 w_27_0 w_28_concat_14

abbrev parallel_generatedRoot_1270.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) (witness : parallel_generatedRoot_1270.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_4_0 : mxxMat2 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_25_0 : mxxMat6 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_43_0 : mxxMat8 := inputs.2.2.2.2.2.2.1
  let w_45_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_54_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_62_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_78_0 : mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_95_0 : mxxMat8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)
  let w_5_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat70 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat70)
  let w_14_0 : mxxMat69 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat70 := (MxxRuntime.matrixPolynomial [(1125871968844913)] : mxxMat70)
  let w_16_0 : mxxMat69 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat6)] : mxxMat5)
  let w_19_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat1 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_concat_1 := witness.w_28_concat_1
  let w_28_concat_2 := witness.w_28_concat_2
  let w_28_concat_3 := witness.w_28_concat_3
  let w_28_concat_4 := witness.w_28_concat_4
  let w_28_concat_5 := witness.w_28_concat_5
  let w_28_concat_6 := witness.w_28_concat_6
  let w_28_concat_7 := witness.w_28_concat_7
  let w_28_concat_8 := witness.w_28_concat_8
  let w_28_concat_9 := witness.w_28_concat_9
  let w_28_concat_10 := witness.w_28_concat_10
  let w_28_concat_11 := witness.w_28_concat_11
  let w_28_concat_12 := witness.w_28_concat_12
  let w_28_concat_13 := witness.w_28_concat_13
  let w_28_concat_14 := witness.w_28_concat_14
  let w_28_concat_15 := witness.w_28_concat_15
  let w_28_concat_16 := witness.w_28_concat_16
  let w_28_concat_17 := witness.w_28_concat_17
  let w_28_concat_18 := witness.w_28_concat_18
  let w_28_concat_19 := witness.w_28_concat_19
  let w_28_concat_20 := witness.w_28_concat_20
  let w_28_concat_21 := witness.w_28_concat_21
  let w_28_concat_22 := witness.w_28_concat_22
  let w_28_concat_23 := witness.w_28_concat_23
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_31_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat10 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat5 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_44_0 := witness.w_44_0
  let w_46_0 := witness.w_46_0
  let w_46_concat_1 := witness.w_46_concat_1
  let w_46_concat_2 := witness.w_46_concat_2
  let w_46_concat_3 := witness.w_46_concat_3
  let w_46_concat_4 := witness.w_46_concat_4
  let w_46_concat_5 := witness.w_46_concat_5
  let w_46_concat_6 := witness.w_46_concat_6
  let w_46_concat_7 := witness.w_46_concat_7
  let w_46_concat_8 := witness.w_46_concat_8
  let w_46_concat_9 := witness.w_46_concat_9
  let w_46_concat_10 := witness.w_46_concat_10
  let w_46_concat_11 := witness.w_46_concat_11
  let w_46_concat_12 := witness.w_46_concat_12
  let w_46_concat_13 := witness.w_46_concat_13
  let w_46_concat_14 := witness.w_46_concat_14
  let w_46_concat_15 := witness.w_46_concat_15
  let w_46_concat_16 := witness.w_46_concat_16
  let w_46_concat_17 := witness.w_46_concat_17
  let w_46_concat_18 := witness.w_46_concat_18
  let w_46_concat_19 := witness.w_46_concat_19
  let w_46_concat_20 := witness.w_46_concat_20
  let w_46_concat_21 := witness.w_46_concat_21
  let w_46_concat_22 := witness.w_46_concat_22
  let w_46_concat_23 := witness.w_46_concat_23
  let w_47_0 := witness.w_47_0
  let w_48_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_49_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_47_0 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_44_0 w_51_0
  let w_53_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_52_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_57_0 w_4_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixMul w_56_0 w_60_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : mxxMat70 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat70)
  let w_67_0 : mxxMat69 := MxxRuntime.matrixMulScalarRight w_65_0 w_66_0
  let w_68_0 : mxxMat70 := (MxxRuntime.matrixPolynomial [(1125871968844913)] : mxxMat70)
  let w_69_0 : mxxMat69 := MxxRuntime.matrixMulScalarRight w_67_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat6)] : mxxMat5)
  let w_72_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_70_0 w_71_0
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 : mxxMat1 := MxxRuntime.matrixMul w_62_0 w_74_0
  let w_76_0 : mxxMat1 := MxxRuntime.matrixSub w_61_0 w_75_0
  let w_77_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_80_concat_1 := witness.w_80_concat_1
  let w_80_concat_2 := witness.w_80_concat_2
  let w_80_concat_3 := witness.w_80_concat_3
  let w_80_concat_4 := witness.w_80_concat_4
  let w_80_concat_5 := witness.w_80_concat_5
  let w_80_concat_6 := witness.w_80_concat_6
  let w_80_concat_7 := witness.w_80_concat_7
  let w_80_concat_8 := witness.w_80_concat_8
  let w_80_concat_9 := witness.w_80_concat_9
  let w_80_concat_10 := witness.w_80_concat_10
  let w_80_concat_11 := witness.w_80_concat_11
  let w_80_concat_12 := witness.w_80_concat_12
  let w_80_concat_13 := witness.w_80_concat_13
  let w_80_concat_14 := witness.w_80_concat_14
  let w_80_concat_15 := witness.w_80_concat_15
  let w_80_concat_16 := witness.w_80_concat_16
  let w_80_concat_17 := witness.w_80_concat_17
  let w_80_concat_18 := witness.w_80_concat_18
  let w_80_concat_19 := witness.w_80_concat_19
  let w_80_concat_20 := witness.w_80_concat_20
  let w_80_concat_21 := witness.w_80_concat_21
  let w_80_concat_22 := witness.w_80_concat_22
  let w_80_concat_23 := witness.w_80_concat_23
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_83_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_81_0 w_82_0
  let w_84_0 := witness.w_84_0
  let w_85_0 := witness.w_85_0
  let w_86_0 : mxxMat10 := MxxRuntime.matrixMul w_79_0 w_85_0
  let w_87_0 := witness.w_87_0
  let w_88_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044255228050623992174508164557277769211170177223088561)] : mxxMat5)
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 : mxxMat5 := MxxRuntime.matrixMul w_87_0 w_90_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 : mxxMat1 := MxxRuntime.matrixAdd w_76_0 w_93_0
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_97_concat_1 := witness.w_97_concat_1
  let w_97_concat_2 := witness.w_97_concat_2
  let w_97_concat_3 := witness.w_97_concat_3
  let w_97_concat_4 := witness.w_97_concat_4
  let w_97_concat_5 := witness.w_97_concat_5
  let w_97_concat_6 := witness.w_97_concat_6
  let w_97_concat_7 := witness.w_97_concat_7
  let w_97_concat_8 := witness.w_97_concat_8
  let w_97_concat_9 := witness.w_97_concat_9
  let w_97_concat_10 := witness.w_97_concat_10
  let w_97_concat_11 := witness.w_97_concat_11
  let w_97_concat_12 := witness.w_97_concat_12
  let w_97_concat_13 := witness.w_97_concat_13
  let w_97_concat_14 := witness.w_97_concat_14
  let w_97_concat_15 := witness.w_97_concat_15
  let w_97_concat_16 := witness.w_97_concat_16
  let w_97_concat_17 := witness.w_97_concat_17
  let w_97_concat_18 := witness.w_97_concat_18
  let w_97_concat_19 := witness.w_97_concat_19
  let w_97_concat_20 := witness.w_97_concat_20
  let w_97_concat_21 := witness.w_97_concat_21
  let w_97_concat_22 := witness.w_97_concat_22
  let w_97_concat_23 := witness.w_97_concat_23
  let w_98_0 := witness.w_98_0
  let w_99_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_100_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_98_0 w_99_0
  let w_101_0 := witness.w_101_0
  let w_102_0 := witness.w_102_0
  let w_103_0 : mxxMat1 := MxxRuntime.matrixMul w_96_0 w_102_0
  let w_104_0 : mxxMat1 := MxxRuntime.matrixAdd w_94_0 w_103_0
  let w_105_0 := witness.w_105_0
  parallel_generatedRoot_1270.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_26_0 w_28_concat_1 w_28_concat_2 w_28_concat_3 w_28_concat_4 w_28_concat_5 w_28_concat_6 w_28_concat_7 w_28_concat_8 w_28_concat_9 w_28_concat_10 w_28_concat_11 w_28_concat_12 w_28_concat_13 w_28_concat_14 ∧
    parallel_generatedRoot_1270.constraints_1 backend hashModel params i_0 w_27_0 w_43_0 w_24_0 w_28_0 w_28_concat_14 w_28_concat_15 w_28_concat_16 w_28_concat_17 w_28_concat_18 w_28_concat_19 w_28_concat_20 w_28_concat_21 w_28_concat_22 w_28_concat_23 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_44_0 ∧
    parallel_generatedRoot_1270.constraints_2 backend w_45_0 w_54_0 w_55_0 w_46_0 w_46_concat_1 w_46_concat_2 w_46_concat_3 w_46_concat_4 w_46_concat_5 w_46_concat_6 w_46_concat_7 w_46_concat_8 w_46_concat_9 w_46_concat_10 w_46_concat_11 w_46_concat_12 w_46_concat_13 w_46_concat_14 w_46_concat_15 w_46_concat_16 w_46_concat_17 w_46_concat_18 w_46_concat_19 w_46_concat_20 w_46_concat_21 w_46_concat_22 w_46_concat_23 w_47_0 w_49_0 w_50_0 w_51_0 w_56_0 w_58_0 w_59_0 w_60_0 w_63_0 w_64_0 ∧
    parallel_generatedRoot_1270.constraints_3 backend i_0 w_27_0 w_78_0 w_64_0 w_65_0 w_69_0 w_70_0 w_72_0 w_73_0 w_74_0 w_79_0 w_80_concat_1 w_80_concat_2 w_80_concat_3 w_80_concat_4 w_80_concat_5 w_80_concat_6 w_80_concat_7 w_80_concat_8 w_80_concat_9 w_80_concat_10 w_80_concat_11 w_80_concat_12 w_80_concat_13 w_80_concat_14 w_80_concat_15 w_80_concat_16 w_80_concat_17 w_80_concat_18 w_80_concat_19 ∧
    parallel_generatedRoot_1270.constraints_4 backend hashModel params i_0 w_27_0 w_45_0 w_95_0 w_77_0 w_80_0 w_80_concat_19 w_80_concat_20 w_80_concat_21 w_80_concat_22 w_80_concat_23 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_96_0 w_97_concat_1 w_97_concat_2 w_97_concat_3 w_97_concat_4 w_97_concat_5 ∧
    parallel_generatedRoot_1270.constraints_5 backend w_45_0 w_53_0 w_97_0 w_97_concat_5 w_97_concat_6 w_97_concat_7 w_97_concat_8 w_97_concat_9 w_97_concat_10 w_97_concat_11 w_97_concat_12 w_97_concat_13 w_97_concat_14 w_97_concat_15 w_97_concat_16 w_97_concat_17 w_97_concat_18 w_97_concat_19 w_97_concat_20 w_97_concat_21 w_97_concat_22 w_97_concat_23 w_98_0 w_100_0 w_101_0 w_102_0 w_104_0 w_105_0 outputs

def parallel_generatedRoot_1270 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) : Prop :=
  ∃ witness : parallel_generatedRoot_1270.Witness,
    parallel_generatedRoot_1270.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_1271_41.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1271_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1271_41.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

abbrev sequential_parallel_generatedRoot_1271_93.constraints_0 (i_1 : Nat) (w_2_0 : mxxMat10) (w_1_0 : mxxMat1) (w_3_0 : mxxMat1) (w_5_0 : mxxMat5) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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

def sequential_parallel_generatedRoot_1271_93 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat10 × mxxMat11 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat11 := inputs.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_1))) : mxxMat1)
  ∃ (w_3_0 : mxxMat1),
    let w_5_0 : mxxMat5 := MxxRuntime.matrixMul w_3_0 w_4_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_6_0
      sequential_parallel_generatedRoot_1271_93.constraints_0 i_1 w_2_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_1271.Witness where
  w_2_0 : mxxMat12
  w_6_0 : mxxMat4
  w_7_0 : mxxMat4
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat71
  w_17_0 : mxxMat1
  w_20_0 : mxxMat14
  w_21_0 : mxxMat14
  w_26_0 : mxxMat10
  w_28_0 : mxxMat15
  w_28_concat_1 : mxxMat16
  w_28_concat_2 : mxxMat17
  w_28_concat_3 : mxxMat18
  w_28_concat_4 : mxxMat19
  w_28_concat_5 : mxxMat20
  w_28_concat_6 : mxxMat21
  w_28_concat_7 : mxxMat22
  w_28_concat_8 : mxxMat23
  w_28_concat_9 : mxxMat24
  w_28_concat_10 : mxxMat25
  w_28_concat_11 : mxxMat26
  w_28_concat_12 : mxxMat27
  w_28_concat_13 : mxxMat28
  w_28_concat_14 : mxxMat29
  w_28_concat_15 : mxxMat30
  w_28_concat_16 : mxxMat31
  w_28_concat_17 : mxxMat32
  w_28_concat_18 : mxxMat33
  w_28_concat_19 : mxxMat34
  w_28_concat_20 : mxxMat35
  w_28_concat_21 : mxxMat36
  w_28_concat_22 : mxxMat37
  w_28_concat_23 : mxxMat38
  w_29_0 : mxxMat15
  w_32_0 : mxxMat39
  w_33_0 : mxxMat39
  w_35_0 : mxxMat1
  w_37_0 : mxxMat11
  w_38_0 : mxxMat11
  w_40_0 : mxxMat1
  w_41_0 : mxxMat1
  w_44_0 : mxxMat10
  w_46_0 : mxxMat14
  w_46_concat_1 : mxxMat40
  w_46_concat_2 : mxxMat41
  w_46_concat_3 : mxxMat42
  w_46_concat_4 : mxxMat43
  w_46_concat_5 : mxxMat44
  w_46_concat_6 : mxxMat45
  w_46_concat_7 : mxxMat46
  w_46_concat_8 : mxxMat47
  w_46_concat_9 : mxxMat48
  w_46_concat_10 : mxxMat49
  w_46_concat_11 : mxxMat50
  w_46_concat_12 : mxxMat51
  w_46_concat_13 : mxxMat52
  w_46_concat_14 : mxxMat53
  w_46_concat_15 : mxxMat54
  w_46_concat_16 : mxxMat55
  w_46_concat_17 : mxxMat56
  w_46_concat_18 : mxxMat57
  w_46_concat_19 : mxxMat58
  w_46_concat_20 : mxxMat59
  w_46_concat_21 : mxxMat60
  w_46_concat_22 : mxxMat61
  w_46_concat_23 : mxxMat62
  w_47_0 : mxxMat14
  w_50_0 : mxxMat63
  w_51_0 : mxxMat63
  w_56_0 : mxxMat12
  w_59_0 : mxxMat4
  w_60_0 : mxxMat4
  w_63_0 : mxxMat1
  w_64_0 : mxxMat1
  w_65_0 : mxxMat71
  w_70_0 : mxxMat1
  w_73_0 : mxxMat14
  w_74_0 : mxxMat14
  w_79_0 : mxxMat10
  w_80_0 : mxxMat15
  w_80_concat_1 : mxxMat16
  w_80_concat_2 : mxxMat17
  w_80_concat_3 : mxxMat18
  w_80_concat_4 : mxxMat19
  w_80_concat_5 : mxxMat20
  w_80_concat_6 : mxxMat21
  w_80_concat_7 : mxxMat22
  w_80_concat_8 : mxxMat23
  w_80_concat_9 : mxxMat24
  w_80_concat_10 : mxxMat25
  w_80_concat_11 : mxxMat26
  w_80_concat_12 : mxxMat27
  w_80_concat_13 : mxxMat28
  w_80_concat_14 : mxxMat29
  w_80_concat_15 : mxxMat30
  w_80_concat_16 : mxxMat31
  w_80_concat_17 : mxxMat32
  w_80_concat_18 : mxxMat33
  w_80_concat_19 : mxxMat34
  w_80_concat_20 : mxxMat35
  w_80_concat_21 : mxxMat36
  w_80_concat_22 : mxxMat37
  w_80_concat_23 : mxxMat38
  w_81_0 : mxxMat15
  w_84_0 : mxxMat39
  w_85_0 : mxxMat39
  w_87_0 : mxxMat1
  w_89_0 : mxxMat11
  w_90_0 : mxxMat11
  w_92_0 : mxxMat1
  w_93_0 : mxxMat1
  w_96_0 : mxxMat10
  w_97_0 : mxxMat14
  w_97_concat_1 : mxxMat40
  w_97_concat_2 : mxxMat41
  w_97_concat_3 : mxxMat42
  w_97_concat_4 : mxxMat43
  w_97_concat_5 : mxxMat44
  w_97_concat_6 : mxxMat45
  w_97_concat_7 : mxxMat46
  w_97_concat_8 : mxxMat47
  w_97_concat_9 : mxxMat48
  w_97_concat_10 : mxxMat49
  w_97_concat_11 : mxxMat50
  w_97_concat_12 : mxxMat51
  w_97_concat_13 : mxxMat52
  w_97_concat_14 : mxxMat53
  w_97_concat_15 : mxxMat54
  w_97_concat_16 : mxxMat55
  w_97_concat_17 : mxxMat56
  w_97_concat_18 : mxxMat57
  w_97_concat_19 : mxxMat58
  w_97_concat_20 : mxxMat59
  w_97_concat_21 : mxxMat60
  w_97_concat_22 : mxxMat61
  w_97_concat_23 : mxxMat62
  w_98_0 : mxxMat14
  w_101_0 : mxxMat63
  w_102_0 : mxxMat63
  w_105_0 : mxxMat2

abbrev parallel_generatedRoot_1271.constraints_5 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_53_0 : mxxMat1) (w_97_0 : mxxMat14) (w_97_concat_5 : mxxMat44) (w_97_concat_6 : mxxMat45) (w_97_concat_7 : mxxMat46) (w_97_concat_8 : mxxMat47) (w_97_concat_9 : mxxMat48) (w_97_concat_10 : mxxMat49) (w_97_concat_11 : mxxMat50) (w_97_concat_12 : mxxMat51) (w_97_concat_13 : mxxMat52) (w_97_concat_14 : mxxMat53) (w_97_concat_15 : mxxMat54) (w_97_concat_16 : mxxMat55) (w_97_concat_17 : mxxMat56) (w_97_concat_18 : mxxMat57) (w_97_concat_19 : mxxMat58) (w_97_concat_20 : mxxMat59) (w_97_concat_21 : mxxMat60) (w_97_concat_22 : mxxMat61) (w_97_concat_23 : mxxMat62) (w_98_0 : mxxMat14) (w_100_0 : mxxMat14) (w_101_0 : mxxMat63) (w_102_0 : mxxMat63) (w_104_0 : mxxMat1) (w_105_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  MxxRuntime.concatDiagonal w_97_concat_5 w_45_0 w_97_concat_6 ∧
  MxxRuntime.concatDiagonal w_97_concat_6 w_45_0 w_97_concat_7 ∧
  MxxRuntime.concatDiagonal w_97_concat_7 w_45_0 w_97_concat_8 ∧
  MxxRuntime.concatDiagonal w_97_concat_8 w_45_0 w_97_concat_9 ∧
  MxxRuntime.concatDiagonal w_97_concat_9 w_45_0 w_97_concat_10 ∧
  MxxRuntime.concatDiagonal w_97_concat_10 w_45_0 w_97_concat_11 ∧
  MxxRuntime.concatDiagonal w_97_concat_11 w_45_0 w_97_concat_12 ∧
  MxxRuntime.concatDiagonal w_97_concat_12 w_45_0 w_97_concat_13 ∧
  MxxRuntime.concatDiagonal w_97_concat_13 w_45_0 w_97_concat_14 ∧
  MxxRuntime.concatDiagonal w_97_concat_14 w_45_0 w_97_concat_15 ∧
  MxxRuntime.concatDiagonal w_97_concat_15 w_45_0 w_97_concat_16 ∧
  MxxRuntime.concatDiagonal w_97_concat_16 w_45_0 w_97_concat_17 ∧
  MxxRuntime.concatDiagonal w_97_concat_17 w_45_0 w_97_concat_18 ∧
  MxxRuntime.concatDiagonal w_97_concat_18 w_45_0 w_97_concat_19 ∧
  MxxRuntime.concatDiagonal w_97_concat_19 w_45_0 w_97_concat_20 ∧
  MxxRuntime.concatDiagonal w_97_concat_20 w_45_0 w_97_concat_21 ∧
  MxxRuntime.concatDiagonal w_97_concat_21 w_45_0 w_97_concat_22 ∧
  MxxRuntime.concatDiagonal w_97_concat_22 w_45_0 w_97_concat_23 ∧
  MxxRuntime.concatDiagonal w_97_concat_23 w_45_0 w_97_0 ∧
  MxxRuntime.centeredRebaseRuns w_97_0 w_98_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_100_0 w_101_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_101_0 w_102_0 ∧
  MxxRuntime.concatRows w_53_0 w_104_0 w_105_0 ∧
  outputs = w_105_0

abbrev parallel_generatedRoot_1271.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_45_0 : mxxMat5) (w_95_0 : mxxMat8) (w_77_0 : mxxMat1) (w_80_0 : mxxMat15) (w_80_concat_19 : mxxMat34) (w_80_concat_20 : mxxMat35) (w_80_concat_21 : mxxMat36) (w_80_concat_22 : mxxMat37) (w_80_concat_23 : mxxMat38) (w_81_0 : mxxMat15) (w_83_0 : mxxMat15) (w_84_0 : mxxMat39) (w_85_0 : mxxMat39) (w_86_0 : mxxMat10) (w_87_0 : mxxMat1) (w_88_0 : mxxMat5) (w_89_0 : mxxMat11) (w_90_0 : mxxMat11) (w_91_0 : mxxMat5) (w_92_0 : mxxMat1) (w_93_0 : mxxMat1) (w_96_0 : mxxMat10) (w_97_concat_1 : mxxMat40) (w_97_concat_2 : mxxMat41) (w_97_concat_3 : mxxMat42) (w_97_concat_4 : mxxMat43) (w_97_concat_5 : mxxMat44) : Prop :=
  MxxRuntime.concatDiagonal w_80_concat_19 w_27_0 w_80_concat_20 ∧
  MxxRuntime.concatDiagonal w_80_concat_20 w_27_0 w_80_concat_21 ∧
  MxxRuntime.concatDiagonal w_80_concat_21 w_27_0 w_80_concat_22 ∧
  MxxRuntime.concatDiagonal w_80_concat_22 w_27_0 w_80_concat_23 ∧
  MxxRuntime.concatDiagonal w_80_concat_23 w_27_0 w_80_0 ∧
  MxxRuntime.centeredRebaseRuns w_80_0 w_81_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_83_0 w_84_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_84_0 w_85_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 625 ∧
  MxxRuntime.sliceMatrix w_86_0 0 1 0 25 w_87_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_88_0 w_89_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_89_0 w_90_0 ∧
  MxxRuntime.tensorRuns w_77_0 w_91_0 w_92_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1271_93 backend hashModel params i_0 i (current, (w_86_0, (w_90_0, ()))) next) (Int.toNat (24)) w_92_0 w_93_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (2500 + (3125 * (Int.ofNat i_0))) ∧
  (2500 + (3125 * (Int.ofNat i_0))) < (3125 + (3125 * (Int.ofNat i_0))) ∧
  (3125 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_95_0 0 1 (2500 + (3125 * (Int.ofNat i_0))) (3125 + (3125 * (Int.ofNat i_0))) w_96_0 ∧
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_97_concat_1 ∧
  MxxRuntime.concatDiagonal w_97_concat_1 w_45_0 w_97_concat_2 ∧
  MxxRuntime.concatDiagonal w_97_concat_2 w_45_0 w_97_concat_3 ∧
  MxxRuntime.concatDiagonal w_97_concat_3 w_45_0 w_97_concat_4 ∧
  MxxRuntime.concatDiagonal w_97_concat_4 w_45_0 w_97_concat_5

abbrev parallel_generatedRoot_1271.constraints_3 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_27_0 : mxxMat1) (w_78_0 : mxxMat6) (w_64_0 : mxxMat1) (w_65_0 : mxxMat71) (w_69_0 : mxxMat71) (w_70_0 : mxxMat1) (w_72_0 : mxxMat1) (w_73_0 : mxxMat14) (w_74_0 : mxxMat14) (w_79_0 : mxxMat10) (w_80_concat_1 : mxxMat16) (w_80_concat_2 : mxxMat17) (w_80_concat_3 : mxxMat18) (w_80_concat_4 : mxxMat19) (w_80_concat_5 : mxxMat20) (w_80_concat_6 : mxxMat21) (w_80_concat_7 : mxxMat22) (w_80_concat_8 : mxxMat23) (w_80_concat_9 : mxxMat24) (w_80_concat_10 : mxxMat25) (w_80_concat_11 : mxxMat26) (w_80_concat_12 : mxxMat27) (w_80_concat_13 : mxxMat28) (w_80_concat_14 : mxxMat29) (w_80_concat_15 : mxxMat30) (w_80_concat_16 : mxxMat31) (w_80_concat_17 : mxxMat32) (w_80_concat_18 : mxxMat33) (w_80_concat_19 : mxxMat34) : Prop :=
  MxxRuntime.modulusReduceRuns w_64_0 w_65_0 ∧
  MxxRuntime.centeredRebaseRuns w_69_0 w_70_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_72_0 w_73_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_73_0 w_74_0 ∧
  0 ≤ (0) ∧
  (0) < 25 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (625 * (Int.ofNat i_0)) ∧
  (625 * (Int.ofNat i_0)) < (625 + (625 * (Int.ofNat i_0))) ∧
  (625 + (625 * (Int.ofNat i_0))) ≤ 22500 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 (625 * (Int.ofNat i_0)) (625 + (625 * (Int.ofNat i_0))) w_79_0 ∧
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_80_concat_1 ∧
  MxxRuntime.concatDiagonal w_80_concat_1 w_27_0 w_80_concat_2 ∧
  MxxRuntime.concatDiagonal w_80_concat_2 w_27_0 w_80_concat_3 ∧
  MxxRuntime.concatDiagonal w_80_concat_3 w_27_0 w_80_concat_4 ∧
  MxxRuntime.concatDiagonal w_80_concat_4 w_27_0 w_80_concat_5 ∧
  MxxRuntime.concatDiagonal w_80_concat_5 w_27_0 w_80_concat_6 ∧
  MxxRuntime.concatDiagonal w_80_concat_6 w_27_0 w_80_concat_7 ∧
  MxxRuntime.concatDiagonal w_80_concat_7 w_27_0 w_80_concat_8 ∧
  MxxRuntime.concatDiagonal w_80_concat_8 w_27_0 w_80_concat_9 ∧
  MxxRuntime.concatDiagonal w_80_concat_9 w_27_0 w_80_concat_10 ∧
  MxxRuntime.concatDiagonal w_80_concat_10 w_27_0 w_80_concat_11 ∧
  MxxRuntime.concatDiagonal w_80_concat_11 w_27_0 w_80_concat_12 ∧
  MxxRuntime.concatDiagonal w_80_concat_12 w_27_0 w_80_concat_13 ∧
  MxxRuntime.concatDiagonal w_80_concat_13 w_27_0 w_80_concat_14 ∧
  MxxRuntime.concatDiagonal w_80_concat_14 w_27_0 w_80_concat_15 ∧
  MxxRuntime.concatDiagonal w_80_concat_15 w_27_0 w_80_concat_16 ∧
  MxxRuntime.concatDiagonal w_80_concat_16 w_27_0 w_80_concat_17 ∧
  MxxRuntime.concatDiagonal w_80_concat_17 w_27_0 w_80_concat_18 ∧
  MxxRuntime.concatDiagonal w_80_concat_18 w_27_0 w_80_concat_19

abbrev parallel_generatedRoot_1271.constraints_2 (backend : MxxRuntime.BackendContext) (w_45_0 : mxxMat5) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_46_0 : mxxMat14) (w_46_concat_1 : mxxMat40) (w_46_concat_2 : mxxMat41) (w_46_concat_3 : mxxMat42) (w_46_concat_4 : mxxMat43) (w_46_concat_5 : mxxMat44) (w_46_concat_6 : mxxMat45) (w_46_concat_7 : mxxMat46) (w_46_concat_8 : mxxMat47) (w_46_concat_9 : mxxMat48) (w_46_concat_10 : mxxMat49) (w_46_concat_11 : mxxMat50) (w_46_concat_12 : mxxMat51) (w_46_concat_13 : mxxMat52) (w_46_concat_14 : mxxMat53) (w_46_concat_15 : mxxMat54) (w_46_concat_16 : mxxMat55) (w_46_concat_17 : mxxMat56) (w_46_concat_18 : mxxMat57) (w_46_concat_19 : mxxMat58) (w_46_concat_20 : mxxMat59) (w_46_concat_21 : mxxMat60) (w_46_concat_22 : mxxMat61) (w_46_concat_23 : mxxMat62) (w_47_0 : mxxMat14) (w_49_0 : mxxMat14) (w_50_0 : mxxMat63) (w_51_0 : mxxMat63) (w_56_0 : mxxMat12) (w_58_0 : mxxMat2) (w_59_0 : mxxMat4) (w_60_0 : mxxMat4) (w_63_0 : mxxMat1) (w_64_0 : mxxMat1) : Prop :=
  MxxRuntime.concatDiagonal w_45_0 w_45_0 w_46_concat_1 ∧
  MxxRuntime.concatDiagonal w_46_concat_1 w_45_0 w_46_concat_2 ∧
  MxxRuntime.concatDiagonal w_46_concat_2 w_45_0 w_46_concat_3 ∧
  MxxRuntime.concatDiagonal w_46_concat_3 w_45_0 w_46_concat_4 ∧
  MxxRuntime.concatDiagonal w_46_concat_4 w_45_0 w_46_concat_5 ∧
  MxxRuntime.concatDiagonal w_46_concat_5 w_45_0 w_46_concat_6 ∧
  MxxRuntime.concatDiagonal w_46_concat_6 w_45_0 w_46_concat_7 ∧
  MxxRuntime.concatDiagonal w_46_concat_7 w_45_0 w_46_concat_8 ∧
  MxxRuntime.concatDiagonal w_46_concat_8 w_45_0 w_46_concat_9 ∧
  MxxRuntime.concatDiagonal w_46_concat_9 w_45_0 w_46_concat_10 ∧
  MxxRuntime.concatDiagonal w_46_concat_10 w_45_0 w_46_concat_11 ∧
  MxxRuntime.concatDiagonal w_46_concat_11 w_45_0 w_46_concat_12 ∧
  MxxRuntime.concatDiagonal w_46_concat_12 w_45_0 w_46_concat_13 ∧
  MxxRuntime.concatDiagonal w_46_concat_13 w_45_0 w_46_concat_14 ∧
  MxxRuntime.concatDiagonal w_46_concat_14 w_45_0 w_46_concat_15 ∧
  MxxRuntime.concatDiagonal w_46_concat_15 w_45_0 w_46_concat_16 ∧
  MxxRuntime.concatDiagonal w_46_concat_16 w_45_0 w_46_concat_17 ∧
  MxxRuntime.concatDiagonal w_46_concat_17 w_45_0 w_46_concat_18 ∧
  MxxRuntime.concatDiagonal w_46_concat_18 w_45_0 w_46_concat_19 ∧
  MxxRuntime.concatDiagonal w_46_concat_19 w_45_0 w_46_concat_20 ∧
  MxxRuntime.concatDiagonal w_46_concat_20 w_45_0 w_46_concat_21 ∧
  MxxRuntime.concatDiagonal w_46_concat_21 w_45_0 w_46_concat_22 ∧
  MxxRuntime.concatDiagonal w_46_concat_22 w_45_0 w_46_concat_23 ∧
  MxxRuntime.concatDiagonal w_46_concat_23 w_45_0 w_46_0 ∧
  MxxRuntime.centeredRebaseRuns w_46_0 w_47_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  MxxRuntime.concatColumns w_54_0 w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_63_0 ∧
  MxxRuntime.centeredRebaseRuns w_63_0 w_64_0

abbrev parallel_generatedRoot_1271.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_27_0 : mxxMat1) (w_43_0 : mxxMat8) (w_24_0 : mxxMat1) (w_28_0 : mxxMat15) (w_28_concat_14 : mxxMat29) (w_28_concat_15 : mxxMat30) (w_28_concat_16 : mxxMat31) (w_28_concat_17 : mxxMat32) (w_28_concat_18 : mxxMat33) (w_28_concat_19 : mxxMat34) (w_28_concat_20 : mxxMat35) (w_28_concat_21 : mxxMat36) (w_28_concat_22 : mxxMat37) (w_28_concat_23 : mxxMat38) (w_29_0 : mxxMat15) (w_31_0 : mxxMat15) (w_32_0 : mxxMat39) (w_33_0 : mxxMat39) (w_34_0 : mxxMat10) (w_35_0 : mxxMat1) (w_36_0 : mxxMat5) (w_37_0 : mxxMat11) (w_38_0 : mxxMat11) (w_39_0 : mxxMat5) (w_40_0 : mxxMat1) (w_41_0 : mxxMat1) (w_44_0 : mxxMat10) : Prop :=
  MxxRuntime.concatDiagonal w_28_concat_14 w_27_0 w_28_concat_15 ∧
  MxxRuntime.concatDiagonal w_28_concat_15 w_27_0 w_28_concat_16 ∧
  MxxRuntime.concatDiagonal w_28_concat_16 w_27_0 w_28_concat_17 ∧
  MxxRuntime.concatDiagonal w_28_concat_17 w_27_0 w_28_concat_18 ∧
  MxxRuntime.concatDiagonal w_28_concat_18 w_27_0 w_28_concat_19 ∧
  MxxRuntime.concatDiagonal w_28_concat_19 w_27_0 w_28_concat_20 ∧
  MxxRuntime.concatDiagonal w_28_concat_20 w_27_0 w_28_concat_21 ∧
  MxxRuntime.concatDiagonal w_28_concat_21 w_27_0 w_28_concat_22 ∧
  MxxRuntime.concatDiagonal w_28_concat_22 w_27_0 w_28_concat_23 ∧
  MxxRuntime.concatDiagonal w_28_concat_23 w_27_0 w_28_0 ∧
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
  MxxRuntime.tensorRuns w_24_0 w_39_0 w_40_0 ∧
  0 ≤ (24) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_1271_41 backend hashModel params i_0 i (current, (w_34_0, (w_38_0, ()))) next) (Int.toNat (24)) w_40_0 w_41_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (2500 + (3125 * (Int.ofNat i_0))) ∧
  (2500 + (3125 * (Int.ofNat i_0))) < (3125 + (3125 * (Int.ofNat i_0))) ∧
  (3125 + (3125 * (Int.ofNat i_0))) ≤ 112500 ∧
  MxxRuntime.sliceMatrix w_43_0 0 1 (2500 + (3125 * (Int.ofNat i_0))) (3125 + (3125 * (Int.ofNat i_0))) w_44_0

abbrev parallel_generatedRoot_1271.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_0_0 : mxxMat1) (w_1_0 : mxxMat1) (w_25_0 : mxxMat6) (w_27_0 : mxxMat1) (w_2_0 : mxxMat12) (w_5_0 : mxxMat2) (w_6_0 : mxxMat4) (w_7_0 : mxxMat4) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat71) (w_16_0 : mxxMat71) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat14) (w_21_0 : mxxMat14) (w_26_0 : mxxMat10) (w_28_concat_1 : mxxMat16) (w_28_concat_2 : mxxMat17) (w_28_concat_3 : mxxMat18) (w_28_concat_4 : mxxMat19) (w_28_concat_5 : mxxMat20) (w_28_concat_6 : mxxMat21) (w_28_concat_7 : mxxMat22) (w_28_concat_8 : mxxMat23) (w_28_concat_9 : mxxMat24) (w_28_concat_10 : mxxMat25) (w_28_concat_11 : mxxMat26) (w_28_concat_12 : mxxMat27) (w_28_concat_13 : mxxMat28) (w_28_concat_14 : mxxMat29) : Prop :=
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
  MxxRuntime.concatDiagonal w_27_0 w_27_0 w_28_concat_1 ∧
  MxxRuntime.concatDiagonal w_28_concat_1 w_27_0 w_28_concat_2 ∧
  MxxRuntime.concatDiagonal w_28_concat_2 w_27_0 w_28_concat_3 ∧
  MxxRuntime.concatDiagonal w_28_concat_3 w_27_0 w_28_concat_4 ∧
  MxxRuntime.concatDiagonal w_28_concat_4 w_27_0 w_28_concat_5 ∧
  MxxRuntime.concatDiagonal w_28_concat_5 w_27_0 w_28_concat_6 ∧
  MxxRuntime.concatDiagonal w_28_concat_6 w_27_0 w_28_concat_7 ∧
  MxxRuntime.concatDiagonal w_28_concat_7 w_27_0 w_28_concat_8 ∧
  MxxRuntime.concatDiagonal w_28_concat_8 w_27_0 w_28_concat_9 ∧
  MxxRuntime.concatDiagonal w_28_concat_9 w_27_0 w_28_concat_10 ∧
  MxxRuntime.concatDiagonal w_28_concat_10 w_27_0 w_28_concat_11 ∧
  MxxRuntime.concatDiagonal w_28_concat_11 w_27_0 w_28_concat_12 ∧
  MxxRuntime.concatDiagonal w_28_concat_12 w_27_0 w_28_concat_13 ∧
  MxxRuntime.concatDiagonal w_28_concat_13 w_27_0 w_28_concat_14

abbrev parallel_generatedRoot_1271.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) (witness : parallel_generatedRoot_1271.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_4_0 : mxxMat2 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_25_0 : mxxMat6 := inputs.2.2.2.2.1
  let w_27_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_43_0 : mxxMat8 := inputs.2.2.2.2.2.2.1
  let w_45_0 : mxxMat5 := inputs.2.2.2.2.2.2.2.1
  let w_54_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_55_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_62_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_78_0 : mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_95_0 : mxxMat8 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)
  let w_5_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_3_0 w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_2_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat72 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat72)
  let w_14_0 : mxxMat71 := MxxRuntime.matrixMulScalarRight w_12_0 w_13_0
  let w_15_0 : mxxMat72 := (MxxRuntime.matrixPolynomial [(157994188800)] : mxxMat72)
  let w_16_0 : mxxMat71 := MxxRuntime.matrixMulScalarRight w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat7)] : mxxMat5)
  let w_19_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat1 := MxxRuntime.matrixMul w_9_0 w_21_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_22_0
  let w_24_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_26_0 := witness.w_26_0
  let w_28_0 := witness.w_28_0
  let w_28_concat_1 := witness.w_28_concat_1
  let w_28_concat_2 := witness.w_28_concat_2
  let w_28_concat_3 := witness.w_28_concat_3
  let w_28_concat_4 := witness.w_28_concat_4
  let w_28_concat_5 := witness.w_28_concat_5
  let w_28_concat_6 := witness.w_28_concat_6
  let w_28_concat_7 := witness.w_28_concat_7
  let w_28_concat_8 := witness.w_28_concat_8
  let w_28_concat_9 := witness.w_28_concat_9
  let w_28_concat_10 := witness.w_28_concat_10
  let w_28_concat_11 := witness.w_28_concat_11
  let w_28_concat_12 := witness.w_28_concat_12
  let w_28_concat_13 := witness.w_28_concat_13
  let w_28_concat_14 := witness.w_28_concat_14
  let w_28_concat_15 := witness.w_28_concat_15
  let w_28_concat_16 := witness.w_28_concat_16
  let w_28_concat_17 := witness.w_28_concat_17
  let w_28_concat_18 := witness.w_28_concat_18
  let w_28_concat_19 := witness.w_28_concat_19
  let w_28_concat_20 := witness.w_28_concat_20
  let w_28_concat_21 := witness.w_28_concat_21
  let w_28_concat_22 := witness.w_28_concat_22
  let w_28_concat_23 := witness.w_28_concat_23
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_31_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_29_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat10 := MxxRuntime.matrixMul w_26_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : mxxMat5 := MxxRuntime.matrixMul w_35_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_41_0
  let w_44_0 := witness.w_44_0
  let w_46_0 := witness.w_46_0
  let w_46_concat_1 := witness.w_46_concat_1
  let w_46_concat_2 := witness.w_46_concat_2
  let w_46_concat_3 := witness.w_46_concat_3
  let w_46_concat_4 := witness.w_46_concat_4
  let w_46_concat_5 := witness.w_46_concat_5
  let w_46_concat_6 := witness.w_46_concat_6
  let w_46_concat_7 := witness.w_46_concat_7
  let w_46_concat_8 := witness.w_46_concat_8
  let w_46_concat_9 := witness.w_46_concat_9
  let w_46_concat_10 := witness.w_46_concat_10
  let w_46_concat_11 := witness.w_46_concat_11
  let w_46_concat_12 := witness.w_46_concat_12
  let w_46_concat_13 := witness.w_46_concat_13
  let w_46_concat_14 := witness.w_46_concat_14
  let w_46_concat_15 := witness.w_46_concat_15
  let w_46_concat_16 := witness.w_46_concat_16
  let w_46_concat_17 := witness.w_46_concat_17
  let w_46_concat_18 := witness.w_46_concat_18
  let w_46_concat_19 := witness.w_46_concat_19
  let w_46_concat_20 := witness.w_46_concat_20
  let w_46_concat_21 := witness.w_46_concat_21
  let w_46_concat_22 := witness.w_46_concat_22
  let w_46_concat_23 := witness.w_46_concat_23
  let w_47_0 := witness.w_47_0
  let w_48_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_49_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_47_0 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat1 := MxxRuntime.matrixMul w_44_0 w_51_0
  let w_53_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_52_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_57_0 w_4_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixMul w_56_0 w_60_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : mxxMat72 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat72)
  let w_67_0 : mxxMat71 := MxxRuntime.matrixMulScalarRight w_65_0 w_66_0
  let w_68_0 : mxxMat72 := (MxxRuntime.matrixPolynomial [(157994188800)] : mxxMat72)
  let w_69_0 : mxxMat71 := MxxRuntime.matrixMulScalarRight w_67_0 w_68_0
  let w_70_0 := witness.w_70_0
  let w_71_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat7)] : mxxMat5)
  let w_72_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_70_0 w_71_0
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 : mxxMat1 := MxxRuntime.matrixMul w_62_0 w_74_0
  let w_76_0 : mxxMat1 := MxxRuntime.matrixSub w_61_0 w_75_0
  let w_77_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_80_concat_1 := witness.w_80_concat_1
  let w_80_concat_2 := witness.w_80_concat_2
  let w_80_concat_3 := witness.w_80_concat_3
  let w_80_concat_4 := witness.w_80_concat_4
  let w_80_concat_5 := witness.w_80_concat_5
  let w_80_concat_6 := witness.w_80_concat_6
  let w_80_concat_7 := witness.w_80_concat_7
  let w_80_concat_8 := witness.w_80_concat_8
  let w_80_concat_9 := witness.w_80_concat_9
  let w_80_concat_10 := witness.w_80_concat_10
  let w_80_concat_11 := witness.w_80_concat_11
  let w_80_concat_12 := witness.w_80_concat_12
  let w_80_concat_13 := witness.w_80_concat_13
  let w_80_concat_14 := witness.w_80_concat_14
  let w_80_concat_15 := witness.w_80_concat_15
  let w_80_concat_16 := witness.w_80_concat_16
  let w_80_concat_17 := witness.w_80_concat_17
  let w_80_concat_18 := witness.w_80_concat_18
  let w_80_concat_19 := witness.w_80_concat_19
  let w_80_concat_20 := witness.w_80_concat_20
  let w_80_concat_21 := witness.w_80_concat_21
  let w_80_concat_22 := witness.w_80_concat_22
  let w_80_concat_23 := witness.w_80_concat_23
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_83_0 : mxxMat15 := MxxRuntime.matrixMulScalarRight w_81_0 w_82_0
  let w_84_0 := witness.w_84_0
  let w_85_0 := witness.w_85_0
  let w_86_0 : mxxMat10 := MxxRuntime.matrixMul w_79_0 w_85_0
  let w_87_0 := witness.w_87_0
  let w_88_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1606938044255707605848740785410594412685698358681526360823809)] : mxxMat5)
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 : mxxMat5 := MxxRuntime.matrixMul w_87_0 w_90_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 : mxxMat1 := MxxRuntime.matrixAdd w_76_0 w_93_0
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_97_concat_1 := witness.w_97_concat_1
  let w_97_concat_2 := witness.w_97_concat_2
  let w_97_concat_3 := witness.w_97_concat_3
  let w_97_concat_4 := witness.w_97_concat_4
  let w_97_concat_5 := witness.w_97_concat_5
  let w_97_concat_6 := witness.w_97_concat_6
  let w_97_concat_7 := witness.w_97_concat_7
  let w_97_concat_8 := witness.w_97_concat_8
  let w_97_concat_9 := witness.w_97_concat_9
  let w_97_concat_10 := witness.w_97_concat_10
  let w_97_concat_11 := witness.w_97_concat_11
  let w_97_concat_12 := witness.w_97_concat_12
  let w_97_concat_13 := witness.w_97_concat_13
  let w_97_concat_14 := witness.w_97_concat_14
  let w_97_concat_15 := witness.w_97_concat_15
  let w_97_concat_16 := witness.w_97_concat_16
  let w_97_concat_17 := witness.w_97_concat_17
  let w_97_concat_18 := witness.w_97_concat_18
  let w_97_concat_19 := witness.w_97_concat_19
  let w_97_concat_20 := witness.w_97_concat_20
  let w_97_concat_21 := witness.w_97_concat_21
  let w_97_concat_22 := witness.w_97_concat_22
  let w_97_concat_23 := witness.w_97_concat_23
  let w_98_0 := witness.w_98_0
  let w_99_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_100_0 : mxxMat14 := MxxRuntime.matrixMulScalarRight w_98_0 w_99_0
  let w_101_0 := witness.w_101_0
  let w_102_0 := witness.w_102_0
  let w_103_0 : mxxMat1 := MxxRuntime.matrixMul w_96_0 w_102_0
  let w_104_0 : mxxMat1 := MxxRuntime.matrixAdd w_94_0 w_103_0
  let w_105_0 := witness.w_105_0
  parallel_generatedRoot_1271.constraints_0 backend i_0 w_0_0 w_1_0 w_25_0 w_27_0 w_2_0 w_5_0 w_6_0 w_7_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_20_0 w_21_0 w_26_0 w_28_concat_1 w_28_concat_2 w_28_concat_3 w_28_concat_4 w_28_concat_5 w_28_concat_6 w_28_concat_7 w_28_concat_8 w_28_concat_9 w_28_concat_10 w_28_concat_11 w_28_concat_12 w_28_concat_13 w_28_concat_14 ∧
    parallel_generatedRoot_1271.constraints_1 backend hashModel params i_0 w_27_0 w_43_0 w_24_0 w_28_0 w_28_concat_14 w_28_concat_15 w_28_concat_16 w_28_concat_17 w_28_concat_18 w_28_concat_19 w_28_concat_20 w_28_concat_21 w_28_concat_22 w_28_concat_23 w_29_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_44_0 ∧
    parallel_generatedRoot_1271.constraints_2 backend w_45_0 w_54_0 w_55_0 w_46_0 w_46_concat_1 w_46_concat_2 w_46_concat_3 w_46_concat_4 w_46_concat_5 w_46_concat_6 w_46_concat_7 w_46_concat_8 w_46_concat_9 w_46_concat_10 w_46_concat_11 w_46_concat_12 w_46_concat_13 w_46_concat_14 w_46_concat_15 w_46_concat_16 w_46_concat_17 w_46_concat_18 w_46_concat_19 w_46_concat_20 w_46_concat_21 w_46_concat_22 w_46_concat_23 w_47_0 w_49_0 w_50_0 w_51_0 w_56_0 w_58_0 w_59_0 w_60_0 w_63_0 w_64_0 ∧
    parallel_generatedRoot_1271.constraints_3 backend i_0 w_27_0 w_78_0 w_64_0 w_65_0 w_69_0 w_70_0 w_72_0 w_73_0 w_74_0 w_79_0 w_80_concat_1 w_80_concat_2 w_80_concat_3 w_80_concat_4 w_80_concat_5 w_80_concat_6 w_80_concat_7 w_80_concat_8 w_80_concat_9 w_80_concat_10 w_80_concat_11 w_80_concat_12 w_80_concat_13 w_80_concat_14 w_80_concat_15 w_80_concat_16 w_80_concat_17 w_80_concat_18 w_80_concat_19 ∧
    parallel_generatedRoot_1271.constraints_4 backend hashModel params i_0 w_27_0 w_45_0 w_95_0 w_77_0 w_80_0 w_80_concat_19 w_80_concat_20 w_80_concat_21 w_80_concat_22 w_80_concat_23 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_96_0 w_97_concat_1 w_97_concat_2 w_97_concat_3 w_97_concat_4 w_97_concat_5 ∧
    parallel_generatedRoot_1271.constraints_5 backend w_45_0 w_53_0 w_97_0 w_97_concat_5 w_97_concat_6 w_97_concat_7 w_97_concat_8 w_97_concat_9 w_97_concat_10 w_97_concat_11 w_97_concat_12 w_97_concat_13 w_97_concat_14 w_97_concat_15 w_97_concat_16 w_97_concat_17 w_97_concat_18 w_97_concat_19 w_97_concat_20 w_97_concat_21 w_97_concat_22 w_97_concat_23 w_98_0 w_100_0 w_101_0 w_102_0 w_104_0 w_105_0 outputs

def parallel_generatedRoot_1271 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat2 × mxxMat1 × mxxMat6 × mxxMat1 × mxxMat8 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat8 × Unit) (outputs : mxxMat2) : Prop :=
  ∃ witness : parallel_generatedRoot_1271.Witness,
    parallel_generatedRoot_1271.body backend hashModel params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat1
  w_27_0 : Fin 16 → mxxMat1
  w_27_1 : Fin 16 → mxxMat1
  w_27_2 : Fin 16 → mxxMat1
  w_27_3 : Fin 16 → mxxMat1
  w_27_4 : Fin 16 → mxxMat1
  w_27_5 : Fin 16 → mxxMat1
  w_28_0 : Fin 8 → mxxMat1
  w_28_1 : Fin 8 → mxxMat1
  w_28_2 : Fin 8 → mxxMat1
  w_28_3 : Fin 8 → mxxMat1
  w_28_4 : Fin 8 → mxxMat1
  w_28_5 : Fin 8 → mxxMat1
  w_31_0 : Fin 16 → mxxMat1
  w_31_1 : Fin 16 → mxxMat1
  w_31_2 : Fin 16 → mxxMat1
  w_31_3 : Fin 16 → mxxMat1
  w_31_4 : Fin 16 → mxxMat1
  w_31_5 : Fin 16 → mxxMat1
  w_32_0 : Fin 8 → mxxMat1
  w_32_1 : Fin 8 → mxxMat1
  w_32_2 : Fin 8 → mxxMat1
  w_32_3 : Fin 8 → mxxMat1
  w_32_4 : Fin 8 → mxxMat1
  w_32_5 : Fin 8 → mxxMat1
  w_33_0 : Fin 8 → mxxMat1
  w_33_1 : Fin 8 → mxxMat1
  w_33_2 : Fin 8 → mxxMat1
  w_33_3 : Fin 8 → mxxMat1
  w_33_4 : Fin 8 → mxxMat1
  w_33_5 : Fin 8 → mxxMat1
  w_34_0 : Fin 8 → mxxMat1
  w_34_1 : Fin 8 → mxxMat1
  w_34_2 : Fin 8 → mxxMat1
  w_34_3 : Fin 8 → mxxMat1
  w_34_4 : Fin 8 → mxxMat1
  w_34_5 : Fin 8 → mxxMat1
  w_35_0 : Fin 8 → mxxMat1
  w_35_1 : Fin 8 → mxxMat1
  w_35_2 : Fin 8 → mxxMat1
  w_35_3 : Fin 8 → mxxMat1
  w_35_4 : Fin 8 → mxxMat1
  w_35_5 : Fin 8 → mxxMat1
  w_36_0 : mxxMat1
  w_37_0 : mxxMat1
  w_39_0 : mxxMat1
  w_40_0 : mxxMat1
  w_42_0 : mxxMat1
  w_43_0 : mxxMat1
  w_45_0 : mxxMat1
  w_46_0 : mxxMat1
  w_48_0 : mxxMat1
  w_49_0 : mxxMat1
  w_51_0 : mxxMat1
  w_52_0 : mxxMat1
  w_54_0 : mxxMat1
  w_54_1 : mxxMat1
  w_54_2 : mxxMat1
  w_54_3 : mxxMat1
  w_54_4 : mxxMat1
  w_54_5 : mxxMat1
  w_55_0 : mxxMat1
  w_56_0 : mxxMat1
  w_59_0 : mxxMat14
  w_60_0 : mxxMat14
  w_62_0 : mxxMat1
  w_63_0 : mxxMat1
  w_66_0 : mxxMat14
  w_67_0 : mxxMat14
  w_70_0 : mxxMat1
  w_71_0 : mxxMat1
  w_72_0 : mxxMat2
  w_73_0 : mxxMat2
  w_74_0 : mxxMat1
  w_75_0 : mxxMat1
  w_76_0 : mxxMat2
  w_77_0 : mxxMat2
  w_78_0 : mxxMat1
  w_79_0 : mxxMat1
  w_80_0 : mxxMat2
  w_81_0 : mxxMat2
  w_82_0 : mxxMat1
  w_83_0 : mxxMat1
  w_84_0 : mxxMat2
  w_85_0 : mxxMat2
  w_86_0 : mxxMat1
  w_87_0 : mxxMat1
  w_88_0 : mxxMat2
  w_89_0 : mxxMat2
  w_90_0 : mxxMat1
  w_91_0 : mxxMat1
  w_92_0 : mxxMat2
  w_93_0 : mxxMat2
  w_94_0 : mxxMat1
  w_95_0 : mxxMat1
  w_96_0 : mxxMat2
  w_97_0 : mxxMat2
  w_98_0 : mxxMat1
  w_99_0 : mxxMat1
  w_100_0 : mxxMat2
  w_101_0 : mxxMat2
  w_102_0 : mxxMat1
  w_103_0 : mxxMat1
  w_104_0 : mxxMat2
  w_105_0 : mxxMat2
  w_106_0 : mxxMat1
  w_107_0 : mxxMat1
  w_108_0 : mxxMat2
  w_109_0 : mxxMat2
  w_110_0 : mxxMat1
  w_111_0 : mxxMat1
  w_112_0 : mxxMat2
  w_113_0 : mxxMat2
  w_114_0 : mxxMat1
  w_115_0 : mxxMat1
  w_116_0 : mxxMat2
  w_117_0 : mxxMat2
  w_118_0 : mxxMat1
  w_119_0 : mxxMat1
  w_120_0 : mxxMat2
  w_121_0 : mxxMat2
  w_122_0 : mxxMat1
  w_123_0 : mxxMat1
  w_124_0 : mxxMat2
  w_125_0 : mxxMat2
  w_126_0 : mxxMat1
  w_127_0 : mxxMat1
  w_128_0 : mxxMat2
  w_129_0 : mxxMat2
  w_130_0 : mxxMat1
  w_131_0 : mxxMat1
  w_132_0 : mxxMat2
  w_133_0 : mxxMat2
  w_135_0 : mxxMat1
  w_136_0 : mxxMat1
  w_137_0 : mxxMat2
  w_138_0 : mxxMat2
  w_140_0 : mxxMat1
  w_141_0 : mxxMat1
  w_142_0 : mxxMat2
  w_143_0 : mxxMat2
  w_145_0 : mxxMat1
  w_146_0 : mxxMat1
  w_147_0 : mxxMat2
  w_148_0 : mxxMat2
  w_150_0 : mxxMat1
  w_151_0 : mxxMat1
  w_152_0 : mxxMat2
  w_153_0 : mxxMat2
  w_155_0 : mxxMat1
  w_156_0 : mxxMat1
  w_157_0 : mxxMat2
  w_158_0 : mxxMat2
  w_160_0 : mxxMat1
  w_161_0 : mxxMat1
  w_162_0 : mxxMat2
  w_163_0 : mxxMat2
  w_165_0 : mxxMat1
  w_166_0 : mxxMat1
  w_167_0 : mxxMat2
  w_168_0 : mxxMat2
  w_170_0 : mxxMat1
  w_171_0 : mxxMat1
  w_172_0 : mxxMat2
  w_173_0 : mxxMat2
  w_175_0 : mxxMat1
  w_176_0 : mxxMat1
  w_177_0 : mxxMat2
  w_178_0 : mxxMat2
  w_180_0 : mxxMat1
  w_181_0 : mxxMat1
  w_182_0 : mxxMat2
  w_183_0 : mxxMat2
  w_185_0 : mxxMat1
  w_186_0 : mxxMat1
  w_187_0 : mxxMat2
  w_188_0 : mxxMat2
  w_190_0 : mxxMat1
  w_191_0 : mxxMat1
  w_192_0 : mxxMat2
  w_193_0 : mxxMat2
  w_195_0 : mxxMat1
  w_196_0 : mxxMat1
  w_197_0 : mxxMat2
  w_198_0 : mxxMat2
  w_200_0 : mxxMat1
  w_201_0 : mxxMat1
  w_202_0 : mxxMat2
  w_203_0 : mxxMat2
  w_205_0 : mxxMat1
  w_206_0 : mxxMat1
  w_207_0 : mxxMat2
  w_208_0 : mxxMat2
  w_210_0 : mxxMat1
  w_211_0 : mxxMat1
  w_212_0 : mxxMat2
  w_213_0 : mxxMat2
  w_215_0 : mxxMat73
  w_216_0 : mxxMat74
  w_217_0 : mxxMat2
  w_218_0 : mxxMat2
  w_219_0 : mxxMat2
  w_220_0 : mxxMat2
  w_221_0 : mxxMat2
  w_222_0 : mxxMat2
  w_223_0 : mxxMat2
  w_224_0 : mxxMat2
  w_225_0 : Fin 36 → mxxMat2
  w_227_0 : mxxMat1
  w_228_0 : mxxMat1
  w_230_0 : mxxMat1
  w_233_0 : mxxMat14
  w_234_0 : mxxMat14
  w_237_0 : mxxMat1
  w_240_0 : mxxMat14
  w_241_0 : mxxMat14
  w_246_0 : mxxMat1
  w_248_0 : mxxMat1
  w_250_0 : mxxMat1
  w_252_0 : mxxMat1
  w_256_0 : mxxMat1
  w_257_0 : mxxMat1
  w_258_0 : mxxMat1
  w_259_0 : mxxMat0
  w_259_concat_1 : mxxMat2
  w_261_0 : mxxMat1
  w_262_0 : mxxMat1
  w_263_0 : mxxMat1
  w_264_0 : mxxMat0
  w_264_concat_1 : mxxMat2
  w_265_0 : mxxMat3
  w_266_0 : mxxMat1
  w_267_0 : mxxMat1
  w_268_0 : mxxMat2
  w_269_0 : mxxMat4
  w_270_0 : mxxMat4
  w_272_0 : mxxMat1
  w_275_0 : mxxMat5
  w_278_0 : mxxMat1
  w_279_0 : mxxMat1
  w_280_0 : mxxMat0
  w_280_concat_1 : mxxMat2
  w_281_0 : mxxMat1
  w_282_0 : mxxMat1
  w_283_0 : mxxMat2
  w_284_0 : mxxMat4
  w_285_0 : mxxMat4
  w_287_0 : mxxMat1
  w_291_0 : mxxMat1
  w_292_0 : mxxMat1
  w_293_0 : mxxMat0
  w_293_concat_1 : mxxMat2
  w_294_0 : mxxMat3
  w_295_0 : mxxMat1
  w_297_0 : mxxMat1
  w_298_0 : mxxMat2
  w_299_0 : mxxMat4
  w_300_0 : mxxMat4
  w_302_0 : mxxMat1
  w_303_0 : mxxMat1
  w_304_0 : mxxMat1
  w_305_0 : mxxMat1
  w_307_0 : mxxMat1
  w_308_0 : mxxMat2
  w_309_0 : mxxMat4
  w_310_0 : mxxMat4
  w_312_0 : mxxMat1
  w_313_0 : mxxMat1
  w_314_0 : mxxMat1
  w_315_0 : mxxMat1
  w_315_1 : mxxMat1
  w_315_2 : mxxMat1
  w_315_3 : mxxMat1
  w_315_4 : mxxMat1
  w_315_5 : mxxMat1
  w_315_6 : mxxMat1
  w_315_7 : mxxMat1
  w_315_8 : mxxMat1
  w_315_9 : mxxMat1
  w_316_0 : mxxMat1
  w_318_0 : mxxMat1
  w_319_0 : mxxMat1
  w_320_0 : mxxMat0
  w_320_concat_1 : mxxMat2
  w_321_0 : mxxMat1
  w_322_0 : mxxMat1
  w_323_0 : mxxMat1
  w_324_0 : mxxMat0
  w_324_concat_1 : mxxMat2
  w_325_0 : mxxMat3
  w_326_0 : mxxMat1
  w_327_0 : mxxMat1
  w_328_0 : mxxMat1
  w_329_0 : mxxMat0
  w_329_concat_1 : mxxMat2
  w_330_0 : mxxMat1
  w_331_0 : mxxMat1
  w_332_0 : mxxMat1
  w_333_0 : mxxMat0
  w_333_concat_1 : mxxMat2
  w_334_0 : mxxMat3
  w_335_0 : mxxMat2
  w_336_0 : mxxMat4
  w_337_0 : mxxMat4
  w_339_0 : mxxMat1
  w_340_0 : mxxMat1
  w_341_0 : mxxMat2
  w_342_0 : mxxMat4
  w_343_0 : mxxMat4
  w_345_0 : mxxMat1
  w_347_0 : mxxMat1
  w_350_0 : mxxMat14
  w_351_0 : mxxMat14
  w_353_0 : mxxMat2
  w_354_0 : mxxMat4
  w_355_0 : mxxMat4
  w_357_0 : mxxMat1
  w_358_0 : mxxMat1
  w_359_0 : mxxMat2
  w_360_0 : mxxMat4
  w_361_0 : mxxMat4
  w_363_0 : mxxMat1
  w_367_0 : mxxMat1
  w_368_0 : mxxMat1
  w_369_0 : mxxMat1
  w_370_0 : mxxMat0
  w_370_concat_1 : mxxMat2
  w_372_0 : mxxMat1
  w_373_0 : mxxMat1
  w_374_0 : mxxMat1
  w_375_0 : mxxMat0
  w_375_concat_1 : mxxMat2
  w_376_0 : mxxMat3
  w_377_0 : mxxMat1
  w_378_0 : mxxMat1
  w_379_0 : mxxMat2
  w_380_0 : mxxMat4
  w_381_0 : mxxMat4
  w_383_0 : mxxMat1
  w_384_0 : mxxMat1
  w_387_0 : mxxMat14
  w_388_0 : mxxMat14
  w_390_0 : mxxMat14
  w_392_0 : mxxMat1
  w_393_0 : mxxMat1
  w_394_0 : mxxMat2
  w_395_0 : mxxMat4
  w_396_0 : mxxMat4
  w_398_0 : mxxMat1
  w_402_0 : mxxMat14
  w_405_0 : mxxMat14
  w_407_0 : mxxMat1
  w_410_0 : mxxMat14
  w_413_0 : mxxMat1
  w_414_0 : mxxMat1
  w_415_0 : mxxMat1
  w_416_0 : mxxMat0
  w_416_concat_1 : mxxMat2
  w_417_0 : mxxMat1
  w_418_0 : mxxMat1
  w_419_0 : mxxMat1
  w_420_0 : mxxMat0
  w_420_concat_1 : mxxMat2
  w_421_0 : mxxMat3
  w_422_0 : mxxMat1
  w_423_0 : mxxMat1
  w_424_0 : mxxMat2
  w_425_0 : mxxMat4
  w_426_0 : mxxMat4
  w_428_0 : mxxMat1
  w_429_0 : mxxMat1
  w_431_0 : mxxMat5
  w_434_0 : mxxMat1
  w_435_0 : mxxMat1
  w_436_0 : mxxMat0
  w_436_concat_1 : mxxMat2
  w_437_0 : mxxMat1
  w_438_0 : mxxMat1
  w_439_0 : mxxMat2
  w_440_0 : mxxMat4
  w_441_0 : mxxMat4
  w_443_0 : mxxMat1
  w_444_0 : mxxMat1
  w_447_0 : mxxMat1
  w_448_0 : mxxMat1
  w_449_0 : mxxMat0
  w_449_concat_1 : mxxMat2
  w_450_0 : mxxMat3
  w_451_0 : mxxMat1
  w_452_0 : mxxMat1
  w_453_0 : mxxMat1
  w_454_0 : mxxMat0
  w_454_concat_1 : mxxMat2
  w_455_0 : mxxMat1
  w_456_0 : mxxMat1
  w_457_0 : mxxMat1
  w_458_0 : mxxMat0
  w_458_concat_1 : mxxMat2
  w_459_0 : mxxMat3
  w_460_0 : mxxMat1
  w_461_0 : mxxMat1
  w_462_0 : mxxMat2
  w_463_0 : mxxMat4
  w_464_0 : mxxMat4
  w_466_0 : mxxMat1
  w_467_0 : mxxMat1
  w_468_0 : mxxMat2
  w_469_0 : mxxMat4
  w_470_0 : mxxMat4
  w_472_0 : mxxMat1
  w_473_0 : mxxMat1
  w_476_0 : mxxMat14
  w_477_0 : mxxMat14
  w_479_0 : mxxMat14
  w_481_0 : mxxMat1
  w_482_0 : mxxMat1
  w_483_0 : mxxMat2
  w_484_0 : mxxMat4
  w_485_0 : mxxMat4
  w_487_0 : mxxMat1
  w_488_0 : mxxMat1
  w_489_0 : mxxMat2
  w_490_0 : mxxMat4
  w_491_0 : mxxMat4
  w_493_0 : mxxMat1
  w_495_0 : mxxMat14
  w_498_0 : mxxMat14
  w_500_0 : mxxMat14
  w_503_0 : mxxMat1
  w_504_0 : mxxMat1
  w_505_0 : mxxMat1
  w_506_0 : mxxMat0
  w_506_concat_1 : mxxMat2
  w_507_0 : mxxMat1
  w_508_0 : mxxMat1
  w_509_0 : mxxMat1
  w_510_0 : mxxMat0
  w_510_concat_1 : mxxMat2
  w_511_0 : mxxMat3
  w_512_0 : mxxMat1
  w_513_0 : mxxMat1
  w_514_0 : mxxMat2
  w_515_0 : mxxMat4
  w_516_0 : mxxMat4
  w_518_0 : mxxMat1
  w_519_0 : mxxMat1
  w_520_0 : mxxMat5
  w_523_0 : mxxMat1
  w_524_0 : mxxMat1
  w_525_0 : mxxMat0
  w_525_concat_1 : mxxMat2
  w_526_0 : mxxMat1
  w_527_0 : mxxMat1
  w_528_0 : mxxMat2
  w_529_0 : mxxMat4
  w_530_0 : mxxMat4
  w_532_0 : mxxMat1
  w_533_0 : mxxMat1
  w_536_0 : mxxMat1
  w_537_0 : mxxMat1
  w_538_0 : mxxMat0
  w_538_concat_1 : mxxMat2
  w_539_0 : mxxMat3
  w_540_0 : mxxMat1
  w_541_0 : mxxMat1
  w_542_0 : mxxMat2
  w_543_0 : mxxMat4
  w_544_0 : mxxMat4
  w_546_0 : mxxMat1
  w_547_0 : mxxMat1
  w_550_0 : mxxMat14
  w_551_0 : mxxMat14
  w_553_0 : mxxMat14
  w_555_0 : mxxMat1
  w_556_0 : mxxMat1
  w_557_0 : mxxMat2
  w_558_0 : mxxMat4
  w_559_0 : mxxMat4
  w_561_0 : mxxMat1
  w_563_0 : mxxMat14
  w_566_0 : mxxMat14
  w_568_0 : mxxMat14
  w_571_0 : mxxMat1
  w_572_0 : mxxMat1
  w_573_0 : mxxMat1
  w_574_0 : mxxMat0
  w_574_concat_1 : mxxMat2
  w_575_0 : mxxMat1
  w_576_0 : mxxMat1
  w_577_0 : mxxMat1
  w_578_0 : mxxMat0
  w_578_concat_1 : mxxMat2
  w_579_0 : mxxMat3
  w_580_0 : mxxMat1
  w_581_0 : mxxMat1
  w_582_0 : mxxMat2
  w_583_0 : mxxMat4
  w_584_0 : mxxMat4
  w_586_0 : mxxMat1
  w_587_0 : mxxMat1
  w_590_0 : mxxMat14
  w_591_0 : mxxMat14
  w_593_0 : mxxMat14
  w_595_0 : mxxMat1
  w_596_0 : mxxMat1
  w_597_0 : mxxMat2
  w_598_0 : mxxMat4
  w_599_0 : mxxMat4
  w_601_0 : mxxMat1
  w_603_0 : mxxMat14
  w_606_0 : mxxMat14
  w_608_0 : mxxMat14
  w_611_0 : mxxMat1
  w_612_0 : mxxMat1
  w_613_0 : mxxMat1
  w_614_0 : mxxMat0
  w_614_concat_1 : mxxMat2
  w_615_0 : mxxMat1
  w_616_0 : mxxMat1
  w_617_0 : mxxMat1
  w_618_0 : mxxMat0
  w_618_concat_1 : mxxMat2
  w_619_0 : mxxMat3
  w_620_0 : mxxMat1
  w_621_0 : mxxMat1
  w_622_0 : mxxMat2
  w_623_0 : mxxMat4
  w_624_0 : mxxMat4
  w_626_0 : mxxMat1
  w_627_0 : mxxMat1
  w_630_0 : mxxMat14
  w_631_0 : mxxMat14
  w_633_0 : mxxMat14
  w_635_0 : mxxMat1
  w_636_0 : mxxMat1
  w_637_0 : mxxMat2
  w_638_0 : mxxMat4
  w_639_0 : mxxMat4
  w_641_0 : mxxMat1
  w_643_0 : mxxMat14
  w_646_0 : mxxMat14
  w_648_0 : mxxMat14
  w_651_0 : mxxMat1
  w_654_0 : mxxMat14
  w_655_0 : mxxMat14
  w_657_0 : mxxMat1
  w_660_0 : mxxMat14
  w_661_0 : mxxMat14
  w_666_0 : mxxMat1
  w_668_0 : mxxMat1
  w_670_0 : mxxMat1
  w_672_0 : mxxMat1
  w_675_0 : mxxMat1
  w_675_1 : mxxMat1
  w_675_2 : mxxMat1
  w_675_3 : mxxMat1
  w_675_4 : mxxMat1
  w_675_5 : mxxMat1
  w_675_6 : mxxMat1
  w_675_7 : mxxMat1
  w_675_8 : mxxMat1
  w_675_9 : mxxMat1
  w_676_0 : mxxMat1
  w_678_0 : mxxMat1
  w_679_0 : mxxMat1
  w_680_0 : mxxMat0
  w_680_concat_1 : mxxMat2
  w_681_0 : mxxMat1
  w_682_0 : mxxMat1
  w_683_0 : mxxMat1
  w_684_0 : mxxMat0
  w_684_concat_1 : mxxMat2
  w_685_0 : mxxMat3
  w_686_0 : mxxMat1
  w_687_0 : mxxMat1
  w_688_0 : mxxMat1
  w_689_0 : mxxMat0
  w_689_concat_1 : mxxMat2
  w_690_0 : mxxMat1
  w_691_0 : mxxMat1
  w_692_0 : mxxMat1
  w_693_0 : mxxMat0
  w_693_concat_1 : mxxMat2
  w_694_0 : mxxMat3
  w_695_0 : mxxMat2
  w_696_0 : mxxMat4
  w_697_0 : mxxMat4
  w_699_0 : mxxMat1
  w_700_0 : mxxMat1
  w_701_0 : mxxMat2
  w_702_0 : mxxMat4
  w_703_0 : mxxMat4
  w_705_0 : mxxMat1
  w_707_0 : mxxMat1
  w_710_0 : mxxMat14
  w_711_0 : mxxMat14
  w_713_0 : mxxMat1
  w_716_0 : mxxMat14
  w_717_0 : mxxMat14
  w_722_0 : mxxMat1
  w_724_0 : mxxMat1
  w_726_0 : mxxMat1
  w_728_0 : mxxMat1
  w_731_0 : mxxMat1
  w_731_1 : mxxMat1
  w_731_2 : mxxMat1
  w_731_3 : mxxMat1
  w_731_4 : mxxMat1
  w_731_5 : mxxMat1
  w_731_6 : mxxMat1
  w_731_7 : mxxMat1
  w_731_8 : mxxMat1
  w_731_9 : mxxMat1
  w_732_0 : mxxMat1
  w_734_0 : mxxMat1
  w_735_0 : mxxMat1
  w_736_0 : mxxMat0
  w_736_concat_1 : mxxMat2
  w_737_0 : mxxMat1
  w_738_0 : mxxMat1
  w_739_0 : mxxMat1
  w_740_0 : mxxMat0
  w_740_concat_1 : mxxMat2
  w_741_0 : mxxMat3
  w_742_0 : mxxMat1
  w_743_0 : mxxMat1
  w_744_0 : mxxMat1
  w_745_0 : mxxMat0
  w_745_concat_1 : mxxMat2
  w_746_0 : mxxMat1
  w_747_0 : mxxMat1
  w_748_0 : mxxMat1
  w_749_0 : mxxMat0
  w_749_concat_1 : mxxMat2
  w_750_0 : mxxMat3
  w_751_0 : mxxMat2
  w_752_0 : mxxMat4
  w_753_0 : mxxMat4
  w_755_0 : mxxMat1
  w_756_0 : mxxMat1
  w_757_0 : mxxMat2
  w_758_0 : mxxMat4
  w_759_0 : mxxMat4
  w_761_0 : mxxMat1
  w_763_0 : mxxMat1
  w_766_0 : mxxMat14
  w_767_0 : mxxMat14
  w_769_0 : mxxMat1
  w_772_0 : mxxMat14
  w_773_0 : mxxMat14
  w_778_0 : mxxMat1
  w_780_0 : mxxMat1
  w_782_0 : mxxMat1
  w_784_0 : mxxMat1
  w_787_0 : mxxMat1
  w_787_1 : mxxMat1
  w_787_2 : mxxMat1
  w_787_3 : mxxMat1
  w_787_4 : mxxMat1
  w_787_5 : mxxMat1
  w_787_6 : mxxMat1
  w_787_7 : mxxMat1
  w_787_8 : mxxMat1
  w_787_9 : mxxMat1
  w_788_0 : mxxMat1
  w_790_0 : mxxMat1
  w_791_0 : mxxMat1
  w_792_0 : mxxMat0
  w_792_concat_1 : mxxMat2
  w_793_0 : mxxMat1
  w_794_0 : mxxMat1
  w_795_0 : mxxMat1
  w_796_0 : mxxMat0
  w_796_concat_1 : mxxMat2
  w_797_0 : mxxMat3
  w_798_0 : mxxMat1
  w_799_0 : mxxMat1
  w_800_0 : mxxMat1
  w_801_0 : mxxMat0
  w_801_concat_1 : mxxMat2
  w_802_0 : mxxMat1
  w_803_0 : mxxMat1
  w_804_0 : mxxMat1
  w_805_0 : mxxMat0
  w_805_concat_1 : mxxMat2
  w_806_0 : mxxMat3
  w_807_0 : mxxMat2
  w_808_0 : mxxMat4
  w_809_0 : mxxMat4
  w_811_0 : mxxMat1
  w_812_0 : mxxMat1
  w_813_0 : mxxMat2
  w_814_0 : mxxMat4
  w_815_0 : mxxMat4
  w_817_0 : mxxMat1
  w_819_0 : mxxMat1
  w_822_0 : mxxMat14
  w_823_0 : mxxMat14
  w_825_0 : mxxMat1
  w_828_0 : mxxMat14
  w_829_0 : mxxMat14
  w_834_0 : mxxMat1
  w_836_0 : mxxMat1
  w_838_0 : mxxMat1
  w_840_0 : mxxMat1
  w_843_0 : mxxMat1
  w_843_1 : mxxMat1
  w_843_2 : mxxMat1
  w_843_3 : mxxMat1
  w_843_4 : mxxMat1
  w_843_5 : mxxMat1
  w_843_6 : mxxMat1
  w_843_7 : mxxMat1
  w_843_8 : mxxMat1
  w_843_9 : mxxMat1
  w_844_0 : mxxMat1
  w_846_0 : mxxMat1
  w_847_0 : mxxMat1
  w_848_0 : mxxMat0
  w_848_concat_1 : mxxMat2
  w_849_0 : mxxMat1
  w_850_0 : mxxMat1
  w_851_0 : mxxMat1
  w_852_0 : mxxMat0
  w_852_concat_1 : mxxMat2
  w_853_0 : mxxMat3
  w_854_0 : mxxMat1
  w_855_0 : mxxMat1
  w_856_0 : mxxMat1
  w_857_0 : mxxMat0
  w_857_concat_1 : mxxMat2
  w_858_0 : mxxMat1
  w_859_0 : mxxMat1
  w_860_0 : mxxMat1
  w_861_0 : mxxMat0
  w_861_concat_1 : mxxMat2
  w_862_0 : mxxMat3
  w_863_0 : mxxMat2
  w_864_0 : mxxMat4
  w_865_0 : mxxMat4
  w_867_0 : mxxMat1
  w_868_0 : mxxMat1
  w_869_0 : mxxMat2
  w_870_0 : mxxMat4
  w_871_0 : mxxMat4
  w_873_0 : mxxMat1
  w_875_0 : mxxMat1
  w_878_0 : mxxMat14
  w_879_0 : mxxMat14
  w_881_0 : mxxMat1
  w_884_0 : mxxMat14
  w_885_0 : mxxMat14
  w_890_0 : mxxMat1
  w_892_0 : mxxMat1
  w_894_0 : mxxMat1
  w_896_0 : mxxMat1
  w_899_0 : mxxMat1
  w_899_1 : mxxMat1
  w_899_2 : mxxMat1
  w_899_3 : mxxMat1
  w_899_4 : mxxMat1
  w_899_5 : mxxMat1
  w_899_6 : mxxMat1
  w_899_7 : mxxMat1
  w_899_8 : mxxMat1
  w_899_9 : mxxMat1
  w_900_0 : mxxMat1
  w_902_0 : mxxMat1
  w_903_0 : mxxMat1
  w_904_0 : mxxMat0
  w_904_concat_1 : mxxMat2
  w_905_0 : mxxMat1
  w_906_0 : mxxMat1
  w_907_0 : mxxMat1
  w_908_0 : mxxMat0
  w_908_concat_1 : mxxMat2
  w_909_0 : mxxMat3
  w_910_0 : mxxMat1
  w_911_0 : mxxMat1
  w_912_0 : mxxMat1
  w_913_0 : mxxMat0
  w_913_concat_1 : mxxMat2
  w_914_0 : mxxMat1
  w_915_0 : mxxMat1
  w_916_0 : mxxMat1
  w_917_0 : mxxMat0
  w_917_concat_1 : mxxMat2
  w_918_0 : mxxMat3
  w_919_0 : mxxMat2
  w_920_0 : mxxMat4
  w_921_0 : mxxMat4
  w_923_0 : mxxMat1
  w_924_0 : mxxMat1
  w_925_0 : mxxMat2
  w_926_0 : mxxMat4
  w_927_0 : mxxMat4
  w_929_0 : mxxMat1
  w_931_0 : mxxMat1
  w_934_0 : mxxMat14
  w_935_0 : mxxMat14
  w_937_0 : mxxMat1
  w_940_0 : mxxMat14
  w_941_0 : mxxMat14
  w_946_0 : mxxMat1
  w_948_0 : mxxMat1
  w_950_0 : mxxMat1
  w_952_0 : mxxMat1
  w_955_0 : mxxMat1
  w_955_1 : mxxMat1
  w_955_2 : mxxMat1
  w_955_3 : mxxMat1
  w_955_4 : mxxMat1
  w_955_5 : mxxMat1
  w_955_6 : mxxMat1
  w_955_7 : mxxMat1
  w_955_8 : mxxMat1
  w_955_9 : mxxMat1
  w_956_0 : mxxMat1
  w_958_0 : mxxMat1
  w_959_0 : mxxMat1
  w_960_0 : mxxMat0
  w_960_concat_1 : mxxMat2
  w_961_0 : mxxMat1
  w_962_0 : mxxMat1
  w_963_0 : mxxMat1
  w_964_0 : mxxMat0
  w_964_concat_1 : mxxMat2
  w_965_0 : mxxMat3
  w_966_0 : mxxMat1
  w_967_0 : mxxMat1
  w_968_0 : mxxMat1
  w_969_0 : mxxMat0
  w_969_concat_1 : mxxMat2
  w_970_0 : mxxMat1
  w_971_0 : mxxMat1
  w_972_0 : mxxMat1
  w_973_0 : mxxMat0
  w_973_concat_1 : mxxMat2
  w_974_0 : mxxMat3
  w_975_0 : mxxMat2
  w_976_0 : mxxMat4
  w_977_0 : mxxMat4
  w_979_0 : mxxMat1
  w_980_0 : mxxMat1
  w_981_0 : mxxMat2
  w_982_0 : mxxMat4
  w_983_0 : mxxMat4
  w_985_0 : mxxMat1
  w_987_0 : mxxMat1
  w_990_0 : mxxMat14
  w_991_0 : mxxMat14
  w_993_0 : mxxMat1
  w_996_0 : mxxMat14
  w_997_0 : mxxMat14
  w_1002_0 : mxxMat1
  w_1004_0 : mxxMat1
  w_1006_0 : mxxMat1
  w_1008_0 : mxxMat1
  w_1011_0 : mxxMat1
  w_1011_1 : mxxMat1
  w_1011_2 : mxxMat1
  w_1011_3 : mxxMat1
  w_1011_4 : mxxMat1
  w_1011_5 : mxxMat1
  w_1011_6 : mxxMat1
  w_1011_7 : mxxMat1
  w_1011_8 : mxxMat1
  w_1011_9 : mxxMat1
  w_1012_0 : mxxMat1
  w_1014_0 : mxxMat1
  w_1015_0 : mxxMat1
  w_1016_0 : mxxMat0
  w_1016_concat_1 : mxxMat2
  w_1017_0 : mxxMat1
  w_1018_0 : mxxMat1
  w_1019_0 : mxxMat1
  w_1020_0 : mxxMat0
  w_1020_concat_1 : mxxMat2
  w_1021_0 : mxxMat3
  w_1022_0 : mxxMat1
  w_1023_0 : mxxMat1
  w_1024_0 : mxxMat1
  w_1025_0 : mxxMat0
  w_1025_concat_1 : mxxMat2
  w_1026_0 : mxxMat1
  w_1027_0 : mxxMat1
  w_1028_0 : mxxMat1
  w_1029_0 : mxxMat0
  w_1029_concat_1 : mxxMat2
  w_1030_0 : mxxMat3
  w_1031_0 : mxxMat2
  w_1032_0 : mxxMat4
  w_1033_0 : mxxMat4
  w_1035_0 : mxxMat1
  w_1036_0 : mxxMat1
  w_1037_0 : mxxMat2
  w_1038_0 : mxxMat4
  w_1039_0 : mxxMat4
  w_1041_0 : mxxMat1
  w_1043_0 : mxxMat1
  w_1046_0 : mxxMat14
  w_1047_0 : mxxMat14
  w_1049_0 : mxxMat1
  w_1052_0 : mxxMat14
  w_1053_0 : mxxMat14
  w_1057_0 : Fin 36 → mxxMat1
  w_1067_0 : mxxMat1
  w_1068_0 : mxxMat1
  w_1071_0 : mxxMat1
  w_1072_0 : mxxMat1
  w_1076_0 : Fin 14400 → mxxMat1
  w_1076_1 : Fin 14400 → mxxMat1
  w_1076_2 : Fin 14400 → mxxMat1
  w_1076_3 : Fin 14400 → mxxMat1
  w_1076_4 : Fin 14400 → mxxMat1
  w_1076_5 : Fin 14400 → mxxMat1
  w_1077_0 : Fin 7200 → mxxMat1
  w_1077_1 : Fin 7200 → mxxMat1
  w_1077_2 : Fin 7200 → mxxMat1
  w_1077_3 : Fin 7200 → mxxMat1
  w_1077_4 : Fin 7200 → mxxMat1
  w_1077_5 : Fin 7200 → mxxMat1
  w_1080_0 : Fin 14400 → mxxMat1
  w_1080_1 : Fin 14400 → mxxMat1
  w_1080_2 : Fin 14400 → mxxMat1
  w_1080_3 : Fin 14400 → mxxMat1
  w_1080_4 : Fin 14400 → mxxMat1
  w_1080_5 : Fin 14400 → mxxMat1
  w_1081_0 : Fin 7200 → mxxMat1
  w_1081_1 : Fin 7200 → mxxMat1
  w_1081_2 : Fin 7200 → mxxMat1
  w_1081_3 : Fin 7200 → mxxMat1
  w_1081_4 : Fin 7200 → mxxMat1
  w_1081_5 : Fin 7200 → mxxMat1
  w_1082_0 : Fin 7200 → mxxMat1
  w_1082_1 : Fin 7200 → mxxMat1
  w_1082_2 : Fin 7200 → mxxMat1
  w_1082_3 : Fin 7200 → mxxMat1
  w_1082_4 : Fin 7200 → mxxMat1
  w_1082_5 : Fin 7200 → mxxMat1
  w_1083_0 : Fin 7200 → mxxMat1
  w_1083_1 : Fin 7200 → mxxMat1
  w_1083_2 : Fin 7200 → mxxMat1
  w_1083_3 : Fin 7200 → mxxMat1
  w_1083_4 : Fin 7200 → mxxMat1
  w_1083_5 : Fin 7200 → mxxMat1
  w_1084_0 : mxxMat1
  w_1085_0 : mxxMat1
  w_1087_0 : mxxMat1
  w_1088_0 : mxxMat1
  w_1090_0 : mxxMat1
  w_1091_0 : mxxMat1
  w_1093_0 : mxxMat1
  w_1094_0 : mxxMat1
  w_1096_0 : mxxMat1
  w_1097_0 : mxxMat1
  w_1099_0 : mxxMat1
  w_1100_0 : mxxMat1
  w_1102_0 : mxxMat1
  w_1102_1 : mxxMat1
  w_1102_2 : mxxMat1
  w_1102_3 : mxxMat1
  w_1102_4 : mxxMat1
  w_1102_5 : mxxMat1
  w_1103_0 : mxxMat6
  w_1105_0 : mxxMat6
  w_1107_0 : mxxMat6
  w_1109_0 : mxxMat6
  w_1111_0 : mxxMat6
  w_1113_0 : mxxMat6
  w_1114_0 : mxxMat6
  w_1114_1 : mxxMat6
  w_1114_2 : mxxMat6
  w_1114_3 : mxxMat6
  w_1114_4 : mxxMat6
  w_1114_5 : mxxMat6
  w_1122_0 : mxxMat1
  w_1123_0 : mxxMat1
  w_1125_0 : mxxMat1
  w_1126_0 : mxxMat1
  w_1130_0 : Fin 72000 → mxxMat1
  w_1130_1 : Fin 72000 → mxxMat1
  w_1130_2 : Fin 72000 → mxxMat1
  w_1130_3 : Fin 72000 → mxxMat1
  w_1130_4 : Fin 72000 → mxxMat1
  w_1130_5 : Fin 72000 → mxxMat1
  w_1131_0 : Fin 36000 → mxxMat1
  w_1131_1 : Fin 36000 → mxxMat1
  w_1131_2 : Fin 36000 → mxxMat1
  w_1131_3 : Fin 36000 → mxxMat1
  w_1131_4 : Fin 36000 → mxxMat1
  w_1131_5 : Fin 36000 → mxxMat1
  w_1134_0 : Fin 72000 → mxxMat1
  w_1134_1 : Fin 72000 → mxxMat1
  w_1134_2 : Fin 72000 → mxxMat1
  w_1134_3 : Fin 72000 → mxxMat1
  w_1134_4 : Fin 72000 → mxxMat1
  w_1134_5 : Fin 72000 → mxxMat1
  w_1135_0 : Fin 36000 → mxxMat1
  w_1135_1 : Fin 36000 → mxxMat1
  w_1135_2 : Fin 36000 → mxxMat1
  w_1135_3 : Fin 36000 → mxxMat1
  w_1135_4 : Fin 36000 → mxxMat1
  w_1135_5 : Fin 36000 → mxxMat1
  w_1136_0 : Fin 36000 → mxxMat1
  w_1136_1 : Fin 36000 → mxxMat1
  w_1136_2 : Fin 36000 → mxxMat1
  w_1136_3 : Fin 36000 → mxxMat1
  w_1136_4 : Fin 36000 → mxxMat1
  w_1136_5 : Fin 36000 → mxxMat1
  w_1137_0 : Fin 36000 → mxxMat1
  w_1137_1 : Fin 36000 → mxxMat1
  w_1137_2 : Fin 36000 → mxxMat1
  w_1137_3 : Fin 36000 → mxxMat1
  w_1137_4 : Fin 36000 → mxxMat1
  w_1137_5 : Fin 36000 → mxxMat1
  w_1138_0 : mxxMat1
  w_1139_0 : mxxMat1
  w_1141_0 : mxxMat1
  w_1142_0 : mxxMat1
  w_1144_0 : mxxMat1
  w_1145_0 : mxxMat1
  w_1147_0 : mxxMat1
  w_1148_0 : mxxMat1
  w_1150_0 : mxxMat1
  w_1151_0 : mxxMat1
  w_1153_0 : mxxMat1
  w_1154_0 : mxxMat1
  w_1156_0 : mxxMat1
  w_1156_1 : mxxMat1
  w_1156_2 : mxxMat1
  w_1156_3 : mxxMat1
  w_1156_4 : mxxMat1
  w_1156_5 : mxxMat1
  w_1157_0 : mxxMat8
  w_1159_0 : mxxMat8
  w_1161_0 : mxxMat8
  w_1163_0 : mxxMat8
  w_1165_0 : mxxMat8
  w_1167_0 : mxxMat8
  w_1168_0 : mxxMat8
  w_1168_1 : mxxMat8
  w_1168_2 : mxxMat8
  w_1168_3 : mxxMat8
  w_1168_4 : mxxMat8
  w_1168_5 : mxxMat8
  w_1172_0 : mxxMat1
  w_1175_0 : mxxMat1
  w_1178_0 : mxxMat1
  w_1181_0 : mxxMat1
  w_1188_0 : mxxMat1
  w_1191_0 : mxxMat1
  w_1198_0 : mxxMat1
  w_1201_0 : mxxMat1
  w_1208_0 : mxxMat1
  w_1211_0 : mxxMat1
  w_1218_0 : mxxMat1
  w_1221_0 : mxxMat1
  w_1231_0 : mxxMat1
  w_1236_0 : mxxMat1
  w_1241_0 : mxxMat1
  w_1246_0 : mxxMat1
  w_1251_0 : mxxMat1
  w_1256_0 : mxxMat1
  w_1261_0 : mxxMat1
  w_1266_0 : Fin 36 → mxxMat1
  w_1267_0 : Fin 36 → mxxMat2
  w_1268_0 : Fin 36 → mxxMat2
  w_1269_0 : Fin 36 → mxxMat2
  w_1270_0 : Fin 36 → mxxMat2
  w_1271_0 : Fin 36 → mxxMat2
  w_1272_0 : mxxMat1
  w_1273_0 : mxxMat1
  w_1276_0 : mxxMat1
  w_1277_0 : mxxMat1
  w_1280_0 : mxxMat2
  w_1281_0 : mxxMat4
  w_1282_0 : mxxMat4
  w_1284_0 : mxxMat1
  w_1285_0 : mxxMat1
  w_1286_0 : mxxMat2
  w_1287_0 : mxxMat4
  w_1288_0 : mxxMat4
  w_1290_0 : mxxMat1
  w_1292_0 : mxxMat1
  w_1294_0 : mxxMat1
  w_1295_0 : mxxMat1
  w_1298_0 : mxxMat1
  w_1299_0 : mxxMat1
  w_1302_0 : mxxMat2
  w_1303_0 : mxxMat4
  w_1304_0 : mxxMat4
  w_1306_0 : mxxMat1
  w_1307_0 : mxxMat1
  w_1308_0 : mxxMat2
  w_1309_0 : mxxMat4
  w_1310_0 : mxxMat4
  w_1312_0 : mxxMat1
  w_1314_0 : mxxMat1
  w_1316_0 : mxxMat1
  w_1317_0 : mxxMat1
  w_1320_0 : mxxMat1
  w_1321_0 : mxxMat1
  w_1324_0 : mxxMat1
  w_1325_0 : mxxMat1
  w_1326_0 : mxxMat5
  w_1329_0 : mxxMat1
  w_1330_0 : mxxMat1
  w_1333_0 : mxxMat5
  w_1335_0 : mxxMat1
  w_1336_0 : mxxMat1
  w_1337_0 : mxxMat1
  w_1338_0 : mxxMat5
  w_1341_0 : mxxMat5
  w_1345_0 : mxxMat1
  w_1346_0 : mxxMat1
  w_1347_0 : mxxMat1
  w_1352_0 : mxxMat5
  w_1355_0 : mxxMat1
  w_1356_0 : mxxMat1
  w_1357_0 : mxxMat5
  w_1361_0 : mxxMat1
  w_1362_0 : mxxMat1
  w_1365_0 : mxxMat5
  w_1367_0 : mxxMat1
  w_1368_0 : mxxMat1
  w_1369_0 : mxxMat5
  w_1372_0 : mxxMat1
  w_1373_0 : mxxMat1
  w_1376_0 : mxxMat5
  w_1378_0 : mxxMat1
  w_1379_0 : mxxMat1
  w_1380_0 : mxxMat5
  w_1383_0 : mxxMat1
  w_1384_0 : mxxMat1
  w_1387_0 : mxxMat5
  w_1389_0 : mxxMat1
  w_1390_0 : mxxMat1
  w_1391_0 : mxxMat1
  w_1392_0 : mxxMat1
  w_1393_0 : mxxMat5
  w_1394_0 : mxxMat2
  w_1395_0 : mxxMat4
  w_1396_0 : mxxMat4
  w_1398_0 : mxxMat1
  w_1399_0 : mxxMat1
  w_1400_0 : mxxMat2
  w_1401_0 : mxxMat4
  w_1402_0 : mxxMat4
  w_1404_0 : mxxMat1
  w_1406_0 : mxxMat1
  w_1408_0 : mxxMat1
  w_1409_0 : mxxMat1
  w_1412_0 : mxxMat1
  w_1413_0 : mxxMat1
  w_1416_0 : mxxMat2
  w_1417_0 : mxxMat4
  w_1418_0 : mxxMat4
  w_1420_0 : mxxMat1
  w_1421_0 : mxxMat1
  w_1422_0 : mxxMat2
  w_1423_0 : mxxMat4
  w_1424_0 : mxxMat4
  w_1426_0 : mxxMat1
  w_1428_0 : mxxMat1
  w_1430_0 : mxxMat1
  w_1431_0 : mxxMat1
  w_1434_0 : mxxMat1
  w_1435_0 : mxxMat1
  w_1438_0 : mxxMat2
  w_1439_0 : mxxMat4
  w_1440_0 : mxxMat4
  w_1442_0 : mxxMat1
  w_1443_0 : mxxMat1
  w_1444_0 : mxxMat2
  w_1445_0 : mxxMat4
  w_1446_0 : mxxMat4
  w_1448_0 : mxxMat1
  w_1450_0 : mxxMat1
  w_1452_0 : mxxMat1
  w_1453_0 : mxxMat1
  w_1456_0 : mxxMat1
  w_1457_0 : mxxMat1
  w_1460_0 : mxxMat2
  w_1461_0 : mxxMat4
  w_1462_0 : mxxMat4
  w_1464_0 : mxxMat1
  w_1465_0 : mxxMat1
  w_1466_0 : mxxMat2
  w_1467_0 : mxxMat4
  w_1468_0 : mxxMat4
  w_1470_0 : mxxMat1
  w_1472_0 : mxxMat1
  w_1474_0 : mxxMat1
  w_1475_0 : mxxMat1
  w_1478_0 : mxxMat1
  w_1479_0 : mxxMat1
  w_1482_0 : mxxMat2
  w_1483_0 : mxxMat4
  w_1484_0 : mxxMat4
  w_1486_0 : mxxMat1
  w_1487_0 : mxxMat1
  w_1488_0 : mxxMat2
  w_1489_0 : mxxMat4
  w_1490_0 : mxxMat4
  w_1492_0 : mxxMat1
  w_1494_0 : mxxMat1
  w_1496_0 : mxxMat1
  w_1497_0 : mxxMat1
  w_1500_0 : mxxMat1
  w_1501_0 : mxxMat1

abbrev generatedRoot.constraints_67 (w_19_0 : mxxMat1) (w_23_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_62_0 : mxxMat1) (w_346_0 : mxxMat1) (w_364_0 : mxxMat1) (w_383_0 : mxxMat1) (w_398_0 : mxxMat1) (w_407_0 : mxxMat1) (w_472_0 : mxxMat1) (w_493_0 : mxxMat1) (w_546_0 : mxxMat1) (w_561_0 : mxxMat1) (w_586_0 : mxxMat1) (w_601_0 : mxxMat1) (w_626_0 : mxxMat1) (w_641_0 : mxxMat1) (w_706_0 : mxxMat1) (w_762_0 : mxxMat1) (w_818_0 : mxxMat1) (w_874_0 : mxxMat1) (w_922_0 : mxxMat0) (w_928_0 : mxxMat0) (w_930_0 : mxxMat1) (w_986_0 : mxxMat1) (w_1042_0 : mxxMat1) (w_1114_0 : mxxMat6) (w_1114_1 : mxxMat6) (w_1114_2 : mxxMat6) (w_1114_3 : mxxMat6) (w_1114_4 : mxxMat6) (w_1114_5 : mxxMat6) (w_1168_0 : mxxMat8) (w_1168_1 : mxxMat8) (w_1168_2 : mxxMat8) (w_1168_3 : mxxMat8) (w_1168_4 : mxxMat8) (w_1168_5 : mxxMat8) (w_1173_0 : mxxMat1) (w_1176_0 : mxxMat1) (w_1178_0 : mxxMat1) (w_1181_0 : mxxMat1) (w_1188_0 : mxxMat1) (w_1191_0 : mxxMat1) (w_1198_0 : mxxMat1) (w_1201_0 : mxxMat1) (w_1208_0 : mxxMat1) (w_1211_0 : mxxMat1) (w_1218_0 : mxxMat1) (w_1221_0 : mxxMat1) (w_1232_0 : mxxMat1) (w_1237_0 : mxxMat1) (w_1242_0 : mxxMat1) (w_1247_0 : mxxMat1) (w_1252_0 : mxxMat1) (w_1257_0 : mxxMat1) (w_1262_0 : mxxMat1) (w_1267_0 : Fin 36 → mxxMat2) (w_1268_0 : Fin 36 → mxxMat2) (w_1269_0 : Fin 36 → mxxMat2) (w_1270_0 : Fin 36 → mxxMat2) (w_1271_0 : Fin 36 → mxxMat2) (w_1275_0 : mxxMat1) (w_1279_0 : mxxMat1) (w_1291_0 : mxxMat1) (w_1293_0 : mxxMat1) (w_1297_0 : mxxMat1) (w_1301_0 : mxxMat1) (w_1313_0 : mxxMat1) (w_1315_0 : mxxMat1) (w_1319_0 : mxxMat1) (w_1323_0 : mxxMat1) (w_1328_0 : mxxMat1) (w_1332_0 : mxxMat1) (w_1334_0 : mxxMat5) (w_1344_0 : mxxMat1) (w_1351_0 : mxxMat1) (w_1354_0 : mxxMat5) (w_1360_0 : mxxMat1) (w_1364_0 : mxxMat1) (w_1366_0 : mxxMat5) (w_1371_0 : mxxMat1) (w_1375_0 : mxxMat1) (w_1377_0 : mxxMat5) (w_1382_0 : mxxMat1) (w_1386_0 : mxxMat1) (w_1388_0 : mxxMat5) (w_1389_0 : mxxMat1) (w_1390_0 : mxxMat1) (w_1391_0 : mxxMat1) (w_1392_0 : mxxMat1) (w_1393_0 : mxxMat5) (w_1405_0 : mxxMat1) (w_1407_0 : mxxMat1) (w_1411_0 : mxxMat1) (w_1415_0 : mxxMat1) (w_1427_0 : mxxMat1) (w_1429_0 : mxxMat1) (w_1433_0 : mxxMat1) (w_1437_0 : mxxMat1) (w_1449_0 : mxxMat1) (w_1451_0 : mxxMat1) (w_1455_0 : mxxMat1) (w_1459_0 : mxxMat1) (w_1471_0 : mxxMat1) (w_1473_0 : mxxMat1) (w_1477_0 : mxxMat1) (w_1481_0 : mxxMat1) (w_1493_0 : mxxMat1) (w_1495_0 : mxxMat1) (w_1499_0 : mxxMat1) (w_1500_0 : mxxMat1) (w_1501_0 : mxxMat1) (w_1503_0 : mxxMat1) (outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_928_0 0 1 0 25 w_1500_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_922_0 0 1 0 25 w_1501_0 ∧
  outputs = (w_1267_0, (w_1268_0, (w_1269_0, (w_1270_0, (w_1271_0, (w_346_0, (w_364_0, (w_1173_0, (w_1176_0, (w_1275_0, (w_1279_0, (w_1168_0, (w_1168_1, (w_1168_2, (w_1168_3, (w_1168_4, (w_1168_5, (w_1291_0, (w_986_0, (w_1293_0, (w_1257_0, (w_1297_0, (w_1301_0, (w_1313_0, (w_1042_0, (w_1315_0, (w_1262_0, (w_1319_0, (w_1323_0, (w_1114_0, (w_1114_1, (w_1114_2, (w_1114_3, (w_1114_4, (w_1114_5, (w_383_0, (w_398_0, (w_1178_0, (w_1181_0, (w_1328_0, (w_1332_0, (w_1334_0, (w_472_0, (w_493_0, (w_1188_0, (w_1191_0, (w_1344_0, (w_1351_0, (w_1354_0, (w_546_0, (w_561_0, (w_1198_0, (w_1201_0, (w_1360_0, (w_1364_0, (w_1366_0, (w_586_0, (w_601_0, (w_1208_0, (w_1211_0, (w_1371_0, (w_1375_0, (w_1377_0, (w_626_0, (w_641_0, (w_1218_0, (w_1221_0, (w_1382_0, (w_1386_0, (w_1388_0, (w_17_0, (w_21_0, (w_18_0, (w_22_0, (w_19_0, (w_23_0, (w_1389_0, (w_62_0, (w_1390_0, (w_407_0, (w_1391_0, (w_1392_0, (w_1393_0, (w_54_0, (w_54_1, (w_54_2, (w_54_3, (w_54_4, (w_54_5, (w_1405_0, (w_706_0, (w_1407_0, (w_1232_0, (w_1411_0, (w_1415_0, (w_1427_0, (w_762_0, (w_1429_0, (w_1237_0, (w_1433_0, (w_1437_0, (w_1449_0, (w_818_0, (w_1451_0, (w_1242_0, (w_1455_0, (w_1459_0, (w_1471_0, (w_874_0, (w_1473_0, (w_1247_0, (w_1477_0, (w_1481_0, (w_1493_0, (w_930_0, (w_1495_0, (w_1252_0, (w_1499_0, (w_1503_0, ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

abbrev generatedRoot.constraints_66 (w_1485_0 : mxxMat0) (w_1489_0 : mxxMat4) (w_1490_0 : mxxMat4) (w_1491_0 : mxxMat0) (w_1492_0 : mxxMat1) (w_1494_0 : mxxMat1) (w_1496_0 : mxxMat1) (w_1497_0 : mxxMat1) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1489_0 w_1490_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1491_0 1 2 0 25 w_1492_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1491_0 2 3 0 25 w_1494_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1491_0 0 1 0 25 w_1496_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1485_0 0 1 0 25 w_1497_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3

abbrev generatedRoot.constraints_65 (backend : MxxRuntime.BackendContext) (w_866_0 : mxxMat0) (w_872_0 : mxxMat0) (w_899_6 : mxxMat1) (w_899_8 : mxxMat1) (w_1478_0 : mxxMat1) (w_1479_0 : mxxMat1) (w_1482_0 : mxxMat2) (w_1483_0 : mxxMat4) (w_1484_0 : mxxMat4) (w_1485_0 : mxxMat0) (w_1486_0 : mxxMat1) (w_1487_0 : mxxMat1) (w_1488_0 : mxxMat2) (w_1489_0 : mxxMat4) : Prop :=
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_872_0 0 1 0 25 w_1478_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_866_0 0 1 0 25 w_1479_0 ∧
  MxxRuntime.concatRows w_899_6 w_899_8 w_1482_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1482_0 w_1483_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1483_0 w_1484_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1485_0 1 2 0 25 w_1486_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1485_0 2 3 0 25 w_1487_0 ∧
  MxxRuntime.concatRows w_1486_0 w_1487_0 w_1488_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1488_0 w_1489_0

abbrev generatedRoot.constraints_64 (backend : MxxRuntime.BackendContext) (w_1463_0 : mxxMat0) (w_1464_0 : mxxMat1) (w_1465_0 : mxxMat1) (w_1466_0 : mxxMat2) (w_1467_0 : mxxMat4) (w_1468_0 : mxxMat4) (w_1469_0 : mxxMat0) (w_1470_0 : mxxMat1) (w_1472_0 : mxxMat1) (w_1474_0 : mxxMat1) (w_1475_0 : mxxMat1) : Prop :=
  MxxRuntime.concatRows w_1464_0 w_1465_0 w_1466_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1466_0 w_1467_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1467_0 w_1468_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1469_0 1 2 0 25 w_1470_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1469_0 2 3 0 25 w_1472_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1469_0 0 1 0 25 w_1474_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1463_0 0 1 0 25 w_1475_0 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_63 (backend : MxxRuntime.BackendContext) (w_810_0 : mxxMat0) (w_816_0 : mxxMat0) (w_843_6 : mxxMat1) (w_843_8 : mxxMat1) (w_1441_0 : mxxMat0) (w_1453_0 : mxxMat1) (w_1456_0 : mxxMat1) (w_1457_0 : mxxMat1) (w_1460_0 : mxxMat2) (w_1461_0 : mxxMat4) (w_1462_0 : mxxMat4) (w_1463_0 : mxxMat0) (w_1464_0 : mxxMat1) (w_1465_0 : mxxMat1) : Prop :=
  MxxRuntime.sliceMatrix w_1441_0 0 1 0 25 w_1453_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_816_0 0 1 0 25 w_1456_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_810_0 0 1 0 25 w_1457_0 ∧
  MxxRuntime.concatRows w_843_6 w_843_8 w_1460_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1460_0 w_1461_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1461_0 w_1462_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1463_0 1 2 0 25 w_1464_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1463_0 2 3 0 25 w_1465_0

abbrev generatedRoot.constraints_62 (backend : MxxRuntime.BackendContext) (w_1441_0 : mxxMat0) (w_1442_0 : mxxMat1) (w_1443_0 : mxxMat1) (w_1444_0 : mxxMat2) (w_1445_0 : mxxMat4) (w_1446_0 : mxxMat4) (w_1447_0 : mxxMat0) (w_1448_0 : mxxMat1) (w_1450_0 : mxxMat1) (w_1452_0 : mxxMat1) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1441_0 2 3 0 25 w_1443_0 ∧
  MxxRuntime.concatRows w_1442_0 w_1443_0 w_1444_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1444_0 w_1445_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1445_0 w_1446_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1447_0 1 2 0 25 w_1448_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1447_0 2 3 0 25 w_1450_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1447_0 0 1 0 25 w_1452_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev generatedRoot.constraints_61 (backend : MxxRuntime.BackendContext) (w_754_0 : mxxMat0) (w_760_0 : mxxMat0) (w_787_6 : mxxMat1) (w_787_8 : mxxMat1) (w_1419_0 : mxxMat0) (w_1431_0 : mxxMat1) (w_1434_0 : mxxMat1) (w_1435_0 : mxxMat1) (w_1438_0 : mxxMat2) (w_1439_0 : mxxMat4) (w_1440_0 : mxxMat4) (w_1441_0 : mxxMat0) (w_1442_0 : mxxMat1) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1419_0 0 1 0 25 w_1431_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_760_0 0 1 0 25 w_1434_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_754_0 0 1 0 25 w_1435_0 ∧
  MxxRuntime.concatRows w_787_6 w_787_8 w_1438_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1438_0 w_1439_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1439_0 w_1440_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1441_0 1 2 0 25 w_1442_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_60 (backend : MxxRuntime.BackendContext) (w_1419_0 : mxxMat0) (w_1420_0 : mxxMat1) (w_1421_0 : mxxMat1) (w_1422_0 : mxxMat2) (w_1423_0 : mxxMat4) (w_1424_0 : mxxMat4) (w_1425_0 : mxxMat0) (w_1426_0 : mxxMat1) (w_1428_0 : mxxMat1) (w_1430_0 : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1419_0 2 3 0 25 w_1421_0 ∧
  MxxRuntime.concatRows w_1420_0 w_1421_0 w_1422_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1422_0 w_1423_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1423_0 w_1424_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1425_0 1 2 0 25 w_1426_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1425_0 2 3 0 25 w_1428_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1425_0 0 1 0 25 w_1430_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_59 (backend : MxxRuntime.BackendContext) (w_698_0 : mxxMat0) (w_704_0 : mxxMat0) (w_731_6 : mxxMat1) (w_731_8 : mxxMat1) (w_1397_0 : mxxMat0) (w_1409_0 : mxxMat1) (w_1412_0 : mxxMat1) (w_1413_0 : mxxMat1) (w_1416_0 : mxxMat2) (w_1417_0 : mxxMat4) (w_1418_0 : mxxMat4) (w_1419_0 : mxxMat0) (w_1420_0 : mxxMat1) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1397_0 0 1 0 25 w_1409_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_704_0 0 1 0 25 w_1412_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_698_0 0 1 0 25 w_1413_0 ∧
  MxxRuntime.concatRows w_731_6 w_731_8 w_1416_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1416_0 w_1417_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1417_0 w_1418_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1419_0 1 2 0 25 w_1420_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3

abbrev generatedRoot.constraints_58 (backend : MxxRuntime.BackendContext) (w_1397_0 : mxxMat0) (w_1398_0 : mxxMat1) (w_1399_0 : mxxMat1) (w_1400_0 : mxxMat2) (w_1401_0 : mxxMat4) (w_1402_0 : mxxMat4) (w_1403_0 : mxxMat0) (w_1404_0 : mxxMat1) (w_1406_0 : mxxMat1) (w_1408_0 : mxxMat1) : Prop :=
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1397_0 2 3 0 25 w_1399_0 ∧
  MxxRuntime.concatRows w_1398_0 w_1399_0 w_1400_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1400_0 w_1401_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1401_0 w_1402_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1403_0 1 2 0 25 w_1404_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1403_0 2 3 0 25 w_1406_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1403_0 0 1 0 25 w_1408_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev generatedRoot.constraints_57 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_14_0 : Fin 2 → mxxMat1) (w_15_0 : Fin 2 → mxxMat1) (w_20_0 : Fin 2 → mxxMat5) (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_430_0 : Fin 4 → mxxMat5) (w_625_0 : mxxMat0) (w_640_0 : mxxMat0) (w_675_6 : mxxMat1) (w_675_8 : mxxMat1) (w_1378_0 : mxxMat1) (w_1379_0 : mxxMat1) (w_1380_0 : mxxMat5) (w_1383_0 : mxxMat1) (w_1384_0 : mxxMat1) (w_1387_0 : mxxMat5) (w_1389_0 : mxxMat1) (w_1390_0 : mxxMat1) (w_1391_0 : mxxMat1) (w_1392_0 : mxxMat1) (w_1393_0 : mxxMat5) (w_1394_0 : mxxMat2) (w_1395_0 : mxxMat4) (w_1396_0 : mxxMat4) (w_1397_0 : mxxMat0) (w_1398_0 : mxxMat1) : Prop :=
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_625_0 0 1 0 25 w_1378_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (2) w_1379_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (1) w_1380_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_640_0 0 1 0 25 w_1383_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (2) w_1384_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (2) w_1387_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_1389_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_1390_0 ∧
  MxxRuntime.familyGetStatic w_14_0 (1) w_1391_0 ∧
  MxxRuntime.familyGetStatic w_15_0 (1) w_1392_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (1) w_1393_0 ∧
  MxxRuntime.concatRows w_675_6 w_675_8 w_1394_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1394_0 w_1395_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1395_0 w_1396_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1397_0 1 2 0 25 w_1398_0 ∧
  0 ≤ 2

abbrev generatedRoot.constraints_56 (w_255_0 : Fin 2 → mxxMat1) (w_260_0 : Fin 2 → mxxMat1) (w_274_0 : Fin 2 → mxxMat5) (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_430_0 : Fin 4 → mxxMat5) (w_545_0 : mxxMat0) (w_560_0 : mxxMat0) (w_585_0 : mxxMat0) (w_600_0 : mxxMat0) (w_1355_0 : mxxMat1) (w_1356_0 : mxxMat1) (w_1357_0 : mxxMat5) (w_1361_0 : mxxMat1) (w_1362_0 : mxxMat1) (w_1365_0 : mxxMat5) (w_1367_0 : mxxMat1) (w_1368_0 : mxxMat1) (w_1369_0 : mxxMat5) (w_1372_0 : mxxMat1) (w_1373_0 : mxxMat1) (w_1376_0 : mxxMat5) : Prop :=
  MxxRuntime.sliceMatrix w_545_0 0 1 0 25 w_1355_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (3) w_1356_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (2) w_1357_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_560_0 0 1 0 25 w_1361_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (3) w_1362_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (3) w_1365_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_585_0 0 1 0 25 w_1367_0 ∧
  MxxRuntime.familyGetStatic w_255_0 (1) w_1368_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (1) w_1369_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_600_0 0 1 0 25 w_1372_0 ∧
  MxxRuntime.familyGetStatic w_260_0 (1) w_1373_0 ∧
  MxxRuntime.familyGetStatic w_274_0 (1) w_1376_0 ∧
  0 ≤ 0 ∧
  0 < 1

abbrev generatedRoot.constraints_55 (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_430_0 : Fin 4 → mxxMat5) (w_465_0 : mxxMat0) (w_486_0 : mxxMat0) (w_492_0 : mxxMat0) (w_1336_0 : mxxMat1) (w_1337_0 : mxxMat1) (w_1338_0 : mxxMat5) (w_1341_0 : mxxMat5) (w_1345_0 : mxxMat1) (w_1346_0 : mxxMat1) (w_1347_0 : mxxMat1) (w_1352_0 : mxxMat5) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_465_0 0 1 0 25 w_1336_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (3) w_1337_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (2) w_1338_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (1) w_1341_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_492_0 0 1 0 25 w_1345_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_486_0 0 1 0 25 w_1346_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (3) w_1347_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (3) w_1352_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev generatedRoot.constraints_54 (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_430_0 : Fin 4 → mxxMat5) (w_382_0 : mxxMat0) (w_397_0 : mxxMat0) (w_471_0 : mxxMat0) (w_1034_0 : mxxMat0) (w_1321_0 : mxxMat1) (w_1324_0 : mxxMat1) (w_1325_0 : mxxMat1) (w_1326_0 : mxxMat5) (w_1329_0 : mxxMat1) (w_1330_0 : mxxMat1) (w_1333_0 : mxxMat5) (w_1335_0 : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1034_0 0 1 0 25 w_1321_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_382_0 0 1 0 25 w_1324_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (3) w_1325_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (1) w_1326_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_397_0 0 1 0 25 w_1329_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (3) w_1330_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (3) w_1333_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_471_0 0 1 0 25 w_1335_0

abbrev generatedRoot.constraints_53 (w_1040_0 : mxxMat0) (w_1305_0 : mxxMat0) (w_1311_0 : mxxMat0) (w_1312_0 : mxxMat1) (w_1314_0 : mxxMat1) (w_1316_0 : mxxMat1) (w_1317_0 : mxxMat1) (w_1320_0 : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1311_0 1 2 0 25 w_1312_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1311_0 2 3 0 25 w_1314_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1311_0 0 1 0 25 w_1316_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1305_0 0 1 0 25 w_1317_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1040_0 0 1 0 25 w_1320_0

abbrev generatedRoot.constraints_52 (backend : MxxRuntime.BackendContext) (w_978_0 : mxxMat0) (w_984_0 : mxxMat0) (w_1011_6 : mxxMat1) (w_1011_8 : mxxMat1) (w_1298_0 : mxxMat1) (w_1299_0 : mxxMat1) (w_1302_0 : mxxMat2) (w_1303_0 : mxxMat4) (w_1304_0 : mxxMat4) (w_1305_0 : mxxMat0) (w_1306_0 : mxxMat1) (w_1307_0 : mxxMat1) (w_1308_0 : mxxMat2) (w_1309_0 : mxxMat4) (w_1310_0 : mxxMat4) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_984_0 0 1 0 25 w_1298_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_978_0 0 1 0 25 w_1299_0 ∧
  MxxRuntime.concatRows w_1011_6 w_1011_8 w_1302_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1302_0 w_1303_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1303_0 w_1304_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1305_0 1 2 0 25 w_1306_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1305_0 2 3 0 25 w_1307_0 ∧
  MxxRuntime.concatRows w_1306_0 w_1307_0 w_1308_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1308_0 w_1309_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1309_0 w_1310_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3

abbrev generatedRoot.constraints_51 (w_1283_0 : mxxMat0) (w_1289_0 : mxxMat0) (w_1290_0 : mxxMat1) (w_1292_0 : mxxMat1) (w_1294_0 : mxxMat1) (w_1295_0 : mxxMat1) : Prop :=
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1289_0 1 2 0 25 w_1290_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1289_0 2 3 0 25 w_1292_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1289_0 0 1 0 25 w_1294_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1283_0 0 1 0 25 w_1295_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_50 (backend : MxxRuntime.BackendContext) (w_356_0 : mxxMat0) (w_362_0 : mxxMat0) (w_955_6 : mxxMat1) (w_955_8 : mxxMat1) (w_1276_0 : mxxMat1) (w_1277_0 : mxxMat1) (w_1280_0 : mxxMat2) (w_1281_0 : mxxMat4) (w_1282_0 : mxxMat4) (w_1283_0 : mxxMat0) (w_1284_0 : mxxMat1) (w_1285_0 : mxxMat1) (w_1286_0 : mxxMat2) (w_1287_0 : mxxMat4) (w_1288_0 : mxxMat4) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_362_0 0 1 0 25 w_1276_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_356_0 0 1 0 25 w_1277_0 ∧
  MxxRuntime.concatRows w_955_6 w_955_8 w_1280_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1280_0 w_1281_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1281_0 w_1282_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1283_0 1 2 0 25 w_1284_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1283_0 2 3 0 25 w_1285_0 ∧
  MxxRuntime.concatRows w_1284_0 w_1285_0 w_1286_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1286_0 w_1287_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1287_0 w_1288_0 ∧
  0 ≤ 1

abbrev generatedRoot.constraints_49 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_55_0 : mxxMat1) (w_61_0 : mxxMat1) (w_68_0 : mxxMat1) (w_225_0 : Fin 36 → mxxMat2) (w_338_0 : mxxMat0) (w_344_0 : mxxMat0) (w_400_0 : mxxMat1) (w_401_0 : mxxMat1) (w_408_0 : mxxMat1) (w_409_0 : mxxMat1) (w_1040_0 : mxxMat0) (w_1056_0 : mxxMat1) (w_1057_0 : Fin 36 → mxxMat1) (w_1114_1 : mxxMat6) (w_1114_3 : mxxMat6) (w_1168_1 : mxxMat8) (w_1168_3 : mxxMat8) (w_1169_0 : mxxMat5) (w_1170_0 : mxxMat1) (w_1171_0 : mxxMat1) (w_1174_0 : mxxMat1) (w_1177_0 : mxxMat1) (w_1184_0 : mxxMat1) (w_1187_0 : mxxMat1) (w_1194_0 : mxxMat1) (w_1197_0 : mxxMat1) (w_1204_0 : mxxMat1) (w_1207_0 : mxxMat1) (w_1214_0 : mxxMat1) (w_1217_0 : mxxMat1) (w_1224_0 : mxxMat1) (w_1227_0 : mxxMat1) (w_1229_0 : mxxMat1) (w_1230_0 : mxxMat1) (w_1234_0 : mxxMat1) (w_1235_0 : mxxMat1) (w_1239_0 : mxxMat1) (w_1240_0 : mxxMat1) (w_1244_0 : mxxMat1) (w_1245_0 : mxxMat1) (w_1249_0 : mxxMat1) (w_1250_0 : mxxMat1) (w_1254_0 : mxxMat1) (w_1255_0 : mxxMat1) (w_1259_0 : mxxMat1) (w_1260_0 : mxxMat1) (w_1261_0 : mxxMat1) (w_1264_0 : mxxMat1) (w_1265_0 : mxxMat1) (w_1266_0 : Fin 36 → mxxMat1) (w_1267_0 : Fin 36 → mxxMat2) (w_1268_0 : Fin 36 → mxxMat2) (w_1269_0 : Fin 36 → mxxMat2) (w_1270_0 : Fin 36 → mxxMat2) (w_1271_0 : Fin 36 → mxxMat2) (w_1272_0 : mxxMat1) (w_1273_0 : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1040_0 2 3 0 25 w_1261_0 ∧
  MxxRuntime.familyPack (36) [w_1170_0, w_1171_0, w_401_0, w_409_0, w_1174_0, w_1177_0, w_1184_0, w_1187_0, w_1194_0, w_1197_0, w_1204_0, w_1207_0, w_1214_0, w_1217_0, w_1224_0, w_1227_0, w_1229_0, w_1230_0, w_1234_0, w_1235_0, w_1239_0, w_1240_0, w_1244_0, w_1245_0, w_1249_0, w_1250_0, w_1254_0, w_1255_0, w_1259_0, w_1260_0, w_1264_0, w_1265_0, w_1056_0, w_1056_0, w_1056_0, w_1056_0] w_1266_0 ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_1267 backend hashModel params i (w_61_0, (w_68_0, ((w_225_0 i), ((w_1057_0 i), (w_1114_1, (w_55_0, (w_1168_1, (w_1169_0, (w_400_0, (w_408_0, ((w_1266_0 i), (w_1114_3, (w_1168_3, ()))))))))))))) (w_1267_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_1268 backend hashModel params i (w_61_0, (w_68_0, ((w_225_0 i), ((w_1057_0 i), (w_1114_1, (w_55_0, (w_1168_1, (w_1169_0, (w_400_0, (w_408_0, ((w_1266_0 i), (w_1114_3, (w_1168_3, ()))))))))))))) (w_1268_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_1269 backend hashModel params i (w_61_0, (w_68_0, ((w_225_0 i), ((w_1057_0 i), (w_1114_1, (w_55_0, (w_1168_1, (w_1169_0, (w_400_0, (w_408_0, ((w_1266_0 i), (w_1114_3, (w_1168_3, ()))))))))))))) (w_1269_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_1270 backend hashModel params i (w_61_0, (w_68_0, ((w_225_0 i), ((w_1057_0 i), (w_1114_1, (w_55_0, (w_1168_1, (w_1169_0, (w_400_0, (w_408_0, ((w_1266_0 i), (w_1114_3, (w_1168_3, ()))))))))))))) (w_1270_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_1271 backend hashModel params i (w_61_0, (w_68_0, ((w_225_0 i), ((w_1057_0 i), (w_1114_1, (w_55_0, (w_1168_1, (w_1169_0, (w_400_0, (w_408_0, ((w_1266_0 i), (w_1114_3, (w_1168_3, ()))))))))))))) (w_1271_0 i)) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_344_0 0 1 0 25 w_1272_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_338_0 0 1 0 25 w_1273_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3

abbrev generatedRoot.constraints_48 (w_760_0 : mxxMat0) (w_816_0 : mxxMat0) (w_872_0 : mxxMat0) (w_928_0 : mxxMat0) (w_984_0 : mxxMat0) (w_1236_0 : mxxMat1) (w_1241_0 : mxxMat1) (w_1246_0 : mxxMat1) (w_1251_0 : mxxMat1) (w_1256_0 : mxxMat1) : Prop :=
  MxxRuntime.sliceMatrix w_760_0 2 3 0 25 w_1236_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_816_0 2 3 0 25 w_1241_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_872_0 2 3 0 25 w_1246_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_928_0 2 3 0 25 w_1251_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_984_0 2 3 0 25 w_1256_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3

abbrev generatedRoot.constraints_47 (w_600_0 : mxxMat0) (w_625_0 : mxxMat0) (w_640_0 : mxxMat0) (w_704_0 : mxxMat0) (w_1211_0 : mxxMat1) (w_1218_0 : mxxMat1) (w_1221_0 : mxxMat1) (w_1231_0 : mxxMat1) : Prop :=
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_600_0 2 3 0 25 w_1211_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_625_0 2 3 0 25 w_1218_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_640_0 2 3 0 25 w_1221_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_704_0 2 3 0 25 w_1231_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev generatedRoot.constraints_46 (w_471_0 : mxxMat0) (w_492_0 : mxxMat0) (w_545_0 : mxxMat0) (w_560_0 : mxxMat0) (w_585_0 : mxxMat0) (w_1188_0 : mxxMat1) (w_1191_0 : mxxMat1) (w_1198_0 : mxxMat1) (w_1201_0 : mxxMat1) (w_1208_0 : mxxMat1) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_471_0 2 3 0 25 w_1188_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_492_0 2 3 0 25 w_1191_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_545_0 2 3 0 25 w_1198_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_560_0 2 3 0 25 w_1201_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_585_0 2 3 0 25 w_1208_0 ∧
  0 ≤ 2 ∧
  2 < 3

abbrev generatedRoot.constraints_45 (w_344_0 : mxxMat0) (w_362_0 : mxxMat0) (w_382_0 : mxxMat0) (w_397_0 : mxxMat0) (w_1172_0 : mxxMat1) (w_1175_0 : mxxMat1) (w_1178_0 : mxxMat1) (w_1181_0 : mxxMat1) : Prop :=
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_344_0 2 3 0 25 w_1172_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_362_0 2 3 0 25 w_1175_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_382_0 2 3 0 25 w_1178_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_397_0 2 3 0 25 w_1181_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_44 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_1115_0 : mxxMat9) (w_1137_0 : Fin 36000 → mxxMat1) (w_1137_1 : Fin 36000 → mxxMat1) (w_1137_2 : Fin 36000 → mxxMat1) (w_1137_3 : Fin 36000 → mxxMat1) (w_1137_4 : Fin 36000 → mxxMat1) (w_1137_5 : Fin 36000 → mxxMat1) (w_1139_0 : mxxMat1) (w_1140_0 : mxxMat1) (w_1141_0 : mxxMat1) (w_1142_0 : mxxMat1) (w_1143_0 : mxxMat1) (w_1144_0 : mxxMat1) (w_1145_0 : mxxMat1) (w_1146_0 : mxxMat1) (w_1147_0 : mxxMat1) (w_1148_0 : mxxMat1) (w_1149_0 : mxxMat1) (w_1150_0 : mxxMat1) (w_1151_0 : mxxMat1) (w_1152_0 : mxxMat1) (w_1153_0 : mxxMat1) (w_1154_0 : mxxMat1) (w_1155_0 : mxxMat1) (w_1156_0 : mxxMat1) (w_1156_1 : mxxMat1) (w_1156_2 : mxxMat1) (w_1156_3 : mxxMat1) (w_1156_4 : mxxMat1) (w_1156_5 : mxxMat1) (w_1157_0 : mxxMat8) (w_1158_0 : mxxMat9) (w_1159_0 : mxxMat8) (w_1160_0 : mxxMat9) (w_1161_0 : mxxMat8) (w_1162_0 : mxxMat9) (w_1163_0 : mxxMat8) (w_1164_0 : mxxMat9) (w_1165_0 : mxxMat8) (w_1166_0 : mxxMat9) (w_1167_0 : mxxMat8) (w_1168_0 : mxxMat8) (w_1168_1 : mxxMat8) (w_1168_2 : mxxMat8) (w_1168_3 : mxxMat8) (w_1168_4 : mxxMat8) (w_1168_5 : mxxMat8) : Prop :=
  MxxRuntime.familyGetStatic w_1137_0 (1) w_1139_0 ∧
  MxxRuntime.familyGetStatic w_1137_1 (0) w_1141_0 ∧
  MxxRuntime.familyGetStatic w_1137_1 (1) w_1142_0 ∧
  MxxRuntime.familyGetStatic w_1137_2 (0) w_1144_0 ∧
  MxxRuntime.familyGetStatic w_1137_2 (1) w_1145_0 ∧
  MxxRuntime.familyGetStatic w_1137_3 (0) w_1147_0 ∧
  MxxRuntime.familyGetStatic w_1137_3 (1) w_1148_0 ∧
  MxxRuntime.familyGetStatic w_1137_4 (0) w_1150_0 ∧
  MxxRuntime.familyGetStatic w_1137_4 (1) w_1151_0 ∧
  MxxRuntime.familyGetStatic w_1137_5 (0) w_1153_0 ∧
  MxxRuntime.familyGetStatic w_1137_5 (1) w_1154_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_1156 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_1137_0, (w_1137_1, (w_1137_2, (w_1137_3, (w_1137_4, (w_1137_5, ())))))))))))) next) (Int.toNat (6)) (w_1140_0, (w_1143_0, (w_1146_0, (w_1149_0, (w_1152_0, (w_1155_0, ())))))) (w_1156_0, (w_1156_1, (w_1156_2, (w_1156_3, (w_1156_4, (w_1156_5, ())))))) ∧
  MxxRuntime.tensorRuns w_1115_0 w_1156_0 w_1157_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_1158_0 w_1156_1 w_1159_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_1160_0 w_1156_2 w_1161_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_1162_0 w_1156_3 w_1163_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_1164_0 w_1156_4 w_1165_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_1166_0 w_1156_5 w_1167_0 ∧
  0 ≤ (4499) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × Unit) => sequential_generatedRoot_1168 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_1137_0, (w_1137_1, (w_1137_2, (w_1137_3, (w_1137_4, (w_1137_5, ())))))))))))) next) (Int.toNat (4499)) (w_1157_0, (w_1159_0, (w_1161_0, (w_1163_0, (w_1165_0, (w_1167_0, ())))))) (w_1168_0, (w_1168_1, (w_1168_2, (w_1168_3, (w_1168_4, (w_1168_5, ())))))) ∧
  0 ≤ 2

abbrev generatedRoot.constraints_43 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat1) (w_12_0 : Fin 8 → mxxMat1) (w_24_0 : ByteArray) (w_255_0 : Fin 2 → mxxMat1) (w_260_0 : Fin 2 → mxxMat1) (w_274_0 : Fin 2 → mxxMat5) (w_1116_0 : Fin 8 → mxxMat1) (w_1117_0 : Fin 8 → mxxMat1) (w_1118_0 : Fin 8 → mxxMat1) (w_1119_0 : Fin 8 → mxxMat1) (w_1120_0 : Fin 72000 → Int) (w_1121_0 : Fin 72000 → Int) (w_1124_0 : mxxMat1) (w_1127_0 : mxxMat1) (w_1128_0 : Fin 72000 → mxxMat1) (w_1129_0 : Fin 72000 → mxxMat1) (w_1132_0 : Fin 72000 → mxxMat1) (w_1133_0 : Fin 72000 → mxxMat1) (w_1083_0 : Fin 7200 → mxxMat1) (w_1083_1 : Fin 7200 → mxxMat1) (w_1083_2 : Fin 7200 → mxxMat1) (w_1083_3 : Fin 7200 → mxxMat1) (w_1083_4 : Fin 7200 → mxxMat1) (w_1083_5 : Fin 7200 → mxxMat1) (w_1102_2 : mxxMat1) (w_1102_3 : mxxMat1) (w_1102_4 : mxxMat1) (w_1102_5 : mxxMat1) (w_1103_0 : mxxMat6) (w_1105_0 : mxxMat6) (w_1106_0 : mxxMat7) (w_1107_0 : mxxMat6) (w_1108_0 : mxxMat7) (w_1109_0 : mxxMat6) (w_1110_0 : mxxMat7) (w_1111_0 : mxxMat6) (w_1112_0 : mxxMat7) (w_1113_0 : mxxMat6) (w_1114_0 : mxxMat6) (w_1114_1 : mxxMat6) (w_1114_2 : mxxMat6) (w_1114_3 : mxxMat6) (w_1114_4 : mxxMat6) (w_1114_5 : mxxMat6) (w_1122_0 : mxxMat1) (w_1123_0 : mxxMat1) (w_1125_0 : mxxMat1) (w_1126_0 : mxxMat1) (w_1130_0 : Fin 72000 → mxxMat1) (w_1130_1 : Fin 72000 → mxxMat1) (w_1130_2 : Fin 72000 → mxxMat1) (w_1130_3 : Fin 72000 → mxxMat1) (w_1130_4 : Fin 72000 → mxxMat1) (w_1130_5 : Fin 72000 → mxxMat1) (w_1131_0 : Fin 36000 → mxxMat1) (w_1131_1 : Fin 36000 → mxxMat1) (w_1131_2 : Fin 36000 → mxxMat1) (w_1131_3 : Fin 36000 → mxxMat1) (w_1131_4 : Fin 36000 → mxxMat1) (w_1131_5 : Fin 36000 → mxxMat1) (w_1134_0 : Fin 72000 → mxxMat1) (w_1134_1 : Fin 72000 → mxxMat1) (w_1134_2 : Fin 72000 → mxxMat1) (w_1134_3 : Fin 72000 → mxxMat1) (w_1134_4 : Fin 72000 → mxxMat1) (w_1134_5 : Fin 72000 → mxxMat1) (w_1135_0 : Fin 36000 → mxxMat1) (w_1135_1 : Fin 36000 → mxxMat1) (w_1135_2 : Fin 36000 → mxxMat1) (w_1135_3 : Fin 36000 → mxxMat1) (w_1135_4 : Fin 36000 → mxxMat1) (w_1135_5 : Fin 36000 → mxxMat1) (w_1136_0 : Fin 36000 → mxxMat1) (w_1136_1 : Fin 36000 → mxxMat1) (w_1136_2 : Fin 36000 → mxxMat1) (w_1136_3 : Fin 36000 → mxxMat1) (w_1136_4 : Fin 36000 → mxxMat1) (w_1136_5 : Fin 36000 → mxxMat1) (w_1137_0 : Fin 36000 → mxxMat1) (w_1137_1 : Fin 36000 → mxxMat1) (w_1137_2 : Fin 36000 → mxxMat1) (w_1137_3 : Fin 36000 → mxxMat1) (w_1137_4 : Fin 36000 → mxxMat1) (w_1137_5 : Fin 36000 → mxxMat1) (w_1138_0 : mxxMat1) : Prop :=
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_1106_0 w_1102_2 w_1107_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_1108_0 w_1102_3 w_1109_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_1110_0 w_1102_4 w_1111_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_1112_0 w_1102_5 w_1113_0 ∧
  0 ≤ (899) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × Unit) => sequential_generatedRoot_1114 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_1083_0, (w_1083_1, (w_1083_2, (w_1083_3, (w_1083_4, (w_1083_5, ())))))))))))) next) (Int.toNat (899)) (w_1103_0, (w_1105_0, (w_1107_0, (w_1109_0, (w_1111_0, (w_1113_0, ())))))) (w_1114_0, (w_1114_1, (w_1114_2, (w_1114_3, (w_1114_4, (w_1114_5, ())))))) ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_1122_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_1123_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_1125_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_1126_0 ∧
  (72000) = 72000 ∧
  (∀ i : Fin 72000, ∃ w_255_0_gathered : mxxMat1, ∃ w_260_0_gathered : mxxMat1, ∃ w_274_0_gathered : mxxMat5, 0 ≤ (w_1121_0 i) ∧
  (w_1121_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_255_0 (w_1121_0 i) w_255_0_gathered ∧
  0 ≤ (w_1121_0 i) ∧
  (w_1121_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_260_0 (w_1121_0 i) w_260_0_gathered ∧
  0 ≤ (w_1121_0 i) ∧
  (w_1121_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_274_0 (w_1121_0 i) w_274_0_gathered ∧
  parallel_generatedRoot_1130 backend hashModel params i ((w_1121_0 i), (w_255_0_gathered, (w_5_0, (w_260_0_gathered, (w_1122_0, (w_1123_0, (w_1124_0, (w_274_0_gathered, (w_1125_0, (w_1126_0, (w_1127_0, (w_24_0, ((w_1128_0 i), ((w_1129_0 i), ())))))))))))))) ((w_1130_0 i), ((w_1130_1 i), ((w_1130_2 i), ((w_1130_3 i), ((w_1130_4 i), ((w_1130_5 i), ()))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_1131 backend hashModel params i (w_1130_0, (w_1130_1, (w_1130_2, (w_1130_3, (w_1130_4, (w_1130_5, ())))))) ((w_1131_0 i), ((w_1131_1 i), ((w_1131_2 i), ((w_1131_3 i), ((w_1131_4 i), ((w_1131_5 i), ()))))))) ∧
  (72000) = 72000 ∧
  (∀ i : Fin 72000, ∃ w_255_0_gathered : mxxMat1, ∃ w_260_0_gathered : mxxMat1, ∃ w_274_0_gathered : mxxMat5, 0 ≤ (w_1120_0 i) ∧
  (w_1120_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_255_0 (w_1120_0 i) w_255_0_gathered ∧
  0 ≤ (w_1120_0 i) ∧
  (w_1120_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_260_0 (w_1120_0 i) w_260_0_gathered ∧
  0 ≤ (w_1120_0 i) ∧
  (w_1120_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_274_0 (w_1120_0 i) w_274_0_gathered ∧
  parallel_generatedRoot_1134 backend hashModel params i ((w_1120_0 i), (w_255_0_gathered, (w_5_0, (w_260_0_gathered, (w_1131_0, (w_1131_2, (w_1131_4, (w_274_0_gathered, (w_1131_1, (w_1131_3, (w_1131_5, (w_24_0, ((w_1132_0 i), ((w_1133_0 i), ())))))))))))))) ((w_1134_0 i), ((w_1134_1 i), ((w_1134_2 i), ((w_1134_3 i), ((w_1134_4 i), ((w_1134_5 i), ()))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_1135 backend hashModel params i (w_1134_0, (w_1134_1, (w_1134_2, (w_1134_3, (w_1134_4, (w_1134_5, ())))))) ((w_1135_0 i), ((w_1135_1 i), ((w_1135_2 i), ((w_1135_3 i), ((w_1135_4 i), ((w_1135_5 i), ()))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_1136 backend hashModel params i (w_1116_0, (w_1117_0, (w_1118_0, (w_1119_0, (w_11_0, (w_5_0, (w_12_0, ((w_1135_4 i), ((w_1135_0 i), ((w_1135_2 i), ((w_1135_5 i), ((w_1135_1 i), ((w_1135_3 i), ()))))))))))))) ((w_1136_0 i), ((w_1136_1 i), ((w_1136_2 i), ((w_1136_3 i), ((w_1136_4 i), ((w_1136_5 i), ()))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_1137 backend hashModel params i ((w_1136_0 i), ((w_1136_1 i), ((w_1136_2 i), ((w_1136_3 i), ((w_1136_4 i), ((w_1136_5 i), ())))))) ((w_1137_0 i), ((w_1137_1 i), ((w_1137_2 i), ((w_1137_3 i), ((w_1137_4 i), ((w_1137_5 i), ()))))))) ∧
  MxxRuntime.familyGetStatic w_1137_0 (0) w_1138_0

abbrev generatedRoot.constraints_42 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_11_0 : Fin 8 → mxxMat1) (w_12_0 : Fin 8 → mxxMat1) (w_24_0 : ByteArray) (w_1059_0 : Fin 8 → mxxMat1) (w_1060_0 : Fin 8 → mxxMat1) (w_1061_0 : Fin 8 → mxxMat1) (w_1062_0 : Fin 8 → mxxMat1) (w_1063_0 : Fin 14400 → Int) (w_1064_0 : Fin 2 → mxxMat1) (w_1065_0 : Fin 2 → mxxMat1) (w_1066_0 : Fin 14400 → Int) (w_1069_0 : mxxMat1) (w_1070_0 : Fin 2 → mxxMat5) (w_1073_0 : mxxMat1) (w_1074_0 : Fin 14400 → mxxMat1) (w_1075_0 : Fin 14400 → mxxMat1) (w_1078_0 : Fin 14400 → mxxMat1) (w_1079_0 : Fin 14400 → mxxMat1) (w_1058_0 : mxxMat7) (w_1067_0 : mxxMat1) (w_1068_0 : mxxMat1) (w_1071_0 : mxxMat1) (w_1072_0 : mxxMat1) (w_1076_0 : Fin 14400 → mxxMat1) (w_1076_1 : Fin 14400 → mxxMat1) (w_1076_2 : Fin 14400 → mxxMat1) (w_1076_3 : Fin 14400 → mxxMat1) (w_1076_4 : Fin 14400 → mxxMat1) (w_1076_5 : Fin 14400 → mxxMat1) (w_1077_0 : Fin 7200 → mxxMat1) (w_1077_1 : Fin 7200 → mxxMat1) (w_1077_2 : Fin 7200 → mxxMat1) (w_1077_3 : Fin 7200 → mxxMat1) (w_1077_4 : Fin 7200 → mxxMat1) (w_1077_5 : Fin 7200 → mxxMat1) (w_1080_0 : Fin 14400 → mxxMat1) (w_1080_1 : Fin 14400 → mxxMat1) (w_1080_2 : Fin 14400 → mxxMat1) (w_1080_3 : Fin 14400 → mxxMat1) (w_1080_4 : Fin 14400 → mxxMat1) (w_1080_5 : Fin 14400 → mxxMat1) (w_1081_0 : Fin 7200 → mxxMat1) (w_1081_1 : Fin 7200 → mxxMat1) (w_1081_2 : Fin 7200 → mxxMat1) (w_1081_3 : Fin 7200 → mxxMat1) (w_1081_4 : Fin 7200 → mxxMat1) (w_1081_5 : Fin 7200 → mxxMat1) (w_1082_0 : Fin 7200 → mxxMat1) (w_1082_1 : Fin 7200 → mxxMat1) (w_1082_2 : Fin 7200 → mxxMat1) (w_1082_3 : Fin 7200 → mxxMat1) (w_1082_4 : Fin 7200 → mxxMat1) (w_1082_5 : Fin 7200 → mxxMat1) (w_1083_0 : Fin 7200 → mxxMat1) (w_1083_1 : Fin 7200 → mxxMat1) (w_1083_2 : Fin 7200 → mxxMat1) (w_1083_3 : Fin 7200 → mxxMat1) (w_1083_4 : Fin 7200 → mxxMat1) (w_1083_5 : Fin 7200 → mxxMat1) (w_1084_0 : mxxMat1) (w_1085_0 : mxxMat1) (w_1086_0 : mxxMat1) (w_1087_0 : mxxMat1) (w_1088_0 : mxxMat1) (w_1089_0 : mxxMat1) (w_1090_0 : mxxMat1) (w_1091_0 : mxxMat1) (w_1092_0 : mxxMat1) (w_1093_0 : mxxMat1) (w_1094_0 : mxxMat1) (w_1095_0 : mxxMat1) (w_1096_0 : mxxMat1) (w_1097_0 : mxxMat1) (w_1098_0 : mxxMat1) (w_1099_0 : mxxMat1) (w_1100_0 : mxxMat1) (w_1101_0 : mxxMat1) (w_1102_0 : mxxMat1) (w_1102_1 : mxxMat1) (w_1102_2 : mxxMat1) (w_1102_3 : mxxMat1) (w_1102_4 : mxxMat1) (w_1102_5 : mxxMat1) (w_1103_0 : mxxMat6) (w_1104_0 : mxxMat7) (w_1105_0 : mxxMat6) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_1072_0 ∧
  (14400) = 14400 ∧
  (∀ i : Fin 14400, ∃ w_1064_0_gathered : mxxMat1, ∃ w_1065_0_gathered : mxxMat1, ∃ w_1070_0_gathered : mxxMat5, 0 ≤ (w_1066_0 i) ∧
  (w_1066_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_1064_0 (w_1066_0 i) w_1064_0_gathered ∧
  0 ≤ (w_1066_0 i) ∧
  (w_1066_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_1065_0 (w_1066_0 i) w_1065_0_gathered ∧
  0 ≤ (w_1066_0 i) ∧
  (w_1066_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_1070_0 (w_1066_0 i) w_1070_0_gathered ∧
  parallel_generatedRoot_1076 backend hashModel params i ((w_1066_0 i), (w_1064_0_gathered, (w_5_0, (w_1065_0_gathered, (w_1067_0, (w_1068_0, (w_1069_0, (w_1070_0_gathered, (w_1071_0, (w_1072_0, (w_1073_0, (w_24_0, ((w_1074_0 i), ((w_1075_0 i), ())))))))))))))) ((w_1076_0 i), ((w_1076_1 i), ((w_1076_2 i), ((w_1076_3 i), ((w_1076_4 i), ((w_1076_5 i), ()))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_1077 backend hashModel params i (w_1076_0, (w_1076_1, (w_1076_2, (w_1076_3, (w_1076_4, (w_1076_5, ())))))) ((w_1077_0 i), ((w_1077_1 i), ((w_1077_2 i), ((w_1077_3 i), ((w_1077_4 i), ((w_1077_5 i), ()))))))) ∧
  (14400) = 14400 ∧
  (∀ i : Fin 14400, ∃ w_1064_0_gathered : mxxMat1, ∃ w_1065_0_gathered : mxxMat1, ∃ w_1070_0_gathered : mxxMat5, 0 ≤ (w_1063_0 i) ∧
  (w_1063_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_1064_0 (w_1063_0 i) w_1064_0_gathered ∧
  0 ≤ (w_1063_0 i) ∧
  (w_1063_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_1065_0 (w_1063_0 i) w_1065_0_gathered ∧
  0 ≤ (w_1063_0 i) ∧
  (w_1063_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_1070_0 (w_1063_0 i) w_1070_0_gathered ∧
  parallel_generatedRoot_1080 backend hashModel params i ((w_1063_0 i), (w_1064_0_gathered, (w_5_0, (w_1065_0_gathered, (w_1077_0, (w_1077_2, (w_1077_4, (w_1070_0_gathered, (w_1077_1, (w_1077_3, (w_1077_5, (w_24_0, ((w_1078_0 i), ((w_1079_0 i), ())))))))))))))) ((w_1080_0 i), ((w_1080_1 i), ((w_1080_2 i), ((w_1080_3 i), ((w_1080_4 i), ((w_1080_5 i), ()))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_1081 backend hashModel params i (w_1080_0, (w_1080_1, (w_1080_2, (w_1080_3, (w_1080_4, (w_1080_5, ())))))) ((w_1081_0 i), ((w_1081_1 i), ((w_1081_2 i), ((w_1081_3 i), ((w_1081_4 i), ((w_1081_5 i), ()))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_1082 backend hashModel params i (w_1059_0, (w_1060_0, (w_1061_0, (w_1062_0, (w_11_0, (w_5_0, (w_12_0, ((w_1081_4 i), ((w_1081_0 i), ((w_1081_2 i), ((w_1081_5 i), ((w_1081_1 i), ((w_1081_3 i), ()))))))))))))) ((w_1082_0 i), ((w_1082_1 i), ((w_1082_2 i), ((w_1082_3 i), ((w_1082_4 i), ((w_1082_5 i), ()))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_1083 backend hashModel params i ((w_1082_0 i), ((w_1082_1 i), ((w_1082_2 i), ((w_1082_3 i), ((w_1082_4 i), ((w_1082_5 i), ())))))) ((w_1083_0 i), ((w_1083_1 i), ((w_1083_2 i), ((w_1083_3 i), ((w_1083_4 i), ((w_1083_5 i), ()))))))) ∧
  MxxRuntime.familyGetStatic w_1083_0 (0) w_1084_0 ∧
  MxxRuntime.familyGetStatic w_1083_0 (1) w_1085_0 ∧
  MxxRuntime.familyGetStatic w_1083_1 (0) w_1087_0 ∧
  MxxRuntime.familyGetStatic w_1083_1 (1) w_1088_0 ∧
  MxxRuntime.familyGetStatic w_1083_2 (0) w_1090_0 ∧
  MxxRuntime.familyGetStatic w_1083_2 (1) w_1091_0 ∧
  MxxRuntime.familyGetStatic w_1083_3 (0) w_1093_0 ∧
  MxxRuntime.familyGetStatic w_1083_3 (1) w_1094_0 ∧
  MxxRuntime.familyGetStatic w_1083_4 (0) w_1096_0 ∧
  MxxRuntime.familyGetStatic w_1083_4 (1) w_1097_0 ∧
  MxxRuntime.familyGetStatic w_1083_5 (0) w_1099_0 ∧
  MxxRuntime.familyGetStatic w_1083_5 (1) w_1100_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_1102 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_1083_0, (w_1083_1, (w_1083_2, (w_1083_3, (w_1083_4, (w_1083_5, ())))))))))))) next) (Int.toNat (6)) (w_1086_0, (w_1089_0, (w_1092_0, (w_1095_0, (w_1098_0, (w_1101_0, ())))))) (w_1102_0, (w_1102_1, (w_1102_2, (w_1102_3, (w_1102_4, (w_1102_5, ())))))) ∧
  MxxRuntime.tensorRuns w_1058_0 w_1102_0 w_1103_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_1104_0 w_1102_1 w_1105_0 ∧
  0 ≤ (0)

abbrev generatedRoot.constraints_41 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_55_0 : mxxMat1) (w_229_0 : mxxMat1) (w_235_0 : mxxMat1) (w_236_0 : mxxMat1) (w_242_0 : mxxMat1) (w_243_0 : mxxMat1) (w_352_0 : mxxMat1) (w_365_0 : mxxMat1) (w_404_0 : mxxMat1) (w_412_0 : mxxMat1) (w_497_0 : mxxMat1) (w_502_0 : mxxMat1) (w_565_0 : mxxMat1) (w_570_0 : mxxMat1) (w_605_0 : mxxMat1) (w_610_0 : mxxMat1) (w_645_0 : mxxMat1) (w_650_0 : mxxMat1) (w_662_0 : mxxMat1) (w_663_0 : mxxMat1) (w_718_0 : mxxMat1) (w_719_0 : mxxMat1) (w_774_0 : mxxMat1) (w_775_0 : mxxMat1) (w_830_0 : mxxMat1) (w_831_0 : mxxMat1) (w_886_0 : mxxMat1) (w_887_0 : mxxMat1) (w_942_0 : mxxMat1) (w_943_0 : mxxMat1) (w_998_0 : mxxMat1) (w_999_0 : mxxMat1) (w_1034_0 : mxxMat0) (w_1035_0 : mxxMat1) (w_1036_0 : mxxMat1) (w_1037_0 : mxxMat2) (w_1038_0 : mxxMat4) (w_1039_0 : mxxMat4) (w_1040_0 : mxxMat0) (w_1041_0 : mxxMat1) (w_1043_0 : mxxMat1) (w_1045_0 : mxxMat1) (w_1046_0 : mxxMat14) (w_1047_0 : mxxMat14) (w_1049_0 : mxxMat1) (w_1051_0 : mxxMat1) (w_1052_0 : mxxMat14) (w_1053_0 : mxxMat14) (w_1054_0 : mxxMat1) (w_1055_0 : mxxMat1) (w_1056_0 : mxxMat1) (w_1057_0 : Fin 36 → mxxMat1) (w_1067_0 : mxxMat1) (w_1068_0 : mxxMat1) (w_1071_0 : mxxMat1) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1034_0 1 2 0 25 w_1035_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1034_0 2 3 0 25 w_1036_0 ∧
  MxxRuntime.concatRows w_1035_0 w_1036_0 w_1037_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1037_0 w_1038_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1038_0 w_1039_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_1040_0 1 2 0 25 w_1041_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_1043_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1045_0 w_1046_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1046_0 w_1047_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_1049_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1051_0 w_1052_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1052_0 w_1053_0 ∧
  MxxRuntime.familyPack (36) [w_235_0, w_236_0, w_242_0, w_243_0, w_352_0, w_365_0, w_404_0, w_412_0, w_497_0, w_502_0, w_565_0, w_570_0, w_605_0, w_610_0, w_645_0, w_650_0, w_662_0, w_663_0, w_718_0, w_719_0, w_774_0, w_775_0, w_830_0, w_831_0, w_886_0, w_887_0, w_942_0, w_943_0, w_998_0, w_999_0, w_1054_0, w_1055_0, w_1056_0, w_1056_0, w_1056_0, w_1056_0] w_1057_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_1067_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_1068_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_1071_0

abbrev generatedRoot.constraints_40 (backend : MxxRuntime.BackendContext) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_1011_7 : mxxMat1) (w_1011_9 : mxxMat1) (w_1016_0 : mxxMat0) (w_1019_0 : mxxMat1) (w_1020_0 : mxxMat0) (w_1020_concat_1 : mxxMat2) (w_1021_0 : mxxMat3) (w_1022_0 : mxxMat1) (w_1023_0 : mxxMat1) (w_1024_0 : mxxMat1) (w_1025_0 : mxxMat0) (w_1025_concat_1 : mxxMat2) (w_1026_0 : mxxMat1) (w_1027_0 : mxxMat1) (w_1028_0 : mxxMat1) (w_1029_0 : mxxMat0) (w_1029_concat_1 : mxxMat2) (w_1030_0 : mxxMat3) (w_1031_0 : mxxMat2) (w_1032_0 : mxxMat4) (w_1033_0 : mxxMat4) : Prop :=
  MxxRuntime.concatRows w_1020_concat_1 w_1019_0 w_1020_0 ∧
  MxxRuntime.concatColumns w_1016_0 w_1020_0 w_1021_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_4 w_1022_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_0 w_1023_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_2 w_1024_0 ∧
  MxxRuntime.concatRows w_1022_0 w_1023_0 w_1025_concat_1 ∧
  MxxRuntime.concatRows w_1025_concat_1 w_1024_0 w_1025_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_5 w_1026_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_1 w_1027_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(15)])) w_54_3 w_1028_0 ∧
  MxxRuntime.concatRows w_1026_0 w_1027_0 w_1029_concat_1 ∧
  MxxRuntime.concatRows w_1029_concat_1 w_1028_0 w_1029_0 ∧
  MxxRuntime.concatColumns w_1025_0 w_1029_0 w_1030_0 ∧
  MxxRuntime.concatRows w_1011_7 w_1011_9 w_1031_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_1031_0 w_1032_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_1032_0 w_1033_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_39 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_1001_0 : Fin 1 → mxxMat1) (w_1003_0 : Fin 1 → mxxMat1) (w_1005_0 : Fin 1 → mxxMat1) (w_1007_0 : Fin 1 → mxxMat1) (w_1009_0 : Fin 1 → mxxMat1) (w_1010_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_55_0 : mxxMat1) (w_229_0 : mxxMat1) (w_981_0 : mxxMat2) (w_982_0 : mxxMat4) (w_983_0 : mxxMat4) (w_984_0 : mxxMat0) (w_985_0 : mxxMat1) (w_987_0 : mxxMat1) (w_989_0 : mxxMat1) (w_990_0 : mxxMat14) (w_991_0 : mxxMat14) (w_993_0 : mxxMat1) (w_995_0 : mxxMat1) (w_996_0 : mxxMat14) (w_997_0 : mxxMat14) (w_1000_0 : mxxMat1) (w_1002_0 : mxxMat1) (w_1004_0 : mxxMat1) (w_1006_0 : mxxMat1) (w_1008_0 : mxxMat1) (w_1011_0 : mxxMat1) (w_1011_1 : mxxMat1) (w_1011_2 : mxxMat1) (w_1011_3 : mxxMat1) (w_1011_4 : mxxMat1) (w_1011_5 : mxxMat1) (w_1011_6 : mxxMat1) (w_1011_7 : mxxMat1) (w_1011_8 : mxxMat1) (w_1011_9 : mxxMat1) (w_1012_0 : mxxMat1) (w_1013_0 : Int) (w_1014_0 : mxxMat1) (w_1015_0 : mxxMat1) (w_1016_0 : mxxMat0) (w_1016_concat_1 : mxxMat2) (w_1017_0 : mxxMat1) (w_1018_0 : mxxMat1) (w_1019_0 : mxxMat1) (w_1020_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_981_0 w_982_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_982_0 w_983_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_984_0 1 2 0 25 w_985_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_987_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_989_0 w_990_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_990_0 w_991_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_993_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_995_0 w_996_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_996_0 w_997_0 ∧
  MxxRuntime.familyGetStatic w_1001_0 (0) w_1002_0 ∧
  MxxRuntime.familyGetStatic w_1003_0 (0) w_1004_0 ∧
  MxxRuntime.familyGetStatic w_1005_0 (0) w_1006_0 ∧
  MxxRuntime.familyGetStatic w_1007_0 (0) w_1008_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_1011 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_1009_0, (w_5_0, (w_1010_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_1001_0, (w_1003_0, (w_1005_0, (w_1007_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_1000_0, (w_1000_0, (w_1000_0, (w_1000_0, (w_1000_0, (w_1000_0, (w_1002_0, (w_1004_0, (w_1006_0, (w_1008_0, ())))))))))) (w_1011_0, (w_1011_1, (w_1011_2, (w_1011_3, (w_1011_4, (w_1011_5, (w_1011_6, (w_1011_7, (w_1011_8, (w_1011_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_1009_0 (0) w_1012_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_1013_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_1014_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_1013_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_1015_0 ∧
  MxxRuntime.concatRows w_1012_0 w_1014_0 w_1016_concat_1 ∧
  MxxRuntime.concatRows w_1016_concat_1 w_1015_0 w_1016_0 ∧
  MxxRuntime.familyGetStatic w_1010_0 (0) w_1017_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_1013_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_1018_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_1013_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_1019_0 ∧
  MxxRuntime.concatRows w_1017_0 w_1018_0 w_1020_concat_1

abbrev generatedRoot.constraints_38 (backend : MxxRuntime.BackendContext) (w_54_1 : mxxMat1) (w_54_3 : mxxMat1) (w_54_5 : mxxMat1) (w_955_7 : mxxMat1) (w_955_9 : mxxMat1) (w_966_0 : mxxMat1) (w_967_0 : mxxMat1) (w_968_0 : mxxMat1) (w_969_0 : mxxMat0) (w_969_concat_1 : mxxMat2) (w_970_0 : mxxMat1) (w_971_0 : mxxMat1) (w_972_0 : mxxMat1) (w_973_0 : mxxMat0) (w_973_concat_1 : mxxMat2) (w_974_0 : mxxMat3) (w_975_0 : mxxMat2) (w_976_0 : mxxMat4) (w_977_0 : mxxMat4) (w_978_0 : mxxMat0) (w_979_0 : mxxMat1) (w_980_0 : mxxMat1) (w_981_0 : mxxMat2) : Prop :=
  MxxRuntime.concatRows w_966_0 w_967_0 w_969_concat_1 ∧
  MxxRuntime.concatRows w_969_concat_1 w_968_0 w_969_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_5 w_970_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_1 w_971_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_3 w_972_0 ∧
  MxxRuntime.concatRows w_970_0 w_971_0 w_973_concat_1 ∧
  MxxRuntime.concatRows w_973_concat_1 w_972_0 w_973_0 ∧
  MxxRuntime.concatColumns w_969_0 w_973_0 w_974_0 ∧
  MxxRuntime.concatRows w_955_7 w_955_9 w_975_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_975_0 w_976_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_976_0 w_977_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_978_0 1 2 0 25 w_979_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_978_0 2 3 0 25 w_980_0 ∧
  MxxRuntime.concatRows w_979_0 w_980_0 w_981_0

abbrev generatedRoot.constraints_37 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_945_0 : Fin 1 → mxxMat1) (w_947_0 : Fin 1 → mxxMat1) (w_949_0 : Fin 1 → mxxMat1) (w_951_0 : Fin 1 → mxxMat1) (w_953_0 : Fin 1 → mxxMat1) (w_954_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_229_0 : mxxMat1) (w_934_0 : mxxMat14) (w_935_0 : mxxMat14) (w_937_0 : mxxMat1) (w_939_0 : mxxMat1) (w_940_0 : mxxMat14) (w_941_0 : mxxMat14) (w_944_0 : mxxMat1) (w_946_0 : mxxMat1) (w_948_0 : mxxMat1) (w_950_0 : mxxMat1) (w_952_0 : mxxMat1) (w_955_0 : mxxMat1) (w_955_1 : mxxMat1) (w_955_2 : mxxMat1) (w_955_3 : mxxMat1) (w_955_4 : mxxMat1) (w_955_5 : mxxMat1) (w_955_6 : mxxMat1) (w_955_7 : mxxMat1) (w_955_8 : mxxMat1) (w_955_9 : mxxMat1) (w_956_0 : mxxMat1) (w_957_0 : Int) (w_958_0 : mxxMat1) (w_959_0 : mxxMat1) (w_960_0 : mxxMat0) (w_960_concat_1 : mxxMat2) (w_961_0 : mxxMat1) (w_962_0 : mxxMat1) (w_963_0 : mxxMat1) (w_964_0 : mxxMat0) (w_964_concat_1 : mxxMat2) (w_965_0 : mxxMat3) (w_966_0 : mxxMat1) (w_967_0 : mxxMat1) (w_968_0 : mxxMat1) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 2048 w_934_0 w_935_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_937_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_939_0 w_940_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_940_0 w_941_0 ∧
  MxxRuntime.familyGetStatic w_945_0 (0) w_946_0 ∧
  MxxRuntime.familyGetStatic w_947_0 (0) w_948_0 ∧
  MxxRuntime.familyGetStatic w_949_0 (0) w_950_0 ∧
  MxxRuntime.familyGetStatic w_951_0 (0) w_952_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_955 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_953_0, (w_5_0, (w_954_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_945_0, (w_947_0, (w_949_0, (w_951_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_944_0, (w_944_0, (w_944_0, (w_944_0, (w_944_0, (w_944_0, (w_946_0, (w_948_0, (w_950_0, (w_952_0, ())))))))))) (w_955_0, (w_955_1, (w_955_2, (w_955_3, (w_955_4, (w_955_5, (w_955_6, (w_955_7, (w_955_8, (w_955_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_953_0 (0) w_956_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_957_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_958_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_957_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_959_0 ∧
  MxxRuntime.concatRows w_956_0 w_958_0 w_960_concat_1 ∧
  MxxRuntime.concatRows w_960_concat_1 w_959_0 w_960_0 ∧
  MxxRuntime.familyGetStatic w_954_0 (0) w_961_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_957_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_962_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_957_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_963_0 ∧
  MxxRuntime.concatRows w_961_0 w_962_0 w_964_concat_1 ∧
  MxxRuntime.concatRows w_964_concat_1 w_963_0 w_964_0 ∧
  MxxRuntime.concatColumns w_960_0 w_964_0 w_965_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_4 w_966_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_0 w_967_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(13)])) w_54_2 w_968_0

abbrev generatedRoot.constraints_36 (backend : MxxRuntime.BackendContext) (w_55_0 : mxxMat1) (w_899_7 : mxxMat1) (w_899_9 : mxxMat1) (w_913_0 : mxxMat0) (w_914_0 : mxxMat1) (w_915_0 : mxxMat1) (w_916_0 : mxxMat1) (w_917_0 : mxxMat0) (w_917_concat_1 : mxxMat2) (w_918_0 : mxxMat3) (w_919_0 : mxxMat2) (w_920_0 : mxxMat4) (w_921_0 : mxxMat4) (w_922_0 : mxxMat0) (w_923_0 : mxxMat1) (w_924_0 : mxxMat1) (w_925_0 : mxxMat2) (w_926_0 : mxxMat4) (w_927_0 : mxxMat4) (w_928_0 : mxxMat0) (w_929_0 : mxxMat1) (w_931_0 : mxxMat1) (w_933_0 : mxxMat1) (w_934_0 : mxxMat14) : Prop :=
  MxxRuntime.concatRows w_914_0 w_915_0 w_917_concat_1 ∧
  MxxRuntime.concatRows w_917_concat_1 w_916_0 w_917_0 ∧
  MxxRuntime.concatColumns w_913_0 w_917_0 w_918_0 ∧
  MxxRuntime.concatRows w_899_7 w_899_9 w_919_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_919_0 w_920_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_920_0 w_921_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_922_0 1 2 0 25 w_923_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_922_0 2 3 0 25 w_924_0 ∧
  MxxRuntime.concatRows w_923_0 w_924_0 w_925_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_925_0 w_926_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_926_0 w_927_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_928_0 1 2 0 25 w_929_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_931_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_933_0 w_934_0

abbrev generatedRoot.constraints_35 (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_898_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_900_0 : mxxMat1) (w_901_0 : Int) (w_902_0 : mxxMat1) (w_903_0 : mxxMat1) (w_904_0 : mxxMat0) (w_904_concat_1 : mxxMat2) (w_905_0 : mxxMat1) (w_906_0 : mxxMat1) (w_907_0 : mxxMat1) (w_908_0 : mxxMat0) (w_908_concat_1 : mxxMat2) (w_909_0 : mxxMat3) (w_910_0 : mxxMat1) (w_911_0 : mxxMat1) (w_912_0 : mxxMat1) (w_913_0 : mxxMat0) (w_913_concat_1 : mxxMat2) (w_914_0 : mxxMat1) (w_915_0 : mxxMat1) (w_916_0 : mxxMat1) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_901_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_902_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_901_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_903_0 ∧
  MxxRuntime.concatRows w_900_0 w_902_0 w_904_concat_1 ∧
  MxxRuntime.concatRows w_904_concat_1 w_903_0 w_904_0 ∧
  MxxRuntime.familyGetStatic w_898_0 (0) w_905_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_901_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_906_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_901_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_907_0 ∧
  MxxRuntime.concatRows w_905_0 w_906_0 w_908_concat_1 ∧
  MxxRuntime.concatRows w_908_concat_1 w_907_0 w_908_0 ∧
  MxxRuntime.concatColumns w_904_0 w_908_0 w_909_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_4 w_910_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_0 w_911_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_2 w_912_0 ∧
  MxxRuntime.concatRows w_910_0 w_911_0 w_913_concat_1 ∧
  MxxRuntime.concatRows w_913_concat_1 w_912_0 w_913_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_5 w_914_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_1 w_915_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(11)])) w_54_3 w_916_0

abbrev generatedRoot.constraints_34 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_889_0 : Fin 1 → mxxMat1) (w_891_0 : Fin 1 → mxxMat1) (w_893_0 : Fin 1 → mxxMat1) (w_895_0 : Fin 1 → mxxMat1) (w_897_0 : Fin 1 → mxxMat1) (w_898_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_55_0 : mxxMat1) (w_229_0 : mxxMat1) (w_866_0 : mxxMat0) (w_867_0 : mxxMat1) (w_868_0 : mxxMat1) (w_869_0 : mxxMat2) (w_870_0 : mxxMat4) (w_871_0 : mxxMat4) (w_872_0 : mxxMat0) (w_873_0 : mxxMat1) (w_875_0 : mxxMat1) (w_877_0 : mxxMat1) (w_878_0 : mxxMat14) (w_879_0 : mxxMat14) (w_881_0 : mxxMat1) (w_883_0 : mxxMat1) (w_884_0 : mxxMat14) (w_885_0 : mxxMat14) (w_888_0 : mxxMat1) (w_890_0 : mxxMat1) (w_892_0 : mxxMat1) (w_894_0 : mxxMat1) (w_896_0 : mxxMat1) (w_899_0 : mxxMat1) (w_899_1 : mxxMat1) (w_899_2 : mxxMat1) (w_899_3 : mxxMat1) (w_899_4 : mxxMat1) (w_899_5 : mxxMat1) (w_899_6 : mxxMat1) (w_899_7 : mxxMat1) (w_899_8 : mxxMat1) (w_899_9 : mxxMat1) (w_900_0 : mxxMat1) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_866_0 1 2 0 25 w_867_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_866_0 2 3 0 25 w_868_0 ∧
  MxxRuntime.concatRows w_867_0 w_868_0 w_869_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_869_0 w_870_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_870_0 w_871_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_872_0 1 2 0 25 w_873_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_875_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_877_0 w_878_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_878_0 w_879_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_881_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_883_0 w_884_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_884_0 w_885_0 ∧
  MxxRuntime.familyGetStatic w_889_0 (0) w_890_0 ∧
  MxxRuntime.familyGetStatic w_891_0 (0) w_892_0 ∧
  MxxRuntime.familyGetStatic w_893_0 (0) w_894_0 ∧
  MxxRuntime.familyGetStatic w_895_0 (0) w_896_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_899 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_897_0, (w_5_0, (w_898_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_889_0, (w_891_0, (w_893_0, (w_895_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_888_0, (w_888_0, (w_888_0, (w_888_0, (w_888_0, (w_888_0, (w_890_0, (w_892_0, (w_894_0, (w_896_0, ())))))))))) (w_899_0, (w_899_1, (w_899_2, (w_899_3, (w_899_4, (w_899_5, (w_899_6, (w_899_7, (w_899_8, (w_899_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_897_0 (0) w_900_0

abbrev generatedRoot.constraints_33 (backend : MxxRuntime.BackendContext) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_843_7 : mxxMat1) (w_843_9 : mxxMat1) (w_848_0 : mxxMat0) (w_852_0 : mxxMat0) (w_853_0 : mxxMat3) (w_854_0 : mxxMat1) (w_855_0 : mxxMat1) (w_856_0 : mxxMat1) (w_857_0 : mxxMat0) (w_857_concat_1 : mxxMat2) (w_858_0 : mxxMat1) (w_859_0 : mxxMat1) (w_860_0 : mxxMat1) (w_861_0 : mxxMat0) (w_861_concat_1 : mxxMat2) (w_862_0 : mxxMat3) (w_863_0 : mxxMat2) (w_864_0 : mxxMat4) (w_865_0 : mxxMat4) : Prop :=
  MxxRuntime.concatColumns w_848_0 w_852_0 w_853_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_4 w_854_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_0 w_855_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_2 w_856_0 ∧
  MxxRuntime.concatRows w_854_0 w_855_0 w_857_concat_1 ∧
  MxxRuntime.concatRows w_857_concat_1 w_856_0 w_857_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_5 w_858_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_1 w_859_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(9)])) w_54_3 w_860_0 ∧
  MxxRuntime.concatRows w_858_0 w_859_0 w_861_concat_1 ∧
  MxxRuntime.concatRows w_861_concat_1 w_860_0 w_861_0 ∧
  MxxRuntime.concatColumns w_857_0 w_861_0 w_862_0 ∧
  MxxRuntime.concatRows w_843_7 w_843_9 w_863_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_863_0 w_864_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_864_0 w_865_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_32 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_833_0 : Fin 1 → mxxMat1) (w_835_0 : Fin 1 → mxxMat1) (w_837_0 : Fin 1 → mxxMat1) (w_839_0 : Fin 1 → mxxMat1) (w_841_0 : Fin 1 → mxxMat1) (w_842_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_55_0 : mxxMat1) (w_229_0 : mxxMat1) (w_814_0 : mxxMat4) (w_815_0 : mxxMat4) (w_816_0 : mxxMat0) (w_817_0 : mxxMat1) (w_819_0 : mxxMat1) (w_821_0 : mxxMat1) (w_822_0 : mxxMat14) (w_823_0 : mxxMat14) (w_825_0 : mxxMat1) (w_827_0 : mxxMat1) (w_828_0 : mxxMat14) (w_829_0 : mxxMat14) (w_832_0 : mxxMat1) (w_834_0 : mxxMat1) (w_836_0 : mxxMat1) (w_838_0 : mxxMat1) (w_840_0 : mxxMat1) (w_843_0 : mxxMat1) (w_843_1 : mxxMat1) (w_843_2 : mxxMat1) (w_843_3 : mxxMat1) (w_843_4 : mxxMat1) (w_843_5 : mxxMat1) (w_843_6 : mxxMat1) (w_843_7 : mxxMat1) (w_843_8 : mxxMat1) (w_843_9 : mxxMat1) (w_844_0 : mxxMat1) (w_845_0 : Int) (w_846_0 : mxxMat1) (w_847_0 : mxxMat1) (w_848_0 : mxxMat0) (w_848_concat_1 : mxxMat2) (w_849_0 : mxxMat1) (w_850_0 : mxxMat1) (w_851_0 : mxxMat1) (w_852_0 : mxxMat0) (w_852_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 2048 w_814_0 w_815_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_816_0 1 2 0 25 w_817_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_819_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_821_0 w_822_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_822_0 w_823_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_825_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_827_0 w_828_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_828_0 w_829_0 ∧
  MxxRuntime.familyGetStatic w_833_0 (0) w_834_0 ∧
  MxxRuntime.familyGetStatic w_835_0 (0) w_836_0 ∧
  MxxRuntime.familyGetStatic w_837_0 (0) w_838_0 ∧
  MxxRuntime.familyGetStatic w_839_0 (0) w_840_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_843 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_841_0, (w_5_0, (w_842_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_833_0, (w_835_0, (w_837_0, (w_839_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_832_0, (w_832_0, (w_832_0, (w_832_0, (w_832_0, (w_832_0, (w_834_0, (w_836_0, (w_838_0, (w_840_0, ())))))))))) (w_843_0, (w_843_1, (w_843_2, (w_843_3, (w_843_4, (w_843_5, (w_843_6, (w_843_7, (w_843_8, (w_843_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_841_0 (0) w_844_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_845_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_846_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_845_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_847_0 ∧
  MxxRuntime.concatRows w_844_0 w_846_0 w_848_concat_1 ∧
  MxxRuntime.concatRows w_848_concat_1 w_847_0 w_848_0 ∧
  MxxRuntime.familyGetStatic w_842_0 (0) w_849_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_845_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_850_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_845_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_851_0 ∧
  MxxRuntime.concatRows w_849_0 w_850_0 w_852_concat_1 ∧
  MxxRuntime.concatRows w_852_concat_1 w_851_0 w_852_0

abbrev generatedRoot.constraints_31 (backend : MxxRuntime.BackendContext) (w_54_1 : mxxMat1) (w_54_3 : mxxMat1) (w_54_5 : mxxMat1) (w_787_7 : mxxMat1) (w_787_9 : mxxMat1) (w_800_0 : mxxMat1) (w_801_0 : mxxMat0) (w_801_concat_1 : mxxMat2) (w_802_0 : mxxMat1) (w_803_0 : mxxMat1) (w_804_0 : mxxMat1) (w_805_0 : mxxMat0) (w_805_concat_1 : mxxMat2) (w_806_0 : mxxMat3) (w_807_0 : mxxMat2) (w_808_0 : mxxMat4) (w_809_0 : mxxMat4) (w_810_0 : mxxMat0) (w_811_0 : mxxMat1) (w_812_0 : mxxMat1) (w_813_0 : mxxMat2) (w_814_0 : mxxMat4) : Prop :=
  MxxRuntime.concatRows w_801_concat_1 w_800_0 w_801_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_5 w_802_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_1 w_803_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_3 w_804_0 ∧
  MxxRuntime.concatRows w_802_0 w_803_0 w_805_concat_1 ∧
  MxxRuntime.concatRows w_805_concat_1 w_804_0 w_805_0 ∧
  MxxRuntime.concatColumns w_801_0 w_805_0 w_806_0 ∧
  MxxRuntime.concatRows w_787_7 w_787_9 w_807_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_807_0 w_808_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_808_0 w_809_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_810_0 1 2 0 25 w_811_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_810_0 2 3 0 25 w_812_0 ∧
  MxxRuntime.concatRows w_811_0 w_812_0 w_813_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_813_0 w_814_0

abbrev generatedRoot.constraints_30 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_777_0 : Fin 1 → mxxMat1) (w_779_0 : Fin 1 → mxxMat1) (w_781_0 : Fin 1 → mxxMat1) (w_783_0 : Fin 1 → mxxMat1) (w_785_0 : Fin 1 → mxxMat1) (w_786_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_229_0 : mxxMat1) (w_769_0 : mxxMat1) (w_771_0 : mxxMat1) (w_772_0 : mxxMat14) (w_773_0 : mxxMat14) (w_776_0 : mxxMat1) (w_778_0 : mxxMat1) (w_780_0 : mxxMat1) (w_782_0 : mxxMat1) (w_784_0 : mxxMat1) (w_787_0 : mxxMat1) (w_787_1 : mxxMat1) (w_787_2 : mxxMat1) (w_787_3 : mxxMat1) (w_787_4 : mxxMat1) (w_787_5 : mxxMat1) (w_787_6 : mxxMat1) (w_787_7 : mxxMat1) (w_787_8 : mxxMat1) (w_787_9 : mxxMat1) (w_788_0 : mxxMat1) (w_789_0 : Int) (w_790_0 : mxxMat1) (w_791_0 : mxxMat1) (w_792_0 : mxxMat0) (w_792_concat_1 : mxxMat2) (w_793_0 : mxxMat1) (w_794_0 : mxxMat1) (w_795_0 : mxxMat1) (w_796_0 : mxxMat0) (w_796_concat_1 : mxxMat2) (w_797_0 : mxxMat3) (w_798_0 : mxxMat1) (w_799_0 : mxxMat1) (w_800_0 : mxxMat1) (w_801_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.centeredRebaseRuns w_229_0 w_769_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_771_0 w_772_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_772_0 w_773_0 ∧
  MxxRuntime.familyGetStatic w_777_0 (0) w_778_0 ∧
  MxxRuntime.familyGetStatic w_779_0 (0) w_780_0 ∧
  MxxRuntime.familyGetStatic w_781_0 (0) w_782_0 ∧
  MxxRuntime.familyGetStatic w_783_0 (0) w_784_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_787 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_785_0, (w_5_0, (w_786_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_777_0, (w_779_0, (w_781_0, (w_783_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_776_0, (w_776_0, (w_776_0, (w_776_0, (w_776_0, (w_776_0, (w_778_0, (w_780_0, (w_782_0, (w_784_0, ())))))))))) (w_787_0, (w_787_1, (w_787_2, (w_787_3, (w_787_4, (w_787_5, (w_787_6, (w_787_7, (w_787_8, (w_787_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_785_0 (0) w_788_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_789_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_790_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_789_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_791_0 ∧
  MxxRuntime.concatRows w_788_0 w_790_0 w_792_concat_1 ∧
  MxxRuntime.concatRows w_792_concat_1 w_791_0 w_792_0 ∧
  MxxRuntime.familyGetStatic w_786_0 (0) w_793_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_789_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_794_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_789_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_795_0 ∧
  MxxRuntime.concatRows w_793_0 w_794_0 w_796_concat_1 ∧
  MxxRuntime.concatRows w_796_concat_1 w_795_0 w_796_0 ∧
  MxxRuntime.concatColumns w_792_0 w_796_0 w_797_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_4 w_798_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_0 w_799_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(7)])) w_54_2 w_800_0 ∧
  MxxRuntime.concatRows w_798_0 w_799_0 w_801_concat_1

abbrev generatedRoot.constraints_29 (backend : MxxRuntime.BackendContext) (w_55_0 : mxxMat1) (w_731_7 : mxxMat1) (w_731_9 : mxxMat1) (w_745_0 : mxxMat0) (w_748_0 : mxxMat1) (w_749_0 : mxxMat0) (w_749_concat_1 : mxxMat2) (w_750_0 : mxxMat3) (w_751_0 : mxxMat2) (w_752_0 : mxxMat4) (w_753_0 : mxxMat4) (w_754_0 : mxxMat0) (w_755_0 : mxxMat1) (w_756_0 : mxxMat1) (w_757_0 : mxxMat2) (w_758_0 : mxxMat4) (w_759_0 : mxxMat4) (w_760_0 : mxxMat0) (w_761_0 : mxxMat1) (w_763_0 : mxxMat1) (w_765_0 : mxxMat1) (w_766_0 : mxxMat14) (w_767_0 : mxxMat14) : Prop :=
  MxxRuntime.concatRows w_749_concat_1 w_748_0 w_749_0 ∧
  MxxRuntime.concatColumns w_745_0 w_749_0 w_750_0 ∧
  MxxRuntime.concatRows w_731_7 w_731_9 w_751_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_751_0 w_752_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_752_0 w_753_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_754_0 1 2 0 25 w_755_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_754_0 2 3 0 25 w_756_0 ∧
  MxxRuntime.concatRows w_755_0 w_756_0 w_757_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_757_0 w_758_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_758_0 w_759_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_760_0 1 2 0 25 w_761_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_763_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_765_0 w_766_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_766_0 w_767_0

abbrev generatedRoot.constraints_28 (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_730_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_732_0 : mxxMat1) (w_733_0 : Int) (w_734_0 : mxxMat1) (w_735_0 : mxxMat1) (w_736_0 : mxxMat0) (w_736_concat_1 : mxxMat2) (w_737_0 : mxxMat1) (w_738_0 : mxxMat1) (w_739_0 : mxxMat1) (w_740_0 : mxxMat0) (w_740_concat_1 : mxxMat2) (w_741_0 : mxxMat3) (w_742_0 : mxxMat1) (w_743_0 : mxxMat1) (w_744_0 : mxxMat1) (w_745_0 : mxxMat0) (w_745_concat_1 : mxxMat2) (w_746_0 : mxxMat1) (w_747_0 : mxxMat1) (w_748_0 : mxxMat1) (w_749_concat_1 : mxxMat2) : Prop :=
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_733_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_734_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_733_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_735_0 ∧
  MxxRuntime.concatRows w_732_0 w_734_0 w_736_concat_1 ∧
  MxxRuntime.concatRows w_736_concat_1 w_735_0 w_736_0 ∧
  MxxRuntime.familyGetStatic w_730_0 (0) w_737_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_733_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_738_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_733_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_739_0 ∧
  MxxRuntime.concatRows w_737_0 w_738_0 w_740_concat_1 ∧
  MxxRuntime.concatRows w_740_concat_1 w_739_0 w_740_0 ∧
  MxxRuntime.concatColumns w_736_0 w_740_0 w_741_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_4 w_742_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_0 w_743_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_2 w_744_0 ∧
  MxxRuntime.concatRows w_742_0 w_743_0 w_745_concat_1 ∧
  MxxRuntime.concatRows w_745_concat_1 w_744_0 w_745_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_5 w_746_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_1 w_747_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(5)])) w_54_3 w_748_0 ∧
  MxxRuntime.concatRows w_746_0 w_747_0 w_749_concat_1

abbrev generatedRoot.constraints_27 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_721_0 : Fin 1 → mxxMat1) (w_723_0 : Fin 1 → mxxMat1) (w_725_0 : Fin 1 → mxxMat1) (w_727_0 : Fin 1 → mxxMat1) (w_729_0 : Fin 1 → mxxMat1) (w_730_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_55_0 : mxxMat1) (w_229_0 : mxxMat1) (w_698_0 : mxxMat0) (w_699_0 : mxxMat1) (w_700_0 : mxxMat1) (w_701_0 : mxxMat2) (w_702_0 : mxxMat4) (w_703_0 : mxxMat4) (w_704_0 : mxxMat0) (w_705_0 : mxxMat1) (w_707_0 : mxxMat1) (w_709_0 : mxxMat1) (w_710_0 : mxxMat14) (w_711_0 : mxxMat14) (w_713_0 : mxxMat1) (w_715_0 : mxxMat1) (w_716_0 : mxxMat14) (w_717_0 : mxxMat14) (w_720_0 : mxxMat1) (w_722_0 : mxxMat1) (w_724_0 : mxxMat1) (w_726_0 : mxxMat1) (w_728_0 : mxxMat1) (w_731_0 : mxxMat1) (w_731_1 : mxxMat1) (w_731_2 : mxxMat1) (w_731_3 : mxxMat1) (w_731_4 : mxxMat1) (w_731_5 : mxxMat1) (w_731_6 : mxxMat1) (w_731_7 : mxxMat1) (w_731_8 : mxxMat1) (w_731_9 : mxxMat1) (w_732_0 : mxxMat1) : Prop :=
  MxxRuntime.sliceMatrix w_698_0 1 2 0 25 w_699_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_698_0 2 3 0 25 w_700_0 ∧
  MxxRuntime.concatRows w_699_0 w_700_0 w_701_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_701_0 w_702_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_702_0 w_703_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_704_0 1 2 0 25 w_705_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_707_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_709_0 w_710_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_710_0 w_711_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_713_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_715_0 w_716_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_716_0 w_717_0 ∧
  MxxRuntime.familyGetStatic w_721_0 (0) w_722_0 ∧
  MxxRuntime.familyGetStatic w_723_0 (0) w_724_0 ∧
  MxxRuntime.familyGetStatic w_725_0 (0) w_726_0 ∧
  MxxRuntime.familyGetStatic w_727_0 (0) w_728_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_731 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_729_0, (w_5_0, (w_730_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_721_0, (w_723_0, (w_725_0, (w_727_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_720_0, (w_720_0, (w_720_0, (w_720_0, (w_720_0, (w_720_0, (w_722_0, (w_724_0, (w_726_0, (w_728_0, ())))))))))) (w_731_0, (w_731_1, (w_731_2, (w_731_3, (w_731_4, (w_731_5, (w_731_6, (w_731_7, (w_731_8, (w_731_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_729_0 (0) w_732_0 ∧
  0 ≤ (0)

abbrev generatedRoot.constraints_26 (backend : MxxRuntime.BackendContext) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_675_7 : mxxMat1) (w_675_9 : mxxMat1) (w_686_0 : mxxMat1) (w_687_0 : mxxMat1) (w_688_0 : mxxMat1) (w_689_0 : mxxMat0) (w_689_concat_1 : mxxMat2) (w_690_0 : mxxMat1) (w_691_0 : mxxMat1) (w_692_0 : mxxMat1) (w_693_0 : mxxMat0) (w_693_concat_1 : mxxMat2) (w_694_0 : mxxMat3) (w_695_0 : mxxMat2) (w_696_0 : mxxMat4) (w_697_0 : mxxMat4) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_4 w_686_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_0 w_687_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_2 w_688_0 ∧
  MxxRuntime.concatRows w_686_0 w_687_0 w_689_concat_1 ∧
  MxxRuntime.concatRows w_689_concat_1 w_688_0 w_689_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_5 w_690_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_1 w_691_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (0) [(3)])) w_54_3 w_692_0 ∧
  MxxRuntime.concatRows w_690_0 w_691_0 w_693_concat_1 ∧
  MxxRuntime.concatRows w_693_concat_1 w_692_0 w_693_0 ∧
  MxxRuntime.concatColumns w_689_0 w_693_0 w_694_0 ∧
  MxxRuntime.concatRows w_675_7 w_675_9 w_695_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_695_0 w_696_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_696_0 w_697_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev generatedRoot.constraints_25 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_665_0 : Fin 1 → mxxMat1) (w_667_0 : Fin 1 → mxxMat1) (w_669_0 : Fin 1 → mxxMat1) (w_671_0 : Fin 1 → mxxMat1) (w_673_0 : Fin 1 → mxxMat1) (w_674_0 : Fin 1 → mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) (w_55_0 : mxxMat1) (w_229_0 : mxxMat1) (w_243_0 : mxxMat1) (w_401_0 : mxxMat1) (w_409_0 : mxxMat1) (w_640_0 : mxxMat0) (w_641_0 : mxxMat1) (w_643_0 : mxxMat14) (w_646_0 : mxxMat14) (w_648_0 : mxxMat14) (w_651_0 : mxxMat1) (w_653_0 : mxxMat1) (w_654_0 : mxxMat14) (w_655_0 : mxxMat14) (w_657_0 : mxxMat1) (w_659_0 : mxxMat1) (w_660_0 : mxxMat14) (w_661_0 : mxxMat14) (w_664_0 : mxxMat1) (w_666_0 : mxxMat1) (w_668_0 : mxxMat1) (w_670_0 : mxxMat1) (w_672_0 : mxxMat1) (w_675_0 : mxxMat1) (w_675_1 : mxxMat1) (w_675_2 : mxxMat1) (w_675_3 : mxxMat1) (w_675_4 : mxxMat1) (w_675_5 : mxxMat1) (w_675_6 : mxxMat1) (w_675_7 : mxxMat1) (w_675_8 : mxxMat1) (w_675_9 : mxxMat1) (w_676_0 : mxxMat1) (w_677_0 : Int) (w_678_0 : mxxMat1) (w_679_0 : mxxMat1) (w_680_0 : mxxMat0) (w_680_concat_1 : mxxMat2) (w_681_0 : mxxMat1) (w_682_0 : mxxMat1) (w_683_0 : mxxMat1) (w_684_0 : mxxMat0) (w_684_concat_1 : mxxMat2) (w_685_0 : mxxMat3) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_640_0 1 2 0 25 w_641_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_401_0 w_643_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_243_0 w_646_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_409_0 w_648_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_651_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_653_0 w_654_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_654_0 w_655_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_657_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_659_0 w_660_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_660_0 w_661_0 ∧
  MxxRuntime.familyGetStatic w_665_0 (0) w_666_0 ∧
  MxxRuntime.familyGetStatic w_667_0 (0) w_668_0 ∧
  MxxRuntime.familyGetStatic w_669_0 (0) w_670_0 ∧
  MxxRuntime.familyGetStatic w_671_0 (0) w_672_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_675 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_673_0, (w_5_0, (w_674_0, (w_54_4, (w_54_0, (w_54_2, (w_54_5, (w_54_1, (w_54_3, (w_665_0, (w_667_0, (w_669_0, (w_671_0, ()))))))))))))))))))))))) next) (Int.toNat (0)) (w_664_0, (w_664_0, (w_664_0, (w_664_0, (w_664_0, (w_664_0, (w_666_0, (w_668_0, (w_670_0, (w_672_0, ())))))))))) (w_675_0, (w_675_1, (w_675_2, (w_675_3, (w_675_4, (w_675_5, (w_675_6, (w_675_7, (w_675_8, (w_675_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_673_0 (0) w_676_0 ∧
  0 ≤ (0) ∧
  (0) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_677_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_678_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_677_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_679_0 ∧
  MxxRuntime.concatRows w_676_0 w_678_0 w_680_concat_1 ∧
  MxxRuntime.concatRows w_680_concat_1 w_679_0 w_680_0 ∧
  MxxRuntime.familyGetStatic w_674_0 (0) w_681_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_677_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_682_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_677_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_683_0 ∧
  MxxRuntime.concatRows w_681_0 w_682_0 w_684_concat_1 ∧
  MxxRuntime.concatRows w_684_concat_1 w_683_0 w_684_0 ∧
  MxxRuntime.concatColumns w_680_0 w_684_0 w_685_0

abbrev generatedRoot.constraints_24 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_371_0 : Fin 4 → mxxMat1) (w_55_0 : mxxMat1) (w_242_0 : mxxMat1) (w_611_0 : mxxMat1) (w_612_0 : mxxMat1) (w_613_0 : mxxMat1) (w_614_0 : mxxMat0) (w_614_concat_1 : mxxMat2) (w_615_0 : mxxMat1) (w_616_0 : mxxMat1) (w_617_0 : mxxMat1) (w_618_0 : mxxMat0) (w_618_concat_1 : mxxMat2) (w_619_0 : mxxMat3) (w_620_0 : mxxMat1) (w_621_0 : mxxMat1) (w_622_0 : mxxMat2) (w_623_0 : mxxMat4) (w_624_0 : mxxMat4) (w_625_0 : mxxMat0) (w_626_0 : mxxMat1) (w_627_0 : mxxMat1) (w_629_0 : mxxMat1) (w_630_0 : mxxMat14) (w_631_0 : mxxMat14) (w_633_0 : mxxMat14) (w_635_0 : mxxMat1) (w_636_0 : mxxMat1) (w_637_0 : mxxMat2) (w_638_0 : mxxMat4) (w_639_0 : mxxMat4) : Prop :=
  MxxRuntime.concatRows w_611_0 w_612_0 w_614_concat_1 ∧
  MxxRuntime.concatRows w_614_concat_1 w_613_0 w_614_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (1) w_615_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_616_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_617_0 ∧
  MxxRuntime.concatRows w_615_0 w_616_0 w_618_concat_1 ∧
  MxxRuntime.concatRows w_618_concat_1 w_617_0 w_618_0 ∧
  MxxRuntime.concatColumns w_614_0 w_618_0 w_619_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_620_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_621_0 ∧
  MxxRuntime.concatRows w_620_0 w_621_0 w_622_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_622_0 w_623_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_623_0 w_624_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_625_0 1 2 0 25 w_626_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_627_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_629_0 w_630_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_630_0 w_631_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_242_0 w_633_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_635_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_636_0 ∧
  MxxRuntime.concatRows w_635_0 w_636_0 w_637_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_637_0 w_638_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_638_0 w_639_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3

abbrev generatedRoot.constraints_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_366_0 : Fin 4 → mxxMat1) (w_55_0 : mxxMat1) (w_242_0 : mxxMat1) (w_243_0 : mxxMat1) (w_401_0 : mxxMat1) (w_409_0 : mxxMat1) (w_580_0 : mxxMat1) (w_581_0 : mxxMat1) (w_582_0 : mxxMat2) (w_583_0 : mxxMat4) (w_584_0 : mxxMat4) (w_585_0 : mxxMat0) (w_586_0 : mxxMat1) (w_587_0 : mxxMat1) (w_589_0 : mxxMat1) (w_590_0 : mxxMat14) (w_591_0 : mxxMat14) (w_593_0 : mxxMat14) (w_595_0 : mxxMat1) (w_596_0 : mxxMat1) (w_597_0 : mxxMat2) (w_598_0 : mxxMat4) (w_599_0 : mxxMat4) (w_600_0 : mxxMat0) (w_601_0 : mxxMat1) (w_603_0 : mxxMat14) (w_606_0 : mxxMat14) (w_608_0 : mxxMat14) (w_611_0 : mxxMat1) (w_612_0 : mxxMat1) (w_613_0 : mxxMat1) : Prop :=
  MxxRuntime.concatRows w_580_0 w_581_0 w_582_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_582_0 w_583_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_583_0 w_584_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_585_0 1 2 0 25 w_586_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_587_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_589_0 w_590_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_590_0 w_591_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_242_0 w_593_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_595_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_596_0 ∧
  MxxRuntime.concatRows w_595_0 w_596_0 w_597_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_597_0 w_598_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_598_0 w_599_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_600_0 1 2 0 25 w_601_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_401_0 w_603_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_243_0 w_606_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_409_0 w_608_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (1) w_611_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_612_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_613_0

abbrev generatedRoot.constraints_22 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_55_0 : mxxMat1) (w_242_0 : mxxMat1) (w_243_0 : mxxMat1) (w_401_0 : mxxMat1) (w_409_0 : mxxMat1) (w_547_0 : mxxMat1) (w_549_0 : mxxMat1) (w_550_0 : mxxMat14) (w_551_0 : mxxMat14) (w_553_0 : mxxMat14) (w_555_0 : mxxMat1) (w_556_0 : mxxMat1) (w_557_0 : mxxMat2) (w_558_0 : mxxMat4) (w_559_0 : mxxMat4) (w_560_0 : mxxMat0) (w_561_0 : mxxMat1) (w_563_0 : mxxMat14) (w_566_0 : mxxMat14) (w_568_0 : mxxMat14) (w_571_0 : mxxMat1) (w_572_0 : mxxMat1) (w_573_0 : mxxMat1) (w_574_0 : mxxMat0) (w_574_concat_1 : mxxMat2) (w_575_0 : mxxMat1) (w_576_0 : mxxMat1) (w_577_0 : mxxMat1) (w_578_0 : mxxMat0) (w_578_concat_1 : mxxMat2) (w_579_0 : mxxMat3) (w_580_0 : mxxMat1) (w_581_0 : mxxMat1) : Prop :=
  MxxRuntime.centeredRebaseRuns w_55_0 w_547_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_549_0 w_550_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_550_0 w_551_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_242_0 w_553_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_555_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_556_0 ∧
  MxxRuntime.concatRows w_555_0 w_556_0 w_557_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_557_0 w_558_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_558_0 w_559_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_560_0 1 2 0 25 w_561_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_401_0 w_563_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_243_0 w_566_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_409_0 w_568_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (1) w_571_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_572_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_573_0 ∧
  MxxRuntime.concatRows w_571_0 w_572_0 w_574_concat_1 ∧
  MxxRuntime.concatRows w_574_concat_1 w_573_0 w_574_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (1) w_575_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_576_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_577_0 ∧
  MxxRuntime.concatRows w_575_0 w_576_0 w_578_concat_1 ∧
  MxxRuntime.concatRows w_578_concat_1 w_577_0 w_578_0 ∧
  MxxRuntime.concatColumns w_574_0 w_578_0 w_579_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_580_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_581_0

abbrev generatedRoot.constraints_21 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_371_0 : Fin 4 → mxxMat1) (w_525_0 : mxxMat0) (w_531_0 : mxxMat0) (w_532_0 : mxxMat1) (w_533_0 : mxxMat1) (w_535_0 : mxxMat1) (w_536_0 : mxxMat1) (w_537_0 : mxxMat1) (w_538_0 : mxxMat0) (w_538_concat_1 : mxxMat2) (w_539_0 : mxxMat3) (w_540_0 : mxxMat1) (w_541_0 : mxxMat1) (w_542_0 : mxxMat2) (w_543_0 : mxxMat4) (w_544_0 : mxxMat4) (w_545_0 : mxxMat0) (w_546_0 : mxxMat1) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_531_0 0 1 0 25 w_532_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (2) w_533_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_531_0 1 2 0 25 w_536_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_531_0 2 3 0 25 w_537_0 ∧
  MxxRuntime.concatRows w_535_0 w_536_0 w_538_concat_1 ∧
  MxxRuntime.concatRows w_538_concat_1 w_537_0 w_538_0 ∧
  MxxRuntime.concatColumns w_525_0 w_538_0 w_539_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_540_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_541_0 ∧
  MxxRuntime.concatRows w_540_0 w_541_0 w_542_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_542_0 w_543_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_543_0 w_544_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_545_0 1 2 0 25 w_546_0

abbrev generatedRoot.constraints_20 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_366_0 : Fin 4 → mxxMat1) (w_430_0 : Fin 4 → mxxMat5) (w_517_0 : mxxMat0) (w_518_0 : mxxMat1) (w_519_0 : mxxMat1) (w_520_0 : mxxMat5) (w_522_0 : mxxMat1) (w_523_0 : mxxMat1) (w_524_0 : mxxMat1) (w_525_0 : mxxMat0) (w_525_concat_1 : mxxMat2) (w_526_0 : mxxMat1) (w_527_0 : mxxMat1) (w_528_0 : mxxMat2) (w_529_0 : mxxMat4) (w_530_0 : mxxMat4) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_517_0 0 1 0 25 w_518_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (2) w_519_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (0) w_520_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_517_0 1 2 0 25 w_523_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_517_0 2 3 0 25 w_524_0 ∧
  MxxRuntime.concatRows w_522_0 w_523_0 w_525_concat_1 ∧
  MxxRuntime.concatRows w_525_concat_1 w_524_0 w_525_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_526_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_527_0 ∧
  MxxRuntime.concatRows w_526_0 w_527_0 w_528_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_528_0 w_529_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_529_0 w_530_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_19 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_243_0 : mxxMat1) (w_401_0 : mxxMat1) (w_409_0 : mxxMat1) (w_487_0 : mxxMat1) (w_488_0 : mxxMat1) (w_489_0 : mxxMat2) (w_490_0 : mxxMat4) (w_491_0 : mxxMat4) (w_492_0 : mxxMat0) (w_493_0 : mxxMat1) (w_495_0 : mxxMat14) (w_498_0 : mxxMat14) (w_500_0 : mxxMat14) (w_503_0 : mxxMat1) (w_504_0 : mxxMat1) (w_505_0 : mxxMat1) (w_506_0 : mxxMat0) (w_506_concat_1 : mxxMat2) (w_507_0 : mxxMat1) (w_508_0 : mxxMat1) (w_509_0 : mxxMat1) (w_510_0 : mxxMat0) (w_510_concat_1 : mxxMat2) (w_511_0 : mxxMat3) (w_512_0 : mxxMat1) (w_513_0 : mxxMat1) (w_514_0 : mxxMat2) (w_515_0 : mxxMat4) (w_516_0 : mxxMat4) : Prop :=
  MxxRuntime.concatRows w_487_0 w_488_0 w_489_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_489_0 w_490_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_490_0 w_491_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_492_0 1 2 0 25 w_493_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_401_0 w_495_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_243_0 w_498_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_409_0 w_500_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (0) w_503_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_504_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_505_0 ∧
  MxxRuntime.concatRows w_503_0 w_504_0 w_506_concat_1 ∧
  MxxRuntime.concatRows w_506_concat_1 w_505_0 w_506_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (0) w_507_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_508_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_509_0 ∧
  MxxRuntime.concatRows w_507_0 w_508_0 w_510_concat_1 ∧
  MxxRuntime.concatRows w_510_concat_1 w_509_0 w_510_0 ∧
  MxxRuntime.concatColumns w_506_0 w_510_0 w_511_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_512_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_513_0 ∧
  MxxRuntime.concatRows w_512_0 w_513_0 w_514_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_514_0 w_515_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_515_0 w_516_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3

abbrev generatedRoot.constraints_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_55_0 : mxxMat1) (w_242_0 : mxxMat1) (w_468_0 : mxxMat2) (w_469_0 : mxxMat4) (w_470_0 : mxxMat4) (w_471_0 : mxxMat0) (w_472_0 : mxxMat1) (w_473_0 : mxxMat1) (w_475_0 : mxxMat1) (w_476_0 : mxxMat14) (w_477_0 : mxxMat14) (w_479_0 : mxxMat14) (w_481_0 : mxxMat1) (w_482_0 : mxxMat1) (w_483_0 : mxxMat2) (w_484_0 : mxxMat4) (w_485_0 : mxxMat4) (w_486_0 : mxxMat0) (w_487_0 : mxxMat1) (w_488_0 : mxxMat1) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_468_0 w_469_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_469_0 w_470_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_471_0 1 2 0 25 w_472_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_473_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_475_0 w_476_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_476_0 w_477_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_242_0 w_479_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_481_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_482_0 ∧
  MxxRuntime.concatRows w_481_0 w_482_0 w_483_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_483_0 w_484_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_484_0 w_485_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_486_0 1 2 0 25 w_487_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_486_0 2 3 0 25 w_488_0

abbrev generatedRoot.constraints_17 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_436_0 : mxxMat0) (w_449_0 : mxxMat0) (w_450_0 : mxxMat3) (w_451_0 : mxxMat1) (w_452_0 : mxxMat1) (w_453_0 : mxxMat1) (w_454_0 : mxxMat0) (w_454_concat_1 : mxxMat2) (w_455_0 : mxxMat1) (w_456_0 : mxxMat1) (w_457_0 : mxxMat1) (w_458_0 : mxxMat0) (w_458_concat_1 : mxxMat2) (w_459_0 : mxxMat3) (w_460_0 : mxxMat1) (w_461_0 : mxxMat1) (w_462_0 : mxxMat2) (w_463_0 : mxxMat4) (w_464_0 : mxxMat4) (w_465_0 : mxxMat0) (w_466_0 : mxxMat1) (w_467_0 : mxxMat1) (w_468_0 : mxxMat2) : Prop :=
  MxxRuntime.concatColumns w_436_0 w_449_0 w_450_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (2) w_451_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_452_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_453_0 ∧
  MxxRuntime.concatRows w_451_0 w_452_0 w_454_concat_1 ∧
  MxxRuntime.concatRows w_454_concat_1 w_453_0 w_454_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (2) w_455_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_456_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_457_0 ∧
  MxxRuntime.concatRows w_455_0 w_456_0 w_458_concat_1 ∧
  MxxRuntime.concatRows w_458_concat_1 w_457_0 w_458_0 ∧
  MxxRuntime.concatColumns w_454_0 w_458_0 w_459_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_460_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_461_0 ∧
  MxxRuntime.concatRows w_460_0 w_461_0 w_462_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_462_0 w_463_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_463_0 w_464_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_465_0 1 2 0 25 w_466_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_465_0 2 3 0 25 w_467_0 ∧
  MxxRuntime.concatRows w_466_0 w_467_0 w_468_0

abbrev generatedRoot.constraints_16 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_371_0 : Fin 4 → mxxMat1) (w_427_0 : mxxMat0) (w_433_0 : mxxMat1) (w_434_0 : mxxMat1) (w_435_0 : mxxMat1) (w_436_0 : mxxMat0) (w_436_concat_1 : mxxMat2) (w_437_0 : mxxMat1) (w_438_0 : mxxMat1) (w_439_0 : mxxMat2) (w_440_0 : mxxMat4) (w_441_0 : mxxMat4) (w_442_0 : mxxMat0) (w_443_0 : mxxMat1) (w_444_0 : mxxMat1) (w_446_0 : mxxMat1) (w_447_0 : mxxMat1) (w_448_0 : mxxMat1) (w_449_0 : mxxMat0) (w_449_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.sliceMatrix w_427_0 2 3 0 25 w_435_0 ∧
  MxxRuntime.concatRows w_433_0 w_434_0 w_436_concat_1 ∧
  MxxRuntime.concatRows w_436_concat_1 w_435_0 w_436_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_437_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_438_0 ∧
  MxxRuntime.concatRows w_437_0 w_438_0 w_439_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_439_0 w_440_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_440_0 w_441_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_442_0 0 1 0 25 w_443_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (1) w_444_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_442_0 1 2 0 25 w_447_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_442_0 2 3 0 25 w_448_0 ∧
  MxxRuntime.concatRows w_446_0 w_447_0 w_449_concat_1 ∧
  MxxRuntime.concatRows w_449_concat_1 w_448_0 w_449_0

abbrev generatedRoot.constraints_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_366_0 : Fin 4 → mxxMat1) (w_430_0 : Fin 4 → mxxMat5) (w_416_0 : mxxMat0) (w_417_0 : mxxMat1) (w_418_0 : mxxMat1) (w_419_0 : mxxMat1) (w_420_0 : mxxMat0) (w_420_concat_1 : mxxMat2) (w_421_0 : mxxMat3) (w_422_0 : mxxMat1) (w_423_0 : mxxMat1) (w_424_0 : mxxMat2) (w_425_0 : mxxMat4) (w_426_0 : mxxMat4) (w_427_0 : mxxMat0) (w_428_0 : mxxMat1) (w_429_0 : mxxMat1) (w_431_0 : mxxMat5) (w_434_0 : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_418_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_419_0 ∧
  MxxRuntime.concatRows w_417_0 w_418_0 w_420_concat_1 ∧
  MxxRuntime.concatRows w_420_concat_1 w_419_0 w_420_0 ∧
  MxxRuntime.concatColumns w_416_0 w_420_0 w_421_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_422_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_423_0 ∧
  MxxRuntime.concatRows w_422_0 w_423_0 w_424_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_424_0 w_425_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_425_0 w_426_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_427_0 0 1 0 25 w_428_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (1) w_429_0 ∧
  MxxRuntime.familyGetStatic w_430_0 (0) w_431_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_427_0 1 2 0 25 w_434_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25

abbrev generatedRoot.constraints_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_55_0 : mxxMat1) (w_242_0 : mxxMat1) (w_243_0 : mxxMat1) (w_382_0 : mxxMat0) (w_383_0 : mxxMat1) (w_384_0 : mxxMat1) (w_386_0 : mxxMat1) (w_387_0 : mxxMat14) (w_388_0 : mxxMat14) (w_390_0 : mxxMat14) (w_392_0 : mxxMat1) (w_393_0 : mxxMat1) (w_394_0 : mxxMat2) (w_395_0 : mxxMat4) (w_396_0 : mxxMat4) (w_397_0 : mxxMat0) (w_398_0 : mxxMat1) (w_401_0 : mxxMat1) (w_402_0 : mxxMat14) (w_405_0 : mxxMat14) (w_407_0 : mxxMat1) (w_409_0 : mxxMat1) (w_410_0 : mxxMat14) (w_413_0 : mxxMat1) (w_414_0 : mxxMat1) (w_415_0 : mxxMat1) (w_416_0 : mxxMat0) (w_416_concat_1 : mxxMat2) (w_417_0 : mxxMat1) : Prop :=
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_382_0 1 2 0 25 w_383_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_384_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_386_0 w_387_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_387_0 w_388_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_242_0 w_390_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_392_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_393_0 ∧
  MxxRuntime.concatRows w_392_0 w_393_0 w_394_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_394_0 w_395_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_395_0 w_396_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_397_0 1 2 0 25 w_398_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_401_0 w_402_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_243_0 w_405_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_407_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_409_0 w_410_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (0) w_413_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_414_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_415_0 ∧
  MxxRuntime.concatRows w_413_0 w_414_0 w_416_concat_1 ∧
  MxxRuntime.concatRows w_416_concat_1 w_415_0 w_416_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (0) w_417_0

abbrev generatedRoot.constraints_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_366_0 : Fin 4 → mxxMat1) (w_371_0 : Fin 4 → mxxMat1) (w_356_0 : mxxMat0) (w_357_0 : mxxMat1) (w_358_0 : mxxMat1) (w_359_0 : mxxMat2) (w_360_0 : mxxMat4) (w_361_0 : mxxMat4) (w_362_0 : mxxMat0) (w_363_0 : mxxMat1) (w_367_0 : mxxMat1) (w_368_0 : mxxMat1) (w_369_0 : mxxMat1) (w_370_0 : mxxMat0) (w_370_concat_1 : mxxMat2) (w_372_0 : mxxMat1) (w_373_0 : mxxMat1) (w_374_0 : mxxMat1) (w_375_0 : mxxMat0) (w_375_concat_1 : mxxMat2) (w_376_0 : mxxMat3) (w_377_0 : mxxMat1) (w_378_0 : mxxMat1) (w_379_0 : mxxMat2) (w_380_0 : mxxMat4) (w_381_0 : mxxMat4) : Prop :=
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_356_0 2 3 0 25 w_358_0 ∧
  MxxRuntime.concatRows w_357_0 w_358_0 w_359_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_359_0 w_360_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_360_0 w_361_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_362_0 1 2 0 25 w_363_0 ∧
  MxxRuntime.familyGetStatic w_366_0 (1) w_367_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_368_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_369_0 ∧
  MxxRuntime.concatRows w_367_0 w_368_0 w_370_concat_1 ∧
  MxxRuntime.concatRows w_370_concat_1 w_369_0 w_370_0 ∧
  MxxRuntime.familyGetStatic w_371_0 (1) w_372_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_373_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_374_0 ∧
  MxxRuntime.concatRows w_372_0 w_373_0 w_375_concat_1 ∧
  MxxRuntime.concatRows w_375_concat_1 w_374_0 w_375_0 ∧
  MxxRuntime.concatColumns w_370_0 w_375_0 w_376_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_377_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_378_0 ∧
  MxxRuntime.concatRows w_377_0 w_378_0 w_379_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_379_0 w_380_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_380_0 w_381_0 ∧
  0 ≤ 1

abbrev generatedRoot.constraints_12 (backend : MxxRuntime.BackendContext) (w_229_0 : mxxMat1) (w_315_7 : mxxMat1) (w_315_9 : mxxMat1) (w_338_0 : mxxMat0) (w_339_0 : mxxMat1) (w_340_0 : mxxMat1) (w_341_0 : mxxMat2) (w_342_0 : mxxMat4) (w_343_0 : mxxMat4) (w_344_0 : mxxMat0) (w_345_0 : mxxMat1) (w_347_0 : mxxMat1) (w_349_0 : mxxMat1) (w_350_0 : mxxMat14) (w_351_0 : mxxMat14) (w_353_0 : mxxMat2) (w_354_0 : mxxMat4) (w_355_0 : mxxMat4) (w_356_0 : mxxMat0) (w_357_0 : mxxMat1) : Prop :=
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_338_0 2 3 0 25 w_340_0 ∧
  MxxRuntime.concatRows w_339_0 w_340_0 w_341_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_341_0 w_342_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_342_0 w_343_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_344_0 1 2 0 25 w_345_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_347_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_349_0 w_350_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_350_0 w_351_0 ∧
  MxxRuntime.concatRows w_315_7 w_315_9 w_353_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_353_0 w_354_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_354_0 w_355_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_356_0 1 2 0 25 w_357_0 ∧
  0 ≤ 2 ∧
  2 < 3

abbrev generatedRoot.constraints_11 (backend : MxxRuntime.BackendContext) (w_302_0 : mxxMat1) (w_303_0 : mxxMat1) (w_304_0 : mxxMat1) (w_312_0 : mxxMat1) (w_313_0 : mxxMat1) (w_314_0 : mxxMat1) (w_315_6 : mxxMat1) (w_315_8 : mxxMat1) (w_326_0 : mxxMat1) (w_327_0 : mxxMat1) (w_328_0 : mxxMat1) (w_329_0 : mxxMat0) (w_329_concat_1 : mxxMat2) (w_330_0 : mxxMat1) (w_331_0 : mxxMat1) (w_332_0 : mxxMat1) (w_333_0 : mxxMat0) (w_333_concat_1 : mxxMat2) (w_334_0 : mxxMat3) (w_335_0 : mxxMat2) (w_336_0 : mxxMat4) (w_337_0 : mxxMat4) (w_338_0 : mxxMat0) (w_339_0 : mxxMat1) : Prop :=
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_302_0 w_326_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_303_0 w_327_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_304_0 w_328_0 ∧
  MxxRuntime.concatRows w_326_0 w_327_0 w_329_concat_1 ∧
  MxxRuntime.concatRows w_329_concat_1 w_328_0 w_329_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_312_0 w_330_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_313_0 w_331_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (3) [(1), (3), (5), (15)])) w_314_0 w_332_0 ∧
  MxxRuntime.concatRows w_330_0 w_331_0 w_333_concat_1 ∧
  MxxRuntime.concatRows w_333_concat_1 w_332_0 w_333_0 ∧
  MxxRuntime.concatColumns w_329_0 w_333_0 w_334_0 ∧
  MxxRuntime.concatRows w_315_6 w_315_8 w_335_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_335_0 w_336_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_336_0 w_337_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_338_0 1 2 0 25 w_339_0

abbrev generatedRoot.constraints_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_5_0 : ByteArray) (w_245_0 : Fin 4 → mxxMat1) (w_247_0 : Fin 4 → mxxMat1) (w_249_0 : Fin 4 → mxxMat1) (w_251_0 : Fin 4 → mxxMat1) (w_253_0 : Fin 4 → mxxMat1) (w_254_0 : Fin 4 → mxxMat1) (w_244_0 : mxxMat1) (w_246_0 : mxxMat1) (w_248_0 : mxxMat1) (w_250_0 : mxxMat1) (w_252_0 : mxxMat1) (w_302_0 : mxxMat1) (w_303_0 : mxxMat1) (w_304_0 : mxxMat1) (w_311_0 : mxxMat0) (w_312_0 : mxxMat1) (w_313_0 : mxxMat1) (w_314_0 : mxxMat1) (w_315_0 : mxxMat1) (w_315_1 : mxxMat1) (w_315_2 : mxxMat1) (w_315_3 : mxxMat1) (w_315_4 : mxxMat1) (w_315_5 : mxxMat1) (w_315_6 : mxxMat1) (w_315_7 : mxxMat1) (w_315_8 : mxxMat1) (w_315_9 : mxxMat1) (w_316_0 : mxxMat1) (w_317_0 : Int) (w_318_0 : mxxMat1) (w_319_0 : mxxMat1) (w_320_0 : mxxMat0) (w_320_concat_1 : mxxMat2) (w_321_0 : mxxMat1) (w_322_0 : mxxMat1) (w_323_0 : mxxMat1) (w_324_0 : mxxMat0) (w_324_concat_1 : mxxMat2) (w_325_0 : mxxMat3) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_311_0 0 1 0 25 w_312_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_311_0 1 2 0 25 w_313_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_311_0 2 3 0 25 w_314_0 ∧
  0 ≤ (3) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_315 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_253_0, (w_5_0, (w_254_0, (w_302_0, (w_303_0, (w_304_0, (w_312_0, (w_313_0, (w_314_0, (w_245_0, (w_247_0, (w_249_0, (w_251_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_244_0, (w_244_0, (w_244_0, (w_244_0, (w_244_0, (w_244_0, (w_246_0, (w_248_0, (w_250_0, (w_252_0, ())))))))))) (w_315_0, (w_315_1, (w_315_2, (w_315_3, (w_315_4, (w_315_5, (w_315_6, (w_315_7, (w_315_8, (w_315_9, ())))))))))) ∧
  MxxRuntime.familyGetStatic w_253_0 (3) w_316_0 ∧
  0 ≤ (3) ∧
  (3) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_317_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_318_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_317_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_319_0 ∧
  MxxRuntime.concatRows w_316_0 w_318_0 w_320_concat_1 ∧
  MxxRuntime.concatRows w_320_concat_1 w_319_0 w_320_0 ∧
  MxxRuntime.familyGetStatic w_254_0 (3) w_321_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_317_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_322_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_317_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_323_0 ∧
  MxxRuntime.concatRows w_321_0 w_322_0 w_324_concat_1 ∧
  MxxRuntime.concatRows w_324_concat_1 w_323_0 w_324_0 ∧
  MxxRuntime.concatColumns w_320_0 w_324_0 w_325_0 ∧
  0 ≤ (3)

abbrev generatedRoot.constraints_9 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_24_0 : ByteArray) (w_306_0 : Fin 1 → mxxMat1) (w_299_0 : mxxMat4) (w_300_0 : mxxMat4) (w_301_0 : mxxMat0) (w_302_0 : mxxMat1) (w_303_0 : mxxMat1) (w_304_0 : mxxMat1) (w_305_0 : mxxMat1) (w_307_0 : mxxMat1) (w_308_0 : mxxMat2) (w_309_0 : mxxMat4) (w_310_0 : mxxMat4) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 2048 w_299_0 w_300_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_301_0 0 1 0 25 w_302_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_301_0 1 2 0 25 w_303_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_301_0 2 3 0 25 w_304_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_24_0) w_305_0 ∧
  MxxRuntime.familyGetStatic w_306_0 (0) w_307_0 ∧
  MxxRuntime.concatRows w_305_0 w_307_0 w_308_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_308_0 w_309_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_309_0 w_310_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_24_0 : ByteArray) (w_296_0 : Fin 1 → mxxMat1) (w_280_0 : mxxMat0) (w_281_0 : mxxMat1) (w_282_0 : mxxMat1) (w_283_0 : mxxMat2) (w_284_0 : mxxMat4) (w_285_0 : mxxMat4) (w_286_0 : mxxMat0) (w_287_0 : mxxMat1) (w_290_0 : mxxMat1) (w_291_0 : mxxMat1) (w_292_0 : mxxMat1) (w_293_0 : mxxMat0) (w_293_concat_1 : mxxMat2) (w_294_0 : mxxMat3) (w_295_0 : mxxMat1) (w_297_0 : mxxMat1) (w_298_0 : mxxMat2) (w_299_0 : mxxMat4) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_282_0 ∧
  MxxRuntime.concatRows w_281_0 w_282_0 w_283_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_283_0 w_284_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_284_0 w_285_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_286_0 0 1 0 25 w_287_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_286_0 1 2 0 25 w_291_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_286_0 2 3 0 25 w_292_0 ∧
  MxxRuntime.concatRows w_290_0 w_291_0 w_293_concat_1 ∧
  MxxRuntime.concatRows w_293_concat_1 w_292_0 w_293_0 ∧
  MxxRuntime.concatColumns w_280_0 w_293_0 w_294_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat1), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat2), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_24_0) w_295_0 ∧
  MxxRuntime.familyGetStatic w_296_0 (0) w_297_0 ∧
  MxxRuntime.concatRows w_295_0 w_297_0 w_298_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_298_0 w_299_0

abbrev generatedRoot.constraints_7 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_274_0 : Fin 2 → mxxMat5) (w_259_0 : mxxMat0) (w_263_0 : mxxMat1) (w_264_0 : mxxMat0) (w_264_concat_1 : mxxMat2) (w_265_0 : mxxMat3) (w_266_0 : mxxMat1) (w_267_0 : mxxMat1) (w_268_0 : mxxMat2) (w_269_0 : mxxMat4) (w_270_0 : mxxMat4) (w_271_0 : mxxMat0) (w_272_0 : mxxMat1) (w_275_0 : mxxMat5) (w_277_0 : mxxMat1) (w_278_0 : mxxMat1) (w_279_0 : mxxMat1) (w_280_0 : mxxMat0) (w_280_concat_1 : mxxMat2) (w_281_0 : mxxMat1) : Prop :=
  MxxRuntime.concatRows w_264_concat_1 w_263_0 w_264_0 ∧
  MxxRuntime.concatColumns w_259_0 w_264_0 w_265_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_266_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_267_0 ∧
  MxxRuntime.concatRows w_266_0 w_267_0 w_268_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_268_0 w_269_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_269_0 w_270_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_271_0 0 1 0 25 w_272_0 ∧
  MxxRuntime.familyGetStatic w_274_0 (1) w_275_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_271_0 1 2 0 25 w_278_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_271_0 2 3 0 25 w_279_0 ∧
  MxxRuntime.concatRows w_277_0 w_278_0 w_280_concat_1 ∧
  MxxRuntime.concatRows w_280_concat_1 w_279_0 w_280_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_281_0

abbrev generatedRoot.constraints_6 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_245_0 : Fin 4 → mxxMat1) (w_247_0 : Fin 4 → mxxMat1) (w_249_0 : Fin 4 → mxxMat1) (w_251_0 : Fin 4 → mxxMat1) (w_255_0 : Fin 2 → mxxMat1) (w_260_0 : Fin 2 → mxxMat1) (w_73_0 : mxxMat2) (w_77_0 : mxxMat2) (w_81_0 : mxxMat2) (w_85_0 : mxxMat2) (w_89_0 : mxxMat2) (w_93_0 : mxxMat2) (w_97_0 : mxxMat2) (w_101_0 : mxxMat2) (w_105_0 : mxxMat2) (w_109_0 : mxxMat2) (w_113_0 : mxxMat2) (w_117_0 : mxxMat2) (w_121_0 : mxxMat2) (w_125_0 : mxxMat2) (w_129_0 : mxxMat2) (w_133_0 : mxxMat2) (w_138_0 : mxxMat2) (w_143_0 : mxxMat2) (w_148_0 : mxxMat2) (w_153_0 : mxxMat2) (w_158_0 : mxxMat2) (w_163_0 : mxxMat2) (w_168_0 : mxxMat2) (w_173_0 : mxxMat2) (w_178_0 : mxxMat2) (w_183_0 : mxxMat2) (w_188_0 : mxxMat2) (w_193_0 : mxxMat2) (w_198_0 : mxxMat2) (w_203_0 : mxxMat2) (w_208_0 : mxxMat2) (w_213_0 : mxxMat2) (w_216_0 : mxxMat74) (w_218_0 : mxxMat2) (w_220_0 : mxxMat2) (w_221_0 : mxxMat2) (w_222_0 : mxxMat2) (w_223_0 : mxxMat2) (w_224_0 : mxxMat2) (w_225_0 : Fin 36 → mxxMat2) (w_227_0 : mxxMat1) (w_228_0 : mxxMat1) (w_229_0 : mxxMat1) (w_230_0 : mxxMat1) (w_232_0 : mxxMat1) (w_233_0 : mxxMat14) (w_234_0 : mxxMat14) (w_237_0 : mxxMat1) (w_239_0 : mxxMat1) (w_240_0 : mxxMat14) (w_241_0 : mxxMat14) (w_246_0 : mxxMat1) (w_248_0 : mxxMat1) (w_250_0 : mxxMat1) (w_252_0 : mxxMat1) (w_256_0 : mxxMat1) (w_257_0 : mxxMat1) (w_258_0 : mxxMat1) (w_259_0 : mxxMat0) (w_259_concat_1 : mxxMat2) (w_261_0 : mxxMat1) (w_262_0 : mxxMat1) (w_263_0 : mxxMat1) (w_264_concat_1 : mxxMat2) : Prop :=
  MxxRuntime.sliceMatrix w_216_0 0 2 50 75 w_221_0 ∧
  MxxRuntime.modulusReduceRuns w_221_0 w_222_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 75 ∧
  75 < 100 ∧
  100 ≤ 100 ∧
  MxxRuntime.sliceMatrix w_216_0 0 2 75 100 w_223_0 ∧
  MxxRuntime.modulusReduceRuns w_223_0 w_224_0 ∧
  MxxRuntime.familyPack (36) [w_73_0, w_77_0, w_81_0, w_85_0, w_89_0, w_93_0, w_97_0, w_101_0, w_105_0, w_109_0, w_113_0, w_117_0, w_121_0, w_125_0, w_129_0, w_133_0, w_138_0, w_143_0, w_148_0, w_153_0, w_158_0, w_163_0, w_168_0, w_173_0, w_178_0, w_183_0, w_188_0, w_193_0, w_198_0, w_203_0, w_208_0, w_213_0, w_218_0, w_220_0, w_222_0, w_224_0] w_225_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_227_0 ∧
  MxxRuntime.centeredRebaseRuns w_227_0 w_228_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_230_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_232_0 w_233_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_233_0 w_234_0 ∧
  MxxRuntime.centeredRebaseRuns w_229_0 w_237_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_239_0 w_240_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_240_0 w_241_0 ∧
  MxxRuntime.familyGetStatic w_245_0 (0) w_246_0 ∧
  MxxRuntime.familyGetStatic w_247_0 (0) w_248_0 ∧
  MxxRuntime.familyGetStatic w_249_0 (0) w_250_0 ∧
  MxxRuntime.familyGetStatic w_251_0 (0) w_252_0 ∧
  MxxRuntime.familyGetStatic w_255_0 (1) w_256_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_257_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_258_0 ∧
  MxxRuntime.concatRows w_256_0 w_257_0 w_259_concat_1 ∧
  MxxRuntime.concatRows w_259_concat_1 w_258_0 w_259_0 ∧
  MxxRuntime.familyGetStatic w_260_0 (1) w_261_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_262_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_263_0 ∧
  MxxRuntime.concatRows w_261_0 w_262_0 w_264_concat_1

abbrev generatedRoot.constraints_5 (hashModel : MxxRuntime.HashModel) (w_69_0 : ByteArray) (w_214_0 : mxxMat75) (w_204_0 : Int) (w_205_0 : mxxMat1) (w_206_0 : mxxMat1) (w_207_0 : mxxMat2) (w_208_0 : mxxMat2) (w_209_0 : Int) (w_210_0 : mxxMat1) (w_211_0 : mxxMat1) (w_212_0 : mxxMat2) (w_213_0 : mxxMat2) (w_215_0 : mxxMat73) (w_216_0 : mxxMat74) (w_217_0 : mxxMat2) (w_218_0 : mxxMat2) (w_219_0 : mxxMat2) (w_220_0 : mxxMat2) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_204_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_205_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_204_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_206_0 ∧
  MxxRuntime.concatRows w_205_0 w_206_0 w_207_0 ∧
  MxxRuntime.modulusReduceRuns w_207_0 w_208_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_209_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_210_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_209_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_211_0 ∧
  MxxRuntime.concatRows w_210_0 w_211_0 w_212_0 ∧
  MxxRuntime.modulusReduceRuns w_212_0 w_213_0 ∧
  MxxRuntime.hashSample (hashModel) ([100, 105, 111, 50, 47, 97, 110, 99, 104, 111, 114, 45, 112, 97, 100, 100, 105, 110, 103]) ([]) (w_69_0) w_215_0 ∧
  MxxRuntime.concatColumns w_214_0 w_215_0 w_216_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 100 ∧
  MxxRuntime.sliceMatrix w_216_0 0 2 0 25 w_217_0 ∧
  MxxRuntime.modulusReduceRuns w_217_0 w_218_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 25 ∧
  25 < 50 ∧
  50 ≤ 100 ∧
  MxxRuntime.sliceMatrix w_216_0 0 2 25 50 w_219_0 ∧
  MxxRuntime.modulusReduceRuns w_219_0 w_220_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 50 ∧
  50 < 75 ∧
  75 ≤ 100

abbrev generatedRoot.constraints_4 (hashModel : MxxRuntime.HashModel) (w_69_0 : ByteArray) (w_164_0 : Int) (w_165_0 : mxxMat1) (w_166_0 : mxxMat1) (w_167_0 : mxxMat2) (w_168_0 : mxxMat2) (w_169_0 : Int) (w_170_0 : mxxMat1) (w_171_0 : mxxMat1) (w_172_0 : mxxMat2) (w_173_0 : mxxMat2) (w_174_0 : Int) (w_175_0 : mxxMat1) (w_176_0 : mxxMat1) (w_177_0 : mxxMat2) (w_178_0 : mxxMat2) (w_179_0 : Int) (w_180_0 : mxxMat1) (w_181_0 : mxxMat1) (w_182_0 : mxxMat2) (w_183_0 : mxxMat2) (w_184_0 : Int) (w_185_0 : mxxMat1) (w_186_0 : mxxMat1) (w_187_0 : mxxMat2) (w_188_0 : mxxMat2) (w_189_0 : Int) (w_190_0 : mxxMat1) (w_191_0 : mxxMat1) (w_192_0 : mxxMat2) (w_193_0 : mxxMat2) (w_194_0 : Int) (w_195_0 : mxxMat1) (w_196_0 : mxxMat1) (w_197_0 : mxxMat2) (w_198_0 : mxxMat2) (w_199_0 : Int) (w_200_0 : mxxMat1) (w_201_0 : mxxMat1) (w_202_0 : mxxMat2) (w_203_0 : mxxMat2) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_164_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_165_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_164_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_166_0 ∧
  MxxRuntime.concatRows w_165_0 w_166_0 w_167_0 ∧
  MxxRuntime.modulusReduceRuns w_167_0 w_168_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_169_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_170_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_169_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_171_0 ∧
  MxxRuntime.concatRows w_170_0 w_171_0 w_172_0 ∧
  MxxRuntime.modulusReduceRuns w_172_0 w_173_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_174_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_175_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_174_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_176_0 ∧
  MxxRuntime.concatRows w_175_0 w_176_0 w_177_0 ∧
  MxxRuntime.modulusReduceRuns w_177_0 w_178_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_179_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_180_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_179_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_181_0 ∧
  MxxRuntime.concatRows w_180_0 w_181_0 w_182_0 ∧
  MxxRuntime.modulusReduceRuns w_182_0 w_183_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_184_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_185_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_184_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_186_0 ∧
  MxxRuntime.concatRows w_185_0 w_186_0 w_187_0 ∧
  MxxRuntime.modulusReduceRuns w_187_0 w_188_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_189_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_190_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_189_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_191_0 ∧
  MxxRuntime.concatRows w_190_0 w_191_0 w_192_0 ∧
  MxxRuntime.modulusReduceRuns w_192_0 w_193_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_194_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_195_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_194_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_196_0 ∧
  MxxRuntime.concatRows w_195_0 w_196_0 w_197_0 ∧
  MxxRuntime.modulusReduceRuns w_197_0 w_198_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_199_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_200_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_199_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_201_0 ∧
  MxxRuntime.concatRows w_200_0 w_201_0 w_202_0 ∧
  MxxRuntime.modulusReduceRuns w_202_0 w_203_0

abbrev generatedRoot.constraints_3 (hashModel : MxxRuntime.HashModel) (w_69_0 : ByteArray) (w_126_0 : mxxMat1) (w_127_0 : mxxMat1) (w_128_0 : mxxMat2) (w_129_0 : mxxMat2) (w_130_0 : mxxMat1) (w_131_0 : mxxMat1) (w_132_0 : mxxMat2) (w_133_0 : mxxMat2) (w_134_0 : Int) (w_135_0 : mxxMat1) (w_136_0 : mxxMat1) (w_137_0 : mxxMat2) (w_138_0 : mxxMat2) (w_139_0 : Int) (w_140_0 : mxxMat1) (w_141_0 : mxxMat1) (w_142_0 : mxxMat2) (w_143_0 : mxxMat2) (w_144_0 : Int) (w_145_0 : mxxMat1) (w_146_0 : mxxMat1) (w_147_0 : mxxMat2) (w_148_0 : mxxMat2) (w_149_0 : Int) (w_150_0 : mxxMat1) (w_151_0 : mxxMat1) (w_152_0 : mxxMat2) (w_153_0 : mxxMat2) (w_154_0 : Int) (w_155_0 : mxxMat1) (w_156_0 : mxxMat1) (w_157_0 : mxxMat2) (w_158_0 : mxxMat2) (w_159_0 : Int) (w_160_0 : mxxMat1) (w_161_0 : mxxMat1) (w_162_0 : mxxMat2) (w_163_0 : mxxMat2) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_126_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_127_0 ∧
  MxxRuntime.concatRows w_126_0 w_127_0 w_128_0 ∧
  MxxRuntime.modulusReduceRuns w_128_0 w_129_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_130_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_131_0 ∧
  MxxRuntime.concatRows w_130_0 w_131_0 w_132_0 ∧
  MxxRuntime.modulusReduceRuns w_132_0 w_133_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_134_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_135_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_134_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_136_0 ∧
  MxxRuntime.concatRows w_135_0 w_136_0 w_137_0 ∧
  MxxRuntime.modulusReduceRuns w_137_0 w_138_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_139_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_140_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_139_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_141_0 ∧
  MxxRuntime.concatRows w_140_0 w_141_0 w_142_0 ∧
  MxxRuntime.modulusReduceRuns w_142_0 w_143_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_144_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_145_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_144_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_146_0 ∧
  MxxRuntime.concatRows w_145_0 w_146_0 w_147_0 ∧
  MxxRuntime.modulusReduceRuns w_147_0 w_148_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_149_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_150_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_149_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_151_0 ∧
  MxxRuntime.concatRows w_150_0 w_151_0 w_152_0 ∧
  MxxRuntime.modulusReduceRuns w_152_0 w_153_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_154_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_155_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_154_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_156_0 ∧
  MxxRuntime.concatRows w_155_0 w_156_0 w_157_0 ∧
  MxxRuntime.modulusReduceRuns w_157_0 w_158_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_159_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_160_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_159_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_161_0 ∧
  MxxRuntime.concatRows w_160_0 w_161_0 w_162_0 ∧
  MxxRuntime.modulusReduceRuns w_162_0 w_163_0

abbrev generatedRoot.constraints_2 (hashModel : MxxRuntime.HashModel) (w_69_0 : ByteArray) (w_94_0 : mxxMat1) (w_95_0 : mxxMat1) (w_96_0 : mxxMat2) (w_97_0 : mxxMat2) (w_98_0 : mxxMat1) (w_99_0 : mxxMat1) (w_100_0 : mxxMat2) (w_101_0 : mxxMat2) (w_102_0 : mxxMat1) (w_103_0 : mxxMat1) (w_104_0 : mxxMat2) (w_105_0 : mxxMat2) (w_106_0 : mxxMat1) (w_107_0 : mxxMat1) (w_108_0 : mxxMat2) (w_109_0 : mxxMat2) (w_110_0 : mxxMat1) (w_111_0 : mxxMat1) (w_112_0 : mxxMat2) (w_113_0 : mxxMat2) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_116_0 : mxxMat2) (w_117_0 : mxxMat2) (w_118_0 : mxxMat1) (w_119_0 : mxxMat1) (w_120_0 : mxxMat2) (w_121_0 : mxxMat2) (w_122_0 : mxxMat1) (w_123_0 : mxxMat1) (w_124_0 : mxxMat2) (w_125_0 : mxxMat2) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_94_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_95_0 ∧
  MxxRuntime.concatRows w_94_0 w_95_0 w_96_0 ∧
  MxxRuntime.modulusReduceRuns w_96_0 w_97_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_98_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_99_0 ∧
  MxxRuntime.concatRows w_98_0 w_99_0 w_100_0 ∧
  MxxRuntime.modulusReduceRuns w_100_0 w_101_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_102_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_103_0 ∧
  MxxRuntime.concatRows w_102_0 w_103_0 w_104_0 ∧
  MxxRuntime.modulusReduceRuns w_104_0 w_105_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_106_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_107_0 ∧
  MxxRuntime.concatRows w_106_0 w_107_0 w_108_0 ∧
  MxxRuntime.modulusReduceRuns w_108_0 w_109_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_110_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_111_0 ∧
  MxxRuntime.concatRows w_110_0 w_111_0 w_112_0 ∧
  MxxRuntime.modulusReduceRuns w_112_0 w_113_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_114_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_115_0 ∧
  MxxRuntime.concatRows w_114_0 w_115_0 w_116_0 ∧
  MxxRuntime.modulusReduceRuns w_116_0 w_117_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_118_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_119_0 ∧
  MxxRuntime.concatRows w_118_0 w_119_0 w_120_0 ∧
  MxxRuntime.modulusReduceRuns w_120_0 w_121_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_122_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_123_0 ∧
  MxxRuntime.concatRows w_122_0 w_123_0 w_124_0 ∧
  MxxRuntime.modulusReduceRuns w_124_0 w_125_0

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_5_0 : ByteArray) (w_69_0 : ByteArray) (w_55_0 : mxxMat1) (w_56_0 : mxxMat1) (w_58_0 : mxxMat1) (w_59_0 : mxxMat14) (w_60_0 : mxxMat14) (w_62_0 : mxxMat1) (w_63_0 : mxxMat1) (w_65_0 : mxxMat1) (w_66_0 : mxxMat14) (w_67_0 : mxxMat14) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_72_0 : mxxMat2) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) (w_76_0 : mxxMat2) (w_77_0 : mxxMat2) (w_78_0 : mxxMat1) (w_79_0 : mxxMat1) (w_80_0 : mxxMat2) (w_81_0 : mxxMat2) (w_82_0 : mxxMat1) (w_83_0 : mxxMat1) (w_84_0 : mxxMat2) (w_85_0 : mxxMat2) (w_86_0 : mxxMat1) (w_87_0 : mxxMat1) (w_88_0 : mxxMat2) (w_89_0 : mxxMat2) (w_90_0 : mxxMat1) (w_91_0 : mxxMat1) (w_92_0 : mxxMat2) (w_93_0 : mxxMat2) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_55_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_62_0 ∧
  MxxRuntime.centeredRebaseRuns w_55_0 w_63_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_65_0 w_66_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_66_0 w_67_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_70_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_71_0 ∧
  MxxRuntime.concatRows w_70_0 w_71_0 w_72_0 ∧
  MxxRuntime.modulusReduceRuns w_72_0 w_73_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_74_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_75_0 ∧
  MxxRuntime.concatRows w_74_0 w_75_0 w_76_0 ∧
  MxxRuntime.modulusReduceRuns w_76_0 w_77_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_78_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_79_0 ∧
  MxxRuntime.concatRows w_78_0 w_79_0 w_80_0 ∧
  MxxRuntime.modulusReduceRuns w_80_0 w_81_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_82_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_83_0 ∧
  MxxRuntime.concatRows w_82_0 w_83_0 w_84_0 ∧
  MxxRuntime.modulusReduceRuns w_84_0 w_85_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_69_0) w_86_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_69_0) w_87_0 ∧
  MxxRuntime.concatRows w_86_0 w_87_0 w_88_0 ∧
  MxxRuntime.modulusReduceRuns w_88_0 w_89_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_69_0) w_90_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_69_0) w_91_0 ∧
  MxxRuntime.concatRows w_90_0 w_91_0 w_92_0 ∧
  MxxRuntime.modulusReduceRuns w_92_0 w_93_0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : Fin 8 → mxxMat1) (w_1_0 : Fin 8 → mxxMat1) (w_2_0 : Fin 8 → mxxMat1) (w_3_0 : Fin 8 → mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_5_0 : ByteArray) (w_6_0 : Fin 8 → mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_8_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_10_0 : Fin 8 → mxxMat1) (w_11_0 : Fin 8 → mxxMat1) (w_12_0 : Fin 8 → mxxMat1) (w_13_0 : Fin 16 → Int) (w_14_0 : Fin 2 → mxxMat1) (w_15_0 : Fin 2 → mxxMat1) (w_16_0 : Fin 16 → Int) (w_19_0 : mxxMat1) (w_20_0 : Fin 2 → mxxMat5) (w_23_0 : mxxMat1) (w_24_0 : ByteArray) (w_25_0 : Fin 16 → mxxMat1) (w_26_0 : Fin 16 → mxxMat1) (w_29_0 : Fin 16 → mxxMat1) (w_30_0 : Fin 16 → mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat1) (w_27_0 : Fin 16 → mxxMat1) (w_27_1 : Fin 16 → mxxMat1) (w_27_2 : Fin 16 → mxxMat1) (w_27_3 : Fin 16 → mxxMat1) (w_27_4 : Fin 16 → mxxMat1) (w_27_5 : Fin 16 → mxxMat1) (w_28_0 : Fin 8 → mxxMat1) (w_28_1 : Fin 8 → mxxMat1) (w_28_2 : Fin 8 → mxxMat1) (w_28_3 : Fin 8 → mxxMat1) (w_28_4 : Fin 8 → mxxMat1) (w_28_5 : Fin 8 → mxxMat1) (w_31_0 : Fin 16 → mxxMat1) (w_31_1 : Fin 16 → mxxMat1) (w_31_2 : Fin 16 → mxxMat1) (w_31_3 : Fin 16 → mxxMat1) (w_31_4 : Fin 16 → mxxMat1) (w_31_5 : Fin 16 → mxxMat1) (w_32_0 : Fin 8 → mxxMat1) (w_32_1 : Fin 8 → mxxMat1) (w_32_2 : Fin 8 → mxxMat1) (w_32_3 : Fin 8 → mxxMat1) (w_32_4 : Fin 8 → mxxMat1) (w_32_5 : Fin 8 → mxxMat1) (w_33_0 : Fin 8 → mxxMat1) (w_33_1 : Fin 8 → mxxMat1) (w_33_2 : Fin 8 → mxxMat1) (w_33_3 : Fin 8 → mxxMat1) (w_33_4 : Fin 8 → mxxMat1) (w_33_5 : Fin 8 → mxxMat1) (w_34_0 : Fin 8 → mxxMat1) (w_34_1 : Fin 8 → mxxMat1) (w_34_2 : Fin 8 → mxxMat1) (w_34_3 : Fin 8 → mxxMat1) (w_34_4 : Fin 8 → mxxMat1) (w_34_5 : Fin 8 → mxxMat1) (w_35_0 : Fin 8 → mxxMat1) (w_35_1 : Fin 8 → mxxMat1) (w_35_2 : Fin 8 → mxxMat1) (w_35_3 : Fin 8 → mxxMat1) (w_35_4 : Fin 8 → mxxMat1) (w_35_5 : Fin 8 → mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat1) (w_39_0 : mxxMat1) (w_40_0 : mxxMat1) (w_41_0 : mxxMat1) (w_42_0 : mxxMat1) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat1) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat1) (w_51_0 : mxxMat1) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_54_1 : mxxMat1) (w_54_2 : mxxMat1) (w_54_3 : mxxMat1) (w_54_4 : mxxMat1) (w_54_5 : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_17_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_5_0) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_21_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_5_0) w_22_0 ∧
  (16) = 16 ∧
  (∀ i : Fin 16, ∃ w_14_0_gathered : mxxMat1, ∃ w_15_0_gathered : mxxMat1, ∃ w_20_0_gathered : mxxMat5, 0 ≤ (w_16_0 i) ∧
  (w_16_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_14_0 (w_16_0 i) w_14_0_gathered ∧
  0 ≤ (w_16_0 i) ∧
  (w_16_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_15_0 (w_16_0 i) w_15_0_gathered ∧
  0 ≤ (w_16_0 i) ∧
  (w_16_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_20_0 (w_16_0 i) w_20_0_gathered ∧
  parallel_generatedRoot_27 backend hashModel params i ((w_16_0 i), (w_14_0_gathered, (w_5_0, (w_15_0_gathered, (w_17_0, (w_18_0, (w_19_0, (w_20_0_gathered, (w_21_0, (w_22_0, (w_23_0, (w_24_0, ((w_25_0 i), ((w_26_0 i), ())))))))))))))) ((w_27_0 i), ((w_27_1 i), ((w_27_2 i), ((w_27_3 i), ((w_27_4 i), ((w_27_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_28 backend hashModel params i (w_27_0, (w_27_1, (w_27_2, (w_27_3, (w_27_4, (w_27_5, ())))))) ((w_28_0 i), ((w_28_1 i), ((w_28_2 i), ((w_28_3 i), ((w_28_4 i), ((w_28_5 i), ()))))))) ∧
  (16) = 16 ∧
  (∀ i : Fin 16, ∃ w_14_0_gathered : mxxMat1, ∃ w_15_0_gathered : mxxMat1, ∃ w_20_0_gathered : mxxMat5, 0 ≤ (w_13_0 i) ∧
  (w_13_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_14_0 (w_13_0 i) w_14_0_gathered ∧
  0 ≤ (w_13_0 i) ∧
  (w_13_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_15_0 (w_13_0 i) w_15_0_gathered ∧
  0 ≤ (w_13_0 i) ∧
  (w_13_0 i) < 2 ∧
  MxxRuntime.familyGetDynamic w_20_0 (w_13_0 i) w_20_0_gathered ∧
  parallel_generatedRoot_31 backend hashModel params i ((w_13_0 i), (w_14_0_gathered, (w_5_0, (w_15_0_gathered, (w_28_0, (w_28_2, (w_28_4, (w_20_0_gathered, (w_28_1, (w_28_3, (w_28_5, (w_24_0, ((w_29_0 i), ((w_30_0 i), ())))))))))))))) ((w_31_0 i), ((w_31_1 i), ((w_31_2 i), ((w_31_3 i), ((w_31_4 i), ((w_31_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_32 backend hashModel params i (w_31_0, (w_31_1, (w_31_2, (w_31_3, (w_31_4, (w_31_5, ())))))) ((w_32_0 i), ((w_32_1 i), ((w_32_2 i), ((w_32_3 i), ((w_32_4 i), ((w_32_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_33 backend hashModel params i (w_7_0, (w_8_0, (w_9_0, (w_10_0, (w_11_0, (w_5_0, (w_12_0, ((w_32_4 i), ((w_32_0 i), ((w_32_2 i), ((w_32_5 i), ((w_32_1 i), ((w_32_3 i), ()))))))))))))) ((w_33_0 i), ((w_33_1 i), ((w_33_2 i), ((w_33_3 i), ((w_33_4 i), ((w_33_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_34 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, (w_6_0, ((w_33_4 i), ((w_33_0 i), ((w_33_2 i), ((w_33_5 i), ((w_33_1 i), ((w_33_3 i), ()))))))))))))) ((w_34_0 i), ((w_34_1 i), ((w_34_2 i), ((w_34_3 i), ((w_34_4 i), ((w_34_5 i), ()))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_35 backend hashModel params i ((w_34_0 i), ((w_34_1 i), ((w_34_2 i), ((w_34_3 i), ((w_34_4 i), ((w_34_5 i), ())))))) ((w_35_0 i), ((w_35_1 i), ((w_35_2 i), ((w_35_3 i), ((w_35_4 i), ((w_35_5 i), ()))))))) ∧
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
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_54 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_35_0, (w_35_1, (w_35_2, (w_35_3, (w_35_4, (w_35_5, ())))))))))))) next) (Int.toNat (6)) (w_38_0, (w_41_0, (w_44_0, (w_47_0, (w_50_0, (w_53_0, ())))))) (w_54_0, (w_54_1, (w_54_2, (w_54_3, (w_54_4, (w_54_5, ()))))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 16 → Int) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 16 → Int) × mxxMat1 × (Fin 2 → mxxMat5) × mxxMat1 × ByteArray × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × ByteArray × mxxMat75 × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × mxxMat1 × (Fin 2 → mxxMat5) × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat5) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 14400 → Int) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 14400 → Int) × mxxMat1 × (Fin 2 → mxxMat5) × mxxMat1 × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 72000 → Int) × (Fin 72000 → Int) × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 8 → mxxMat1 := inputs.1
  let w_1_0 : Fin 8 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 8 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 8 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.2.2.1
  let w_6_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_7_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_12_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_13_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_15_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 2 → mxxMat5 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_30_0 : Fin 16 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_69_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_214_0 : mxxMat75 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_245_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_247_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_249_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_251_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_253_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_254_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_255_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_260_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_273_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_274_0 : Fin 2 → mxxMat5 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_288_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_296_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_306_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_366_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_371_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_430_0 : Fin 4 → mxxMat5 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_665_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_667_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_669_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_671_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_673_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_674_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_721_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_723_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_725_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_727_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_729_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_730_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_777_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_779_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_781_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_783_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_785_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_786_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_833_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_835_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_837_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_839_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_841_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_842_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_889_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_891_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_893_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_895_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_897_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_898_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_945_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_947_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_949_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_951_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_953_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_954_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1001_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1003_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1005_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1007_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1009_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1010_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1059_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1060_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1061_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1062_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1063_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1064_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1065_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1066_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1069_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1070_0 : Fin 2 → mxxMat5 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1073_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1074_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1075_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1078_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1079_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1116_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1117_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1118_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1119_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1120_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1121_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1124_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1127_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1128_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1129_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1132_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1133_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_27_0 := witness.w_27_0
  let w_27_1 := witness.w_27_1
  let w_27_2 := witness.w_27_2
  let w_27_3 := witness.w_27_3
  let w_27_4 := witness.w_27_4
  let w_27_5 := witness.w_27_5
  let w_28_0 := witness.w_28_0
  let w_28_1 := witness.w_28_1
  let w_28_2 := witness.w_28_2
  let w_28_3 := witness.w_28_3
  let w_28_4 := witness.w_28_4
  let w_28_5 := witness.w_28_5
  let w_31_0 := witness.w_31_0
  let w_31_1 := witness.w_31_1
  let w_31_2 := witness.w_31_2
  let w_31_3 := witness.w_31_3
  let w_31_4 := witness.w_31_4
  let w_31_5 := witness.w_31_5
  let w_32_0 := witness.w_32_0
  let w_32_1 := witness.w_32_1
  let w_32_2 := witness.w_32_2
  let w_32_3 := witness.w_32_3
  let w_32_4 := witness.w_32_4
  let w_32_5 := witness.w_32_5
  let w_33_0 := witness.w_33_0
  let w_33_1 := witness.w_33_1
  let w_33_2 := witness.w_33_2
  let w_33_3 := witness.w_33_3
  let w_33_4 := witness.w_33_4
  let w_33_5 := witness.w_33_5
  let w_34_0 := witness.w_34_0
  let w_34_1 := witness.w_34_1
  let w_34_2 := witness.w_34_2
  let w_34_3 := witness.w_34_3
  let w_34_4 := witness.w_34_4
  let w_34_5 := witness.w_34_5
  let w_35_0 := witness.w_35_0
  let w_35_1 := witness.w_35_1
  let w_35_2 := witness.w_35_2
  let w_35_3 := witness.w_35_3
  let w_35_4 := witness.w_35_4
  let w_35_5 := witness.w_35_5
  let w_36_0 := witness.w_36_0
  let w_37_0 := witness.w_37_0
  let w_38_0 : mxxMat1 := MxxRuntime.matrixAdd w_36_0 w_37_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat1 := MxxRuntime.matrixAdd w_39_0 w_40_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_43_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat1 := MxxRuntime.matrixAdd w_45_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 : mxxMat1 := MxxRuntime.matrixAdd w_48_0 w_49_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : mxxMat1 := MxxRuntime.matrixAdd w_51_0 w_52_0
  let w_54_0 := witness.w_54_0
  let w_54_1 := witness.w_54_1
  let w_54_2 := witness.w_54_2
  let w_54_3 := witness.w_54_3
  let w_54_4 := witness.w_54_4
  let w_54_5 := witness.w_54_5
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_58_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_56_0 w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixMul w_54_1 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_65_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_63_0 w_64_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat1 := MxxRuntime.matrixMul w_62_0 w_67_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 := witness.w_72_0
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 := witness.w_78_0
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 := witness.w_82_0
  let w_83_0 := witness.w_83_0
  let w_84_0 := witness.w_84_0
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 := witness.w_87_0
  let w_88_0 := witness.w_88_0
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 := witness.w_91_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 := witness.w_94_0
  let w_95_0 := witness.w_95_0
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_98_0 := witness.w_98_0
  let w_99_0 := witness.w_99_0
  let w_100_0 := witness.w_100_0
  let w_101_0 := witness.w_101_0
  let w_102_0 := witness.w_102_0
  let w_103_0 := witness.w_103_0
  let w_104_0 := witness.w_104_0
  let w_105_0 := witness.w_105_0
  let w_106_0 := witness.w_106_0
  let w_107_0 := witness.w_107_0
  let w_108_0 := witness.w_108_0
  let w_109_0 := witness.w_109_0
  let w_110_0 := witness.w_110_0
  let w_111_0 := witness.w_111_0
  let w_112_0 := witness.w_112_0
  let w_113_0 := witness.w_113_0
  let w_114_0 := witness.w_114_0
  let w_115_0 := witness.w_115_0
  let w_116_0 := witness.w_116_0
  let w_117_0 := witness.w_117_0
  let w_118_0 := witness.w_118_0
  let w_119_0 := witness.w_119_0
  let w_120_0 := witness.w_120_0
  let w_121_0 := witness.w_121_0
  let w_122_0 := witness.w_122_0
  let w_123_0 := witness.w_123_0
  let w_124_0 := witness.w_124_0
  let w_125_0 := witness.w_125_0
  let w_126_0 := witness.w_126_0
  let w_127_0 := witness.w_127_0
  let w_128_0 := witness.w_128_0
  let w_129_0 := witness.w_129_0
  let w_130_0 := witness.w_130_0
  let w_131_0 := witness.w_131_0
  let w_132_0 := witness.w_132_0
  let w_133_0 := witness.w_133_0
  let w_134_0 : Int := 1
  let w_135_0 := witness.w_135_0
  let w_136_0 := witness.w_136_0
  let w_137_0 := witness.w_137_0
  let w_138_0 := witness.w_138_0
  let w_139_0 : Int := 1
  let w_140_0 := witness.w_140_0
  let w_141_0 := witness.w_141_0
  let w_142_0 := witness.w_142_0
  let w_143_0 := witness.w_143_0
  let w_144_0 : Int := 3
  let w_145_0 := witness.w_145_0
  let w_146_0 := witness.w_146_0
  let w_147_0 := witness.w_147_0
  let w_148_0 := witness.w_148_0
  let w_149_0 : Int := 3
  let w_150_0 := witness.w_150_0
  let w_151_0 := witness.w_151_0
  let w_152_0 := witness.w_152_0
  let w_153_0 := witness.w_153_0
  let w_154_0 : Int := 5
  let w_155_0 := witness.w_155_0
  let w_156_0 := witness.w_156_0
  let w_157_0 := witness.w_157_0
  let w_158_0 := witness.w_158_0
  let w_159_0 : Int := 5
  let w_160_0 := witness.w_160_0
  let w_161_0 := witness.w_161_0
  let w_162_0 := witness.w_162_0
  let w_163_0 := witness.w_163_0
  let w_164_0 : Int := 7
  let w_165_0 := witness.w_165_0
  let w_166_0 := witness.w_166_0
  let w_167_0 := witness.w_167_0
  let w_168_0 := witness.w_168_0
  let w_169_0 : Int := 7
  let w_170_0 := witness.w_170_0
  let w_171_0 := witness.w_171_0
  let w_172_0 := witness.w_172_0
  let w_173_0 := witness.w_173_0
  let w_174_0 : Int := 9
  let w_175_0 := witness.w_175_0
  let w_176_0 := witness.w_176_0
  let w_177_0 := witness.w_177_0
  let w_178_0 := witness.w_178_0
  let w_179_0 : Int := 9
  let w_180_0 := witness.w_180_0
  let w_181_0 := witness.w_181_0
  let w_182_0 := witness.w_182_0
  let w_183_0 := witness.w_183_0
  let w_184_0 : Int := 11
  let w_185_0 := witness.w_185_0
  let w_186_0 := witness.w_186_0
  let w_187_0 := witness.w_187_0
  let w_188_0 := witness.w_188_0
  let w_189_0 : Int := 11
  let w_190_0 := witness.w_190_0
  let w_191_0 := witness.w_191_0
  let w_192_0 := witness.w_192_0
  let w_193_0 := witness.w_193_0
  let w_194_0 : Int := 13
  let w_195_0 := witness.w_195_0
  let w_196_0 := witness.w_196_0
  let w_197_0 := witness.w_197_0
  let w_198_0 := witness.w_198_0
  let w_199_0 : Int := 13
  let w_200_0 := witness.w_200_0
  let w_201_0 := witness.w_201_0
  let w_202_0 := witness.w_202_0
  let w_203_0 := witness.w_203_0
  let w_204_0 : Int := 15
  let w_205_0 := witness.w_205_0
  let w_206_0 := witness.w_206_0
  let w_207_0 := witness.w_207_0
  let w_208_0 := witness.w_208_0
  let w_209_0 : Int := 15
  let w_210_0 := witness.w_210_0
  let w_211_0 := witness.w_211_0
  let w_212_0 := witness.w_212_0
  let w_213_0 := witness.w_213_0
  let w_215_0 := witness.w_215_0
  let w_216_0 := witness.w_216_0
  let w_217_0 := witness.w_217_0
  let w_218_0 := witness.w_218_0
  let w_219_0 := witness.w_219_0
  let w_220_0 := witness.w_220_0
  let w_221_0 := witness.w_221_0
  let w_222_0 := witness.w_222_0
  let w_223_0 := witness.w_223_0
  let w_224_0 := witness.w_224_0
  let w_225_0 := witness.w_225_0
  let w_226_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_227_0 := witness.w_227_0
  let w_228_0 := witness.w_228_0
  let w_229_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_226_0 w_228_0
  let w_230_0 := witness.w_230_0
  let w_231_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_232_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_230_0 w_231_0
  let w_233_0 := witness.w_233_0
  let w_234_0 := witness.w_234_0
  let w_235_0 : mxxMat1 := MxxRuntime.matrixMul w_17_0 w_234_0
  let w_236_0 : mxxMat1 := MxxRuntime.matrixMul w_21_0 w_234_0
  let w_237_0 := witness.w_237_0
  let w_238_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_239_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_237_0 w_238_0
  let w_240_0 := witness.w_240_0
  let w_241_0 := witness.w_241_0
  let w_242_0 : mxxMat1 := MxxRuntime.matrixMul w_61_0 w_241_0
  let w_243_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_241_0
  let w_244_0 : mxxMat1 := (0 : mxxMat1)
  let w_246_0 := witness.w_246_0
  let w_248_0 := witness.w_248_0
  let w_250_0 := witness.w_250_0
  let w_252_0 := witness.w_252_0
  let w_256_0 := witness.w_256_0
  let w_257_0 := witness.w_257_0
  let w_258_0 := witness.w_258_0
  let w_259_0 := witness.w_259_0
  let w_259_concat_1 := witness.w_259_concat_1
  let w_261_0 := witness.w_261_0
  let w_262_0 := witness.w_262_0
  let w_263_0 := witness.w_263_0
  let w_264_0 := witness.w_264_0
  let w_264_concat_1 := witness.w_264_concat_1
  let w_265_0 := witness.w_265_0
  let w_266_0 := witness.w_266_0
  let w_267_0 := witness.w_267_0
  let w_268_0 := witness.w_268_0
  let w_269_0 := witness.w_269_0
  let w_270_0 := witness.w_270_0
  let w_271_0 : mxxMat0 := MxxRuntime.matrixMul w_265_0 w_270_0
  let w_272_0 := witness.w_272_0
  let w_275_0 := witness.w_275_0
  let w_276_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_273_0 w_275_0
  let w_277_0 : mxxMat1 := MxxRuntime.matrixAdd w_272_0 w_276_0
  let w_278_0 := witness.w_278_0
  let w_279_0 := witness.w_279_0
  let w_280_0 := witness.w_280_0
  let w_280_concat_1 := witness.w_280_concat_1
  let w_281_0 := witness.w_281_0
  let w_282_0 := witness.w_282_0
  let w_283_0 := witness.w_283_0
  let w_284_0 := witness.w_284_0
  let w_285_0 := witness.w_285_0
  let w_286_0 : mxxMat0 := MxxRuntime.matrixMul w_265_0 w_285_0
  let w_287_0 := witness.w_287_0
  let w_289_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_288_0 w_275_0
  let w_290_0 : mxxMat1 := MxxRuntime.matrixAdd w_287_0 w_289_0
  let w_291_0 := witness.w_291_0
  let w_292_0 := witness.w_292_0
  let w_293_0 := witness.w_293_0
  let w_293_concat_1 := witness.w_293_concat_1
  let w_294_0 := witness.w_294_0
  let w_295_0 := witness.w_295_0
  let w_297_0 := witness.w_297_0
  let w_298_0 := witness.w_298_0
  let w_299_0 := witness.w_299_0
  let w_300_0 := witness.w_300_0
  let w_301_0 : mxxMat0 := MxxRuntime.matrixMul w_294_0 w_300_0
  let w_302_0 := witness.w_302_0
  let w_303_0 := witness.w_303_0
  let w_304_0 := witness.w_304_0
  let w_305_0 := witness.w_305_0
  let w_307_0 := witness.w_307_0
  let w_308_0 := witness.w_308_0
  let w_309_0 := witness.w_309_0
  let w_310_0 := witness.w_310_0
  let w_311_0 : mxxMat0 := MxxRuntime.matrixMul w_294_0 w_310_0
  let w_312_0 := witness.w_312_0
  let w_313_0 := witness.w_313_0
  let w_314_0 := witness.w_314_0
  let w_315_0 := witness.w_315_0
  let w_315_1 := witness.w_315_1
  let w_315_2 := witness.w_315_2
  let w_315_3 := witness.w_315_3
  let w_315_4 := witness.w_315_4
  let w_315_5 := witness.w_315_5
  let w_315_6 := witness.w_315_6
  let w_315_7 := witness.w_315_7
  let w_315_8 := witness.w_315_8
  let w_315_9 := witness.w_315_9
  let w_316_0 := witness.w_316_0
  let w_317_0 : Int := (MxxIR.selectInt (3) [(1), (3), (5), (15)])
  let w_318_0 := witness.w_318_0
  let w_319_0 := witness.w_319_0
  let w_320_0 := witness.w_320_0
  let w_320_concat_1 := witness.w_320_concat_1
  let w_321_0 := witness.w_321_0
  let w_322_0 := witness.w_322_0
  let w_323_0 := witness.w_323_0
  let w_324_0 := witness.w_324_0
  let w_324_concat_1 := witness.w_324_concat_1
  let w_325_0 := witness.w_325_0
  let w_326_0 := witness.w_326_0
  let w_327_0 := witness.w_327_0
  let w_328_0 := witness.w_328_0
  let w_329_0 := witness.w_329_0
  let w_329_concat_1 := witness.w_329_concat_1
  let w_330_0 := witness.w_330_0
  let w_331_0 := witness.w_331_0
  let w_332_0 := witness.w_332_0
  let w_333_0 := witness.w_333_0
  let w_333_concat_1 := witness.w_333_concat_1
  let w_334_0 := witness.w_334_0
  let w_335_0 := witness.w_335_0
  let w_336_0 := witness.w_336_0
  let w_337_0 := witness.w_337_0
  let w_338_0 : mxxMat0 := MxxRuntime.matrixMul w_334_0 w_337_0
  let w_339_0 := witness.w_339_0
  let w_340_0 := witness.w_340_0
  let w_341_0 := witness.w_341_0
  let w_342_0 := witness.w_342_0
  let w_343_0 := witness.w_343_0
  let w_344_0 : mxxMat0 := MxxRuntime.matrixMul w_325_0 w_343_0
  let w_345_0 := witness.w_345_0
  let w_346_0 : mxxMat1 := MxxRuntime.matrixAdd w_315_2 w_345_0
  let w_347_0 := witness.w_347_0
  let w_348_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_349_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_347_0 w_348_0
  let w_350_0 := witness.w_350_0
  let w_351_0 := witness.w_351_0
  let w_352_0 : mxxMat1 := MxxRuntime.matrixMul w_346_0 w_351_0
  let w_353_0 := witness.w_353_0
  let w_354_0 := witness.w_354_0
  let w_355_0 := witness.w_355_0
  let w_356_0 : mxxMat0 := MxxRuntime.matrixMul w_334_0 w_355_0
  let w_357_0 := witness.w_357_0
  let w_358_0 := witness.w_358_0
  let w_359_0 := witness.w_359_0
  let w_360_0 := witness.w_360_0
  let w_361_0 := witness.w_361_0
  let w_362_0 : mxxMat0 := MxxRuntime.matrixMul w_325_0 w_361_0
  let w_363_0 := witness.w_363_0
  let w_364_0 : mxxMat1 := MxxRuntime.matrixAdd w_315_3 w_363_0
  let w_365_0 : mxxMat1 := MxxRuntime.matrixMul w_364_0 w_351_0
  let w_367_0 := witness.w_367_0
  let w_368_0 := witness.w_368_0
  let w_369_0 := witness.w_369_0
  let w_370_0 := witness.w_370_0
  let w_370_concat_1 := witness.w_370_concat_1
  let w_372_0 := witness.w_372_0
  let w_373_0 := witness.w_373_0
  let w_374_0 := witness.w_374_0
  let w_375_0 := witness.w_375_0
  let w_375_concat_1 := witness.w_375_concat_1
  let w_376_0 := witness.w_376_0
  let w_377_0 := witness.w_377_0
  let w_378_0 := witness.w_378_0
  let w_379_0 := witness.w_379_0
  let w_380_0 := witness.w_380_0
  let w_381_0 := witness.w_381_0
  let w_382_0 : mxxMat0 := MxxRuntime.matrixMul w_376_0 w_381_0
  let w_383_0 := witness.w_383_0
  let w_384_0 := witness.w_384_0
  let w_385_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_386_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_384_0 w_385_0
  let w_387_0 := witness.w_387_0
  let w_388_0 := witness.w_388_0
  let w_389_0 : mxxMat1 := MxxRuntime.matrixMul w_383_0 w_388_0
  let w_390_0 := witness.w_390_0
  let w_391_0 : mxxMat1 := MxxRuntime.matrixMul w_389_0 w_390_0
  let w_392_0 := witness.w_392_0
  let w_393_0 := witness.w_393_0
  let w_394_0 := witness.w_394_0
  let w_395_0 := witness.w_395_0
  let w_396_0 := witness.w_396_0
  let w_397_0 : mxxMat0 := MxxRuntime.matrixMul w_376_0 w_396_0
  let w_398_0 := witness.w_398_0
  let w_399_0 : mxxMat1 := MxxRuntime.matrixMul w_398_0 w_388_0
  let w_400_0 : mxxMat1 := MxxRuntime.matrixMul w_54_3 w_60_0
  let w_401_0 : mxxMat1 := MxxRuntime.matrixMul w_400_0 w_241_0
  let w_402_0 := witness.w_402_0
  let w_403_0 : mxxMat1 := MxxRuntime.matrixMul w_399_0 w_402_0
  let w_404_0 : mxxMat1 := MxxRuntime.matrixAdd w_391_0 w_403_0
  let w_405_0 := witness.w_405_0
  let w_406_0 : mxxMat1 := MxxRuntime.matrixMul w_389_0 w_405_0
  let w_407_0 := witness.w_407_0
  let w_408_0 : mxxMat1 := MxxRuntime.matrixMul w_407_0 w_67_0
  let w_409_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_241_0
  let w_410_0 := witness.w_410_0
  let w_411_0 : mxxMat1 := MxxRuntime.matrixMul w_399_0 w_410_0
  let w_412_0 : mxxMat1 := MxxRuntime.matrixAdd w_406_0 w_411_0
  let w_413_0 := witness.w_413_0
  let w_414_0 := witness.w_414_0
  let w_415_0 := witness.w_415_0
  let w_416_0 := witness.w_416_0
  let w_416_concat_1 := witness.w_416_concat_1
  let w_417_0 := witness.w_417_0
  let w_418_0 := witness.w_418_0
  let w_419_0 := witness.w_419_0
  let w_420_0 := witness.w_420_0
  let w_420_concat_1 := witness.w_420_concat_1
  let w_421_0 := witness.w_421_0
  let w_422_0 := witness.w_422_0
  let w_423_0 := witness.w_423_0
  let w_424_0 := witness.w_424_0
  let w_425_0 := witness.w_425_0
  let w_426_0 := witness.w_426_0
  let w_427_0 : mxxMat0 := MxxRuntime.matrixMul w_421_0 w_426_0
  let w_428_0 := witness.w_428_0
  let w_429_0 := witness.w_429_0
  let w_431_0 := witness.w_431_0
  let w_432_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_429_0 w_431_0
  let w_433_0 : mxxMat1 := MxxRuntime.matrixAdd w_428_0 w_432_0
  let w_434_0 := witness.w_434_0
  let w_435_0 := witness.w_435_0
  let w_436_0 := witness.w_436_0
  let w_436_concat_1 := witness.w_436_concat_1
  let w_437_0 := witness.w_437_0
  let w_438_0 := witness.w_438_0
  let w_439_0 := witness.w_439_0
  let w_440_0 := witness.w_440_0
  let w_441_0 := witness.w_441_0
  let w_442_0 : mxxMat0 := MxxRuntime.matrixMul w_421_0 w_441_0
  let w_443_0 := witness.w_443_0
  let w_444_0 := witness.w_444_0
  let w_445_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_444_0 w_431_0
  let w_446_0 : mxxMat1 := MxxRuntime.matrixAdd w_443_0 w_445_0
  let w_447_0 := witness.w_447_0
  let w_448_0 := witness.w_448_0
  let w_449_0 := witness.w_449_0
  let w_449_concat_1 := witness.w_449_concat_1
  let w_450_0 := witness.w_450_0
  let w_451_0 := witness.w_451_0
  let w_452_0 := witness.w_452_0
  let w_453_0 := witness.w_453_0
  let w_454_0 := witness.w_454_0
  let w_454_concat_1 := witness.w_454_concat_1
  let w_455_0 := witness.w_455_0
  let w_456_0 := witness.w_456_0
  let w_457_0 := witness.w_457_0
  let w_458_0 := witness.w_458_0
  let w_458_concat_1 := witness.w_458_concat_1
  let w_459_0 := witness.w_459_0
  let w_460_0 := witness.w_460_0
  let w_461_0 := witness.w_461_0
  let w_462_0 := witness.w_462_0
  let w_463_0 := witness.w_463_0
  let w_464_0 := witness.w_464_0
  let w_465_0 : mxxMat0 := MxxRuntime.matrixMul w_459_0 w_464_0
  let w_466_0 := witness.w_466_0
  let w_467_0 := witness.w_467_0
  let w_468_0 := witness.w_468_0
  let w_469_0 := witness.w_469_0
  let w_470_0 := witness.w_470_0
  let w_471_0 : mxxMat0 := MxxRuntime.matrixMul w_450_0 w_470_0
  let w_472_0 := witness.w_472_0
  let w_473_0 := witness.w_473_0
  let w_474_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_475_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_473_0 w_474_0
  let w_476_0 := witness.w_476_0
  let w_477_0 := witness.w_477_0
  let w_478_0 : mxxMat1 := MxxRuntime.matrixMul w_472_0 w_477_0
  let w_479_0 := witness.w_479_0
  let w_480_0 : mxxMat1 := MxxRuntime.matrixMul w_478_0 w_479_0
  let w_481_0 := witness.w_481_0
  let w_482_0 := witness.w_482_0
  let w_483_0 := witness.w_483_0
  let w_484_0 := witness.w_484_0
  let w_485_0 := witness.w_485_0
  let w_486_0 : mxxMat0 := MxxRuntime.matrixMul w_459_0 w_485_0
  let w_487_0 := witness.w_487_0
  let w_488_0 := witness.w_488_0
  let w_489_0 := witness.w_489_0
  let w_490_0 := witness.w_490_0
  let w_491_0 := witness.w_491_0
  let w_492_0 : mxxMat0 := MxxRuntime.matrixMul w_450_0 w_491_0
  let w_493_0 := witness.w_493_0
  let w_494_0 : mxxMat1 := MxxRuntime.matrixMul w_493_0 w_477_0
  let w_495_0 := witness.w_495_0
  let w_496_0 : mxxMat1 := MxxRuntime.matrixMul w_494_0 w_495_0
  let w_497_0 : mxxMat1 := MxxRuntime.matrixAdd w_480_0 w_496_0
  let w_498_0 := witness.w_498_0
  let w_499_0 : mxxMat1 := MxxRuntime.matrixMul w_478_0 w_498_0
  let w_500_0 := witness.w_500_0
  let w_501_0 : mxxMat1 := MxxRuntime.matrixMul w_494_0 w_500_0
  let w_502_0 : mxxMat1 := MxxRuntime.matrixAdd w_499_0 w_501_0
  let w_503_0 := witness.w_503_0
  let w_504_0 := witness.w_504_0
  let w_505_0 := witness.w_505_0
  let w_506_0 := witness.w_506_0
  let w_506_concat_1 := witness.w_506_concat_1
  let w_507_0 := witness.w_507_0
  let w_508_0 := witness.w_508_0
  let w_509_0 := witness.w_509_0
  let w_510_0 := witness.w_510_0
  let w_510_concat_1 := witness.w_510_concat_1
  let w_511_0 := witness.w_511_0
  let w_512_0 := witness.w_512_0
  let w_513_0 := witness.w_513_0
  let w_514_0 := witness.w_514_0
  let w_515_0 := witness.w_515_0
  let w_516_0 := witness.w_516_0
  let w_517_0 : mxxMat0 := MxxRuntime.matrixMul w_511_0 w_516_0
  let w_518_0 := witness.w_518_0
  let w_519_0 := witness.w_519_0
  let w_520_0 := witness.w_520_0
  let w_521_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_519_0 w_520_0
  let w_522_0 : mxxMat1 := MxxRuntime.matrixAdd w_518_0 w_521_0
  let w_523_0 := witness.w_523_0
  let w_524_0 := witness.w_524_0
  let w_525_0 := witness.w_525_0
  let w_525_concat_1 := witness.w_525_concat_1
  let w_526_0 := witness.w_526_0
  let w_527_0 := witness.w_527_0
  let w_528_0 := witness.w_528_0
  let w_529_0 := witness.w_529_0
  let w_530_0 := witness.w_530_0
  let w_531_0 : mxxMat0 := MxxRuntime.matrixMul w_511_0 w_530_0
  let w_532_0 := witness.w_532_0
  let w_533_0 := witness.w_533_0
  let w_534_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_533_0 w_520_0
  let w_535_0 : mxxMat1 := MxxRuntime.matrixAdd w_532_0 w_534_0
  let w_536_0 := witness.w_536_0
  let w_537_0 := witness.w_537_0
  let w_538_0 := witness.w_538_0
  let w_538_concat_1 := witness.w_538_concat_1
  let w_539_0 := witness.w_539_0
  let w_540_0 := witness.w_540_0
  let w_541_0 := witness.w_541_0
  let w_542_0 := witness.w_542_0
  let w_543_0 := witness.w_543_0
  let w_544_0 := witness.w_544_0
  let w_545_0 : mxxMat0 := MxxRuntime.matrixMul w_539_0 w_544_0
  let w_546_0 := witness.w_546_0
  let w_547_0 := witness.w_547_0
  let w_548_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_549_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_547_0 w_548_0
  let w_550_0 := witness.w_550_0
  let w_551_0 := witness.w_551_0
  let w_552_0 : mxxMat1 := MxxRuntime.matrixMul w_546_0 w_551_0
  let w_553_0 := witness.w_553_0
  let w_554_0 : mxxMat1 := MxxRuntime.matrixMul w_552_0 w_553_0
  let w_555_0 := witness.w_555_0
  let w_556_0 := witness.w_556_0
  let w_557_0 := witness.w_557_0
  let w_558_0 := witness.w_558_0
  let w_559_0 := witness.w_559_0
  let w_560_0 : mxxMat0 := MxxRuntime.matrixMul w_539_0 w_559_0
  let w_561_0 := witness.w_561_0
  let w_562_0 : mxxMat1 := MxxRuntime.matrixMul w_561_0 w_551_0
  let w_563_0 := witness.w_563_0
  let w_564_0 : mxxMat1 := MxxRuntime.matrixMul w_562_0 w_563_0
  let w_565_0 : mxxMat1 := MxxRuntime.matrixAdd w_554_0 w_564_0
  let w_566_0 := witness.w_566_0
  let w_567_0 : mxxMat1 := MxxRuntime.matrixMul w_552_0 w_566_0
  let w_568_0 := witness.w_568_0
  let w_569_0 : mxxMat1 := MxxRuntime.matrixMul w_562_0 w_568_0
  let w_570_0 : mxxMat1 := MxxRuntime.matrixAdd w_567_0 w_569_0
  let w_571_0 := witness.w_571_0
  let w_572_0 := witness.w_572_0
  let w_573_0 := witness.w_573_0
  let w_574_0 := witness.w_574_0
  let w_574_concat_1 := witness.w_574_concat_1
  let w_575_0 := witness.w_575_0
  let w_576_0 := witness.w_576_0
  let w_577_0 := witness.w_577_0
  let w_578_0 := witness.w_578_0
  let w_578_concat_1 := witness.w_578_concat_1
  let w_579_0 := witness.w_579_0
  let w_580_0 := witness.w_580_0
  let w_581_0 := witness.w_581_0
  let w_582_0 := witness.w_582_0
  let w_583_0 := witness.w_583_0
  let w_584_0 := witness.w_584_0
  let w_585_0 : mxxMat0 := MxxRuntime.matrixMul w_579_0 w_584_0
  let w_586_0 := witness.w_586_0
  let w_587_0 := witness.w_587_0
  let w_588_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_589_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_587_0 w_588_0
  let w_590_0 := witness.w_590_0
  let w_591_0 := witness.w_591_0
  let w_592_0 : mxxMat1 := MxxRuntime.matrixMul w_586_0 w_591_0
  let w_593_0 := witness.w_593_0
  let w_594_0 : mxxMat1 := MxxRuntime.matrixMul w_592_0 w_593_0
  let w_595_0 := witness.w_595_0
  let w_596_0 := witness.w_596_0
  let w_597_0 := witness.w_597_0
  let w_598_0 := witness.w_598_0
  let w_599_0 := witness.w_599_0
  let w_600_0 : mxxMat0 := MxxRuntime.matrixMul w_579_0 w_599_0
  let w_601_0 := witness.w_601_0
  let w_602_0 : mxxMat1 := MxxRuntime.matrixMul w_601_0 w_591_0
  let w_603_0 := witness.w_603_0
  let w_604_0 : mxxMat1 := MxxRuntime.matrixMul w_602_0 w_603_0
  let w_605_0 : mxxMat1 := MxxRuntime.matrixAdd w_594_0 w_604_0
  let w_606_0 := witness.w_606_0
  let w_607_0 : mxxMat1 := MxxRuntime.matrixMul w_592_0 w_606_0
  let w_608_0 := witness.w_608_0
  let w_609_0 : mxxMat1 := MxxRuntime.matrixMul w_602_0 w_608_0
  let w_610_0 : mxxMat1 := MxxRuntime.matrixAdd w_607_0 w_609_0
  let w_611_0 := witness.w_611_0
  let w_612_0 := witness.w_612_0
  let w_613_0 := witness.w_613_0
  let w_614_0 := witness.w_614_0
  let w_614_concat_1 := witness.w_614_concat_1
  let w_615_0 := witness.w_615_0
  let w_616_0 := witness.w_616_0
  let w_617_0 := witness.w_617_0
  let w_618_0 := witness.w_618_0
  let w_618_concat_1 := witness.w_618_concat_1
  let w_619_0 := witness.w_619_0
  let w_620_0 := witness.w_620_0
  let w_621_0 := witness.w_621_0
  let w_622_0 := witness.w_622_0
  let w_623_0 := witness.w_623_0
  let w_624_0 := witness.w_624_0
  let w_625_0 : mxxMat0 := MxxRuntime.matrixMul w_619_0 w_624_0
  let w_626_0 := witness.w_626_0
  let w_627_0 := witness.w_627_0
  let w_628_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_629_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_627_0 w_628_0
  let w_630_0 := witness.w_630_0
  let w_631_0 := witness.w_631_0
  let w_632_0 : mxxMat1 := MxxRuntime.matrixMul w_626_0 w_631_0
  let w_633_0 := witness.w_633_0
  let w_634_0 : mxxMat1 := MxxRuntime.matrixMul w_632_0 w_633_0
  let w_635_0 := witness.w_635_0
  let w_636_0 := witness.w_636_0
  let w_637_0 := witness.w_637_0
  let w_638_0 := witness.w_638_0
  let w_639_0 := witness.w_639_0
  let w_640_0 : mxxMat0 := MxxRuntime.matrixMul w_619_0 w_639_0
  let w_641_0 := witness.w_641_0
  let w_642_0 : mxxMat1 := MxxRuntime.matrixMul w_641_0 w_631_0
  let w_643_0 := witness.w_643_0
  let w_644_0 : mxxMat1 := MxxRuntime.matrixMul w_642_0 w_643_0
  let w_645_0 : mxxMat1 := MxxRuntime.matrixAdd w_634_0 w_644_0
  let w_646_0 := witness.w_646_0
  let w_647_0 : mxxMat1 := MxxRuntime.matrixMul w_632_0 w_646_0
  let w_648_0 := witness.w_648_0
  let w_649_0 : mxxMat1 := MxxRuntime.matrixMul w_642_0 w_648_0
  let w_650_0 : mxxMat1 := MxxRuntime.matrixAdd w_647_0 w_649_0
  let w_651_0 := witness.w_651_0
  let w_652_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_653_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_651_0 w_652_0
  let w_654_0 := witness.w_654_0
  let w_655_0 := witness.w_655_0
  let w_656_0 : mxxMat1 := MxxRuntime.matrixMul w_54_1 w_655_0
  let w_657_0 := witness.w_657_0
  let w_658_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_659_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_657_0 w_658_0
  let w_660_0 := witness.w_660_0
  let w_661_0 := witness.w_661_0
  let w_662_0 : mxxMat1 := MxxRuntime.matrixMul w_656_0 w_661_0
  let w_663_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_661_0
  let w_664_0 : mxxMat1 := (0 : mxxMat1)
  let w_666_0 := witness.w_666_0
  let w_668_0 := witness.w_668_0
  let w_670_0 := witness.w_670_0
  let w_672_0 := witness.w_672_0
  let w_675_0 := witness.w_675_0
  let w_675_1 := witness.w_675_1
  let w_675_2 := witness.w_675_2
  let w_675_3 := witness.w_675_3
  let w_675_4 := witness.w_675_4
  let w_675_5 := witness.w_675_5
  let w_675_6 := witness.w_675_6
  let w_675_7 := witness.w_675_7
  let w_675_8 := witness.w_675_8
  let w_675_9 := witness.w_675_9
  let w_676_0 := witness.w_676_0
  let w_677_0 : Int := (MxxIR.selectInt (0) [(3)])
  let w_678_0 := witness.w_678_0
  let w_679_0 := witness.w_679_0
  let w_680_0 := witness.w_680_0
  let w_680_concat_1 := witness.w_680_concat_1
  let w_681_0 := witness.w_681_0
  let w_682_0 := witness.w_682_0
  let w_683_0 := witness.w_683_0
  let w_684_0 := witness.w_684_0
  let w_684_concat_1 := witness.w_684_concat_1
  let w_685_0 := witness.w_685_0
  let w_686_0 := witness.w_686_0
  let w_687_0 := witness.w_687_0
  let w_688_0 := witness.w_688_0
  let w_689_0 := witness.w_689_0
  let w_689_concat_1 := witness.w_689_concat_1
  let w_690_0 := witness.w_690_0
  let w_691_0 := witness.w_691_0
  let w_692_0 := witness.w_692_0
  let w_693_0 := witness.w_693_0
  let w_693_concat_1 := witness.w_693_concat_1
  let w_694_0 := witness.w_694_0
  let w_695_0 := witness.w_695_0
  let w_696_0 := witness.w_696_0
  let w_697_0 := witness.w_697_0
  let w_698_0 : mxxMat0 := MxxRuntime.matrixMul w_694_0 w_697_0
  let w_699_0 := witness.w_699_0
  let w_700_0 := witness.w_700_0
  let w_701_0 := witness.w_701_0
  let w_702_0 := witness.w_702_0
  let w_703_0 := witness.w_703_0
  let w_704_0 : mxxMat0 := MxxRuntime.matrixMul w_685_0 w_703_0
  let w_705_0 := witness.w_705_0
  let w_706_0 : mxxMat1 := MxxRuntime.matrixAdd w_675_3 w_705_0
  let w_707_0 := witness.w_707_0
  let w_708_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_709_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_707_0 w_708_0
  let w_710_0 := witness.w_710_0
  let w_711_0 := witness.w_711_0
  let w_712_0 : mxxMat1 := MxxRuntime.matrixMul w_706_0 w_711_0
  let w_713_0 := witness.w_713_0
  let w_714_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_715_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_713_0 w_714_0
  let w_716_0 := witness.w_716_0
  let w_717_0 := witness.w_717_0
  let w_718_0 : mxxMat1 := MxxRuntime.matrixMul w_712_0 w_717_0
  let w_719_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_717_0
  let w_720_0 : mxxMat1 := (0 : mxxMat1)
  let w_722_0 := witness.w_722_0
  let w_724_0 := witness.w_724_0
  let w_726_0 := witness.w_726_0
  let w_728_0 := witness.w_728_0
  let w_731_0 := witness.w_731_0
  let w_731_1 := witness.w_731_1
  let w_731_2 := witness.w_731_2
  let w_731_3 := witness.w_731_3
  let w_731_4 := witness.w_731_4
  let w_731_5 := witness.w_731_5
  let w_731_6 := witness.w_731_6
  let w_731_7 := witness.w_731_7
  let w_731_8 := witness.w_731_8
  let w_731_9 := witness.w_731_9
  let w_732_0 := witness.w_732_0
  let w_733_0 : Int := (MxxIR.selectInt (0) [(5)])
  let w_734_0 := witness.w_734_0
  let w_735_0 := witness.w_735_0
  let w_736_0 := witness.w_736_0
  let w_736_concat_1 := witness.w_736_concat_1
  let w_737_0 := witness.w_737_0
  let w_738_0 := witness.w_738_0
  let w_739_0 := witness.w_739_0
  let w_740_0 := witness.w_740_0
  let w_740_concat_1 := witness.w_740_concat_1
  let w_741_0 := witness.w_741_0
  let w_742_0 := witness.w_742_0
  let w_743_0 := witness.w_743_0
  let w_744_0 := witness.w_744_0
  let w_745_0 := witness.w_745_0
  let w_745_concat_1 := witness.w_745_concat_1
  let w_746_0 := witness.w_746_0
  let w_747_0 := witness.w_747_0
  let w_748_0 := witness.w_748_0
  let w_749_0 := witness.w_749_0
  let w_749_concat_1 := witness.w_749_concat_1
  let w_750_0 := witness.w_750_0
  let w_751_0 := witness.w_751_0
  let w_752_0 := witness.w_752_0
  let w_753_0 := witness.w_753_0
  let w_754_0 : mxxMat0 := MxxRuntime.matrixMul w_750_0 w_753_0
  let w_755_0 := witness.w_755_0
  let w_756_0 := witness.w_756_0
  let w_757_0 := witness.w_757_0
  let w_758_0 := witness.w_758_0
  let w_759_0 := witness.w_759_0
  let w_760_0 : mxxMat0 := MxxRuntime.matrixMul w_741_0 w_759_0
  let w_761_0 := witness.w_761_0
  let w_762_0 : mxxMat1 := MxxRuntime.matrixAdd w_731_3 w_761_0
  let w_763_0 := witness.w_763_0
  let w_764_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_765_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_763_0 w_764_0
  let w_766_0 := witness.w_766_0
  let w_767_0 := witness.w_767_0
  let w_768_0 : mxxMat1 := MxxRuntime.matrixMul w_762_0 w_767_0
  let w_769_0 := witness.w_769_0
  let w_770_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_771_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_769_0 w_770_0
  let w_772_0 := witness.w_772_0
  let w_773_0 := witness.w_773_0
  let w_774_0 : mxxMat1 := MxxRuntime.matrixMul w_768_0 w_773_0
  let w_775_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_773_0
  let w_776_0 : mxxMat1 := (0 : mxxMat1)
  let w_778_0 := witness.w_778_0
  let w_780_0 := witness.w_780_0
  let w_782_0 := witness.w_782_0
  let w_784_0 := witness.w_784_0
  let w_787_0 := witness.w_787_0
  let w_787_1 := witness.w_787_1
  let w_787_2 := witness.w_787_2
  let w_787_3 := witness.w_787_3
  let w_787_4 := witness.w_787_4
  let w_787_5 := witness.w_787_5
  let w_787_6 := witness.w_787_6
  let w_787_7 := witness.w_787_7
  let w_787_8 := witness.w_787_8
  let w_787_9 := witness.w_787_9
  let w_788_0 := witness.w_788_0
  let w_789_0 : Int := (MxxIR.selectInt (0) [(7)])
  let w_790_0 := witness.w_790_0
  let w_791_0 := witness.w_791_0
  let w_792_0 := witness.w_792_0
  let w_792_concat_1 := witness.w_792_concat_1
  let w_793_0 := witness.w_793_0
  let w_794_0 := witness.w_794_0
  let w_795_0 := witness.w_795_0
  let w_796_0 := witness.w_796_0
  let w_796_concat_1 := witness.w_796_concat_1
  let w_797_0 := witness.w_797_0
  let w_798_0 := witness.w_798_0
  let w_799_0 := witness.w_799_0
  let w_800_0 := witness.w_800_0
  let w_801_0 := witness.w_801_0
  let w_801_concat_1 := witness.w_801_concat_1
  let w_802_0 := witness.w_802_0
  let w_803_0 := witness.w_803_0
  let w_804_0 := witness.w_804_0
  let w_805_0 := witness.w_805_0
  let w_805_concat_1 := witness.w_805_concat_1
  let w_806_0 := witness.w_806_0
  let w_807_0 := witness.w_807_0
  let w_808_0 := witness.w_808_0
  let w_809_0 := witness.w_809_0
  let w_810_0 : mxxMat0 := MxxRuntime.matrixMul w_806_0 w_809_0
  let w_811_0 := witness.w_811_0
  let w_812_0 := witness.w_812_0
  let w_813_0 := witness.w_813_0
  let w_814_0 := witness.w_814_0
  let w_815_0 := witness.w_815_0
  let w_816_0 : mxxMat0 := MxxRuntime.matrixMul w_797_0 w_815_0
  let w_817_0 := witness.w_817_0
  let w_818_0 : mxxMat1 := MxxRuntime.matrixAdd w_787_3 w_817_0
  let w_819_0 := witness.w_819_0
  let w_820_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_821_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_819_0 w_820_0
  let w_822_0 := witness.w_822_0
  let w_823_0 := witness.w_823_0
  let w_824_0 : mxxMat1 := MxxRuntime.matrixMul w_818_0 w_823_0
  let w_825_0 := witness.w_825_0
  let w_826_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_827_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_825_0 w_826_0
  let w_828_0 := witness.w_828_0
  let w_829_0 := witness.w_829_0
  let w_830_0 : mxxMat1 := MxxRuntime.matrixMul w_824_0 w_829_0
  let w_831_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_829_0
  let w_832_0 : mxxMat1 := (0 : mxxMat1)
  let w_834_0 := witness.w_834_0
  let w_836_0 := witness.w_836_0
  let w_838_0 := witness.w_838_0
  let w_840_0 := witness.w_840_0
  let w_843_0 := witness.w_843_0
  let w_843_1 := witness.w_843_1
  let w_843_2 := witness.w_843_2
  let w_843_3 := witness.w_843_3
  let w_843_4 := witness.w_843_4
  let w_843_5 := witness.w_843_5
  let w_843_6 := witness.w_843_6
  let w_843_7 := witness.w_843_7
  let w_843_8 := witness.w_843_8
  let w_843_9 := witness.w_843_9
  let w_844_0 := witness.w_844_0
  let w_845_0 : Int := (MxxIR.selectInt (0) [(9)])
  let w_846_0 := witness.w_846_0
  let w_847_0 := witness.w_847_0
  let w_848_0 := witness.w_848_0
  let w_848_concat_1 := witness.w_848_concat_1
  let w_849_0 := witness.w_849_0
  let w_850_0 := witness.w_850_0
  let w_851_0 := witness.w_851_0
  let w_852_0 := witness.w_852_0
  let w_852_concat_1 := witness.w_852_concat_1
  let w_853_0 := witness.w_853_0
  let w_854_0 := witness.w_854_0
  let w_855_0 := witness.w_855_0
  let w_856_0 := witness.w_856_0
  let w_857_0 := witness.w_857_0
  let w_857_concat_1 := witness.w_857_concat_1
  let w_858_0 := witness.w_858_0
  let w_859_0 := witness.w_859_0
  let w_860_0 := witness.w_860_0
  let w_861_0 := witness.w_861_0
  let w_861_concat_1 := witness.w_861_concat_1
  let w_862_0 := witness.w_862_0
  let w_863_0 := witness.w_863_0
  let w_864_0 := witness.w_864_0
  let w_865_0 := witness.w_865_0
  let w_866_0 : mxxMat0 := MxxRuntime.matrixMul w_862_0 w_865_0
  let w_867_0 := witness.w_867_0
  let w_868_0 := witness.w_868_0
  let w_869_0 := witness.w_869_0
  let w_870_0 := witness.w_870_0
  let w_871_0 := witness.w_871_0
  let w_872_0 : mxxMat0 := MxxRuntime.matrixMul w_853_0 w_871_0
  let w_873_0 := witness.w_873_0
  let w_874_0 : mxxMat1 := MxxRuntime.matrixAdd w_843_3 w_873_0
  let w_875_0 := witness.w_875_0
  let w_876_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_877_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_875_0 w_876_0
  let w_878_0 := witness.w_878_0
  let w_879_0 := witness.w_879_0
  let w_880_0 : mxxMat1 := MxxRuntime.matrixMul w_874_0 w_879_0
  let w_881_0 := witness.w_881_0
  let w_882_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_883_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_881_0 w_882_0
  let w_884_0 := witness.w_884_0
  let w_885_0 := witness.w_885_0
  let w_886_0 : mxxMat1 := MxxRuntime.matrixMul w_880_0 w_885_0
  let w_887_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_885_0
  let w_888_0 : mxxMat1 := (0 : mxxMat1)
  let w_890_0 := witness.w_890_0
  let w_892_0 := witness.w_892_0
  let w_894_0 := witness.w_894_0
  let w_896_0 := witness.w_896_0
  let w_899_0 := witness.w_899_0
  let w_899_1 := witness.w_899_1
  let w_899_2 := witness.w_899_2
  let w_899_3 := witness.w_899_3
  let w_899_4 := witness.w_899_4
  let w_899_5 := witness.w_899_5
  let w_899_6 := witness.w_899_6
  let w_899_7 := witness.w_899_7
  let w_899_8 := witness.w_899_8
  let w_899_9 := witness.w_899_9
  let w_900_0 := witness.w_900_0
  let w_901_0 : Int := (MxxIR.selectInt (0) [(11)])
  let w_902_0 := witness.w_902_0
  let w_903_0 := witness.w_903_0
  let w_904_0 := witness.w_904_0
  let w_904_concat_1 := witness.w_904_concat_1
  let w_905_0 := witness.w_905_0
  let w_906_0 := witness.w_906_0
  let w_907_0 := witness.w_907_0
  let w_908_0 := witness.w_908_0
  let w_908_concat_1 := witness.w_908_concat_1
  let w_909_0 := witness.w_909_0
  let w_910_0 := witness.w_910_0
  let w_911_0 := witness.w_911_0
  let w_912_0 := witness.w_912_0
  let w_913_0 := witness.w_913_0
  let w_913_concat_1 := witness.w_913_concat_1
  let w_914_0 := witness.w_914_0
  let w_915_0 := witness.w_915_0
  let w_916_0 := witness.w_916_0
  let w_917_0 := witness.w_917_0
  let w_917_concat_1 := witness.w_917_concat_1
  let w_918_0 := witness.w_918_0
  let w_919_0 := witness.w_919_0
  let w_920_0 := witness.w_920_0
  let w_921_0 := witness.w_921_0
  let w_922_0 : mxxMat0 := MxxRuntime.matrixMul w_918_0 w_921_0
  let w_923_0 := witness.w_923_0
  let w_924_0 := witness.w_924_0
  let w_925_0 := witness.w_925_0
  let w_926_0 := witness.w_926_0
  let w_927_0 := witness.w_927_0
  let w_928_0 : mxxMat0 := MxxRuntime.matrixMul w_909_0 w_927_0
  let w_929_0 := witness.w_929_0
  let w_930_0 : mxxMat1 := MxxRuntime.matrixAdd w_899_3 w_929_0
  let w_931_0 := witness.w_931_0
  let w_932_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_933_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_931_0 w_932_0
  let w_934_0 := witness.w_934_0
  let w_935_0 := witness.w_935_0
  let w_936_0 : mxxMat1 := MxxRuntime.matrixMul w_930_0 w_935_0
  let w_937_0 := witness.w_937_0
  let w_938_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_939_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_937_0 w_938_0
  let w_940_0 := witness.w_940_0
  let w_941_0 := witness.w_941_0
  let w_942_0 : mxxMat1 := MxxRuntime.matrixMul w_936_0 w_941_0
  let w_943_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_941_0
  let w_944_0 : mxxMat1 := (0 : mxxMat1)
  let w_946_0 := witness.w_946_0
  let w_948_0 := witness.w_948_0
  let w_950_0 := witness.w_950_0
  let w_952_0 := witness.w_952_0
  let w_955_0 := witness.w_955_0
  let w_955_1 := witness.w_955_1
  let w_955_2 := witness.w_955_2
  let w_955_3 := witness.w_955_3
  let w_955_4 := witness.w_955_4
  let w_955_5 := witness.w_955_5
  let w_955_6 := witness.w_955_6
  let w_955_7 := witness.w_955_7
  let w_955_8 := witness.w_955_8
  let w_955_9 := witness.w_955_9
  let w_956_0 := witness.w_956_0
  let w_957_0 : Int := (MxxIR.selectInt (0) [(13)])
  let w_958_0 := witness.w_958_0
  let w_959_0 := witness.w_959_0
  let w_960_0 := witness.w_960_0
  let w_960_concat_1 := witness.w_960_concat_1
  let w_961_0 := witness.w_961_0
  let w_962_0 := witness.w_962_0
  let w_963_0 := witness.w_963_0
  let w_964_0 := witness.w_964_0
  let w_964_concat_1 := witness.w_964_concat_1
  let w_965_0 := witness.w_965_0
  let w_966_0 := witness.w_966_0
  let w_967_0 := witness.w_967_0
  let w_968_0 := witness.w_968_0
  let w_969_0 := witness.w_969_0
  let w_969_concat_1 := witness.w_969_concat_1
  let w_970_0 := witness.w_970_0
  let w_971_0 := witness.w_971_0
  let w_972_0 := witness.w_972_0
  let w_973_0 := witness.w_973_0
  let w_973_concat_1 := witness.w_973_concat_1
  let w_974_0 := witness.w_974_0
  let w_975_0 := witness.w_975_0
  let w_976_0 := witness.w_976_0
  let w_977_0 := witness.w_977_0
  let w_978_0 : mxxMat0 := MxxRuntime.matrixMul w_974_0 w_977_0
  let w_979_0 := witness.w_979_0
  let w_980_0 := witness.w_980_0
  let w_981_0 := witness.w_981_0
  let w_982_0 := witness.w_982_0
  let w_983_0 := witness.w_983_0
  let w_984_0 : mxxMat0 := MxxRuntime.matrixMul w_965_0 w_983_0
  let w_985_0 := witness.w_985_0
  let w_986_0 : mxxMat1 := MxxRuntime.matrixAdd w_955_3 w_985_0
  let w_987_0 := witness.w_987_0
  let w_988_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_989_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_987_0 w_988_0
  let w_990_0 := witness.w_990_0
  let w_991_0 := witness.w_991_0
  let w_992_0 : mxxMat1 := MxxRuntime.matrixMul w_986_0 w_991_0
  let w_993_0 := witness.w_993_0
  let w_994_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_995_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_993_0 w_994_0
  let w_996_0 := witness.w_996_0
  let w_997_0 := witness.w_997_0
  let w_998_0 : mxxMat1 := MxxRuntime.matrixMul w_992_0 w_997_0
  let w_999_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_997_0
  let w_1000_0 : mxxMat1 := (0 : mxxMat1)
  let w_1002_0 := witness.w_1002_0
  let w_1004_0 := witness.w_1004_0
  let w_1006_0 := witness.w_1006_0
  let w_1008_0 := witness.w_1008_0
  let w_1011_0 := witness.w_1011_0
  let w_1011_1 := witness.w_1011_1
  let w_1011_2 := witness.w_1011_2
  let w_1011_3 := witness.w_1011_3
  let w_1011_4 := witness.w_1011_4
  let w_1011_5 := witness.w_1011_5
  let w_1011_6 := witness.w_1011_6
  let w_1011_7 := witness.w_1011_7
  let w_1011_8 := witness.w_1011_8
  let w_1011_9 := witness.w_1011_9
  let w_1012_0 := witness.w_1012_0
  let w_1013_0 : Int := (MxxIR.selectInt (0) [(15)])
  let w_1014_0 := witness.w_1014_0
  let w_1015_0 := witness.w_1015_0
  let w_1016_0 := witness.w_1016_0
  let w_1016_concat_1 := witness.w_1016_concat_1
  let w_1017_0 := witness.w_1017_0
  let w_1018_0 := witness.w_1018_0
  let w_1019_0 := witness.w_1019_0
  let w_1020_0 := witness.w_1020_0
  let w_1020_concat_1 := witness.w_1020_concat_1
  let w_1021_0 := witness.w_1021_0
  let w_1022_0 := witness.w_1022_0
  let w_1023_0 := witness.w_1023_0
  let w_1024_0 := witness.w_1024_0
  let w_1025_0 := witness.w_1025_0
  let w_1025_concat_1 := witness.w_1025_concat_1
  let w_1026_0 := witness.w_1026_0
  let w_1027_0 := witness.w_1027_0
  let w_1028_0 := witness.w_1028_0
  let w_1029_0 := witness.w_1029_0
  let w_1029_concat_1 := witness.w_1029_concat_1
  let w_1030_0 := witness.w_1030_0
  let w_1031_0 := witness.w_1031_0
  let w_1032_0 := witness.w_1032_0
  let w_1033_0 := witness.w_1033_0
  let w_1034_0 : mxxMat0 := MxxRuntime.matrixMul w_1030_0 w_1033_0
  let w_1035_0 := witness.w_1035_0
  let w_1036_0 := witness.w_1036_0
  let w_1037_0 := witness.w_1037_0
  let w_1038_0 := witness.w_1038_0
  let w_1039_0 := witness.w_1039_0
  let w_1040_0 : mxxMat0 := MxxRuntime.matrixMul w_1021_0 w_1039_0
  let w_1041_0 := witness.w_1041_0
  let w_1042_0 : mxxMat1 := MxxRuntime.matrixAdd w_1011_3 w_1041_0
  let w_1043_0 := witness.w_1043_0
  let w_1044_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_1045_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1043_0 w_1044_0
  let w_1046_0 := witness.w_1046_0
  let w_1047_0 := witness.w_1047_0
  let w_1048_0 : mxxMat1 := MxxRuntime.matrixMul w_1042_0 w_1047_0
  let w_1049_0 := witness.w_1049_0
  let w_1050_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat5)
  let w_1051_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1049_0 w_1050_0
  let w_1052_0 := witness.w_1052_0
  let w_1053_0 := witness.w_1053_0
  let w_1054_0 : mxxMat1 := MxxRuntime.matrixMul w_1048_0 w_1053_0
  let w_1055_0 : mxxMat1 := MxxRuntime.matrixMul w_68_0 w_1053_0
  let w_1056_0 : mxxMat1 := (0 : mxxMat1)
  let w_1057_0 := witness.w_1057_0
  let w_1058_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_1067_0 := witness.w_1067_0
  let w_1068_0 := witness.w_1068_0
  let w_1071_0 := witness.w_1071_0
  let w_1072_0 := witness.w_1072_0
  let w_1076_0 := witness.w_1076_0
  let w_1076_1 := witness.w_1076_1
  let w_1076_2 := witness.w_1076_2
  let w_1076_3 := witness.w_1076_3
  let w_1076_4 := witness.w_1076_4
  let w_1076_5 := witness.w_1076_5
  let w_1077_0 := witness.w_1077_0
  let w_1077_1 := witness.w_1077_1
  let w_1077_2 := witness.w_1077_2
  let w_1077_3 := witness.w_1077_3
  let w_1077_4 := witness.w_1077_4
  let w_1077_5 := witness.w_1077_5
  let w_1080_0 := witness.w_1080_0
  let w_1080_1 := witness.w_1080_1
  let w_1080_2 := witness.w_1080_2
  let w_1080_3 := witness.w_1080_3
  let w_1080_4 := witness.w_1080_4
  let w_1080_5 := witness.w_1080_5
  let w_1081_0 := witness.w_1081_0
  let w_1081_1 := witness.w_1081_1
  let w_1081_2 := witness.w_1081_2
  let w_1081_3 := witness.w_1081_3
  let w_1081_4 := witness.w_1081_4
  let w_1081_5 := witness.w_1081_5
  let w_1082_0 := witness.w_1082_0
  let w_1082_1 := witness.w_1082_1
  let w_1082_2 := witness.w_1082_2
  let w_1082_3 := witness.w_1082_3
  let w_1082_4 := witness.w_1082_4
  let w_1082_5 := witness.w_1082_5
  let w_1083_0 := witness.w_1083_0
  let w_1083_1 := witness.w_1083_1
  let w_1083_2 := witness.w_1083_2
  let w_1083_3 := witness.w_1083_3
  let w_1083_4 := witness.w_1083_4
  let w_1083_5 := witness.w_1083_5
  let w_1084_0 := witness.w_1084_0
  let w_1085_0 := witness.w_1085_0
  let w_1086_0 : mxxMat1 := MxxRuntime.matrixAdd w_1084_0 w_1085_0
  let w_1087_0 := witness.w_1087_0
  let w_1088_0 := witness.w_1088_0
  let w_1089_0 : mxxMat1 := MxxRuntime.matrixAdd w_1087_0 w_1088_0
  let w_1090_0 := witness.w_1090_0
  let w_1091_0 := witness.w_1091_0
  let w_1092_0 : mxxMat1 := MxxRuntime.matrixAdd w_1090_0 w_1091_0
  let w_1093_0 := witness.w_1093_0
  let w_1094_0 := witness.w_1094_0
  let w_1095_0 : mxxMat1 := MxxRuntime.matrixAdd w_1093_0 w_1094_0
  let w_1096_0 := witness.w_1096_0
  let w_1097_0 := witness.w_1097_0
  let w_1098_0 : mxxMat1 := MxxRuntime.matrixAdd w_1096_0 w_1097_0
  let w_1099_0 := witness.w_1099_0
  let w_1100_0 := witness.w_1100_0
  let w_1101_0 : mxxMat1 := MxxRuntime.matrixAdd w_1099_0 w_1100_0
  let w_1102_0 := witness.w_1102_0
  let w_1102_1 := witness.w_1102_1
  let w_1102_2 := witness.w_1102_2
  let w_1102_3 := witness.w_1102_3
  let w_1102_4 := witness.w_1102_4
  let w_1102_5 := witness.w_1102_5
  let w_1103_0 := witness.w_1103_0
  let w_1104_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_1105_0 := witness.w_1105_0
  let w_1106_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_1107_0 := witness.w_1107_0
  let w_1108_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_1109_0 := witness.w_1109_0
  let w_1110_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_1111_0 := witness.w_1111_0
  let w_1112_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_1113_0 := witness.w_1113_0
  let w_1114_0 := witness.w_1114_0
  let w_1114_1 := witness.w_1114_1
  let w_1114_2 := witness.w_1114_2
  let w_1114_3 := witness.w_1114_3
  let w_1114_4 := witness.w_1114_4
  let w_1114_5 := witness.w_1114_5
  let w_1115_0 : mxxMat9 := (MxxRuntime.unitRow (0) : mxxMat9)
  let w_1122_0 := witness.w_1122_0
  let w_1123_0 := witness.w_1123_0
  let w_1125_0 := witness.w_1125_0
  let w_1126_0 := witness.w_1126_0
  let w_1130_0 := witness.w_1130_0
  let w_1130_1 := witness.w_1130_1
  let w_1130_2 := witness.w_1130_2
  let w_1130_3 := witness.w_1130_3
  let w_1130_4 := witness.w_1130_4
  let w_1130_5 := witness.w_1130_5
  let w_1131_0 := witness.w_1131_0
  let w_1131_1 := witness.w_1131_1
  let w_1131_2 := witness.w_1131_2
  let w_1131_3 := witness.w_1131_3
  let w_1131_4 := witness.w_1131_4
  let w_1131_5 := witness.w_1131_5
  let w_1134_0 := witness.w_1134_0
  let w_1134_1 := witness.w_1134_1
  let w_1134_2 := witness.w_1134_2
  let w_1134_3 := witness.w_1134_3
  let w_1134_4 := witness.w_1134_4
  let w_1134_5 := witness.w_1134_5
  let w_1135_0 := witness.w_1135_0
  let w_1135_1 := witness.w_1135_1
  let w_1135_2 := witness.w_1135_2
  let w_1135_3 := witness.w_1135_3
  let w_1135_4 := witness.w_1135_4
  let w_1135_5 := witness.w_1135_5
  let w_1136_0 := witness.w_1136_0
  let w_1136_1 := witness.w_1136_1
  let w_1136_2 := witness.w_1136_2
  let w_1136_3 := witness.w_1136_3
  let w_1136_4 := witness.w_1136_4
  let w_1136_5 := witness.w_1136_5
  let w_1137_0 := witness.w_1137_0
  let w_1137_1 := witness.w_1137_1
  let w_1137_2 := witness.w_1137_2
  let w_1137_3 := witness.w_1137_3
  let w_1137_4 := witness.w_1137_4
  let w_1137_5 := witness.w_1137_5
  let w_1138_0 := witness.w_1138_0
  let w_1139_0 := witness.w_1139_0
  let w_1140_0 : mxxMat1 := MxxRuntime.matrixAdd w_1138_0 w_1139_0
  let w_1141_0 := witness.w_1141_0
  let w_1142_0 := witness.w_1142_0
  let w_1143_0 : mxxMat1 := MxxRuntime.matrixAdd w_1141_0 w_1142_0
  let w_1144_0 := witness.w_1144_0
  let w_1145_0 := witness.w_1145_0
  let w_1146_0 : mxxMat1 := MxxRuntime.matrixAdd w_1144_0 w_1145_0
  let w_1147_0 := witness.w_1147_0
  let w_1148_0 := witness.w_1148_0
  let w_1149_0 : mxxMat1 := MxxRuntime.matrixAdd w_1147_0 w_1148_0
  let w_1150_0 := witness.w_1150_0
  let w_1151_0 := witness.w_1151_0
  let w_1152_0 : mxxMat1 := MxxRuntime.matrixAdd w_1150_0 w_1151_0
  let w_1153_0 := witness.w_1153_0
  let w_1154_0 := witness.w_1154_0
  let w_1155_0 : mxxMat1 := MxxRuntime.matrixAdd w_1153_0 w_1154_0
  let w_1156_0 := witness.w_1156_0
  let w_1156_1 := witness.w_1156_1
  let w_1156_2 := witness.w_1156_2
  let w_1156_3 := witness.w_1156_3
  let w_1156_4 := witness.w_1156_4
  let w_1156_5 := witness.w_1156_5
  let w_1157_0 := witness.w_1157_0
  let w_1158_0 : mxxMat9 := (MxxRuntime.unitRow (0) : mxxMat9)
  let w_1159_0 := witness.w_1159_0
  let w_1160_0 : mxxMat9 := (MxxRuntime.unitRow (0) : mxxMat9)
  let w_1161_0 := witness.w_1161_0
  let w_1162_0 : mxxMat9 := (MxxRuntime.unitRow (0) : mxxMat9)
  let w_1163_0 := witness.w_1163_0
  let w_1164_0 : mxxMat9 := (MxxRuntime.unitRow (0) : mxxMat9)
  let w_1165_0 := witness.w_1165_0
  let w_1166_0 : mxxMat9 := (MxxRuntime.unitRow (0) : mxxMat9)
  let w_1167_0 := witness.w_1167_0
  let w_1168_0 := witness.w_1168_0
  let w_1168_1 := witness.w_1168_1
  let w_1168_2 := witness.w_1168_2
  let w_1168_3 := witness.w_1168_3
  let w_1168_4 := witness.w_1168_4
  let w_1168_5 := witness.w_1168_5
  let w_1169_0 : mxxMat5 := (MxxRuntime.matrixPolynomial [(mxxNat8)] : mxxMat5)
  let w_1170_0 : mxxMat1 := MxxRuntime.matrixMul w_18_0 w_234_0
  let w_1171_0 : mxxMat1 := MxxRuntime.matrixMul w_22_0 w_234_0
  let w_1172_0 := witness.w_1172_0
  let w_1173_0 : mxxMat1 := MxxRuntime.matrixAdd w_315_4 w_1172_0
  let w_1174_0 : mxxMat1 := MxxRuntime.matrixMul w_1173_0 w_351_0
  let w_1175_0 := witness.w_1175_0
  let w_1176_0 : mxxMat1 := MxxRuntime.matrixAdd w_315_5 w_1175_0
  let w_1177_0 : mxxMat1 := MxxRuntime.matrixMul w_1176_0 w_351_0
  let w_1178_0 := witness.w_1178_0
  let w_1179_0 : mxxMat1 := MxxRuntime.matrixMul w_1178_0 w_388_0
  let w_1180_0 : mxxMat1 := MxxRuntime.matrixMul w_1179_0 w_390_0
  let w_1181_0 := witness.w_1181_0
  let w_1182_0 : mxxMat1 := MxxRuntime.matrixMul w_1181_0 w_388_0
  let w_1183_0 : mxxMat1 := MxxRuntime.matrixMul w_1182_0 w_402_0
  let w_1184_0 : mxxMat1 := MxxRuntime.matrixAdd w_1180_0 w_1183_0
  let w_1185_0 : mxxMat1 := MxxRuntime.matrixMul w_1179_0 w_405_0
  let w_1186_0 : mxxMat1 := MxxRuntime.matrixMul w_1182_0 w_410_0
  let w_1187_0 : mxxMat1 := MxxRuntime.matrixAdd w_1185_0 w_1186_0
  let w_1188_0 := witness.w_1188_0
  let w_1189_0 : mxxMat1 := MxxRuntime.matrixMul w_1188_0 w_477_0
  let w_1190_0 : mxxMat1 := MxxRuntime.matrixMul w_1189_0 w_479_0
  let w_1191_0 := witness.w_1191_0
  let w_1192_0 : mxxMat1 := MxxRuntime.matrixMul w_1191_0 w_477_0
  let w_1193_0 : mxxMat1 := MxxRuntime.matrixMul w_1192_0 w_495_0
  let w_1194_0 : mxxMat1 := MxxRuntime.matrixAdd w_1190_0 w_1193_0
  let w_1195_0 : mxxMat1 := MxxRuntime.matrixMul w_1189_0 w_498_0
  let w_1196_0 : mxxMat1 := MxxRuntime.matrixMul w_1192_0 w_500_0
  let w_1197_0 : mxxMat1 := MxxRuntime.matrixAdd w_1195_0 w_1196_0
  let w_1198_0 := witness.w_1198_0
  let w_1199_0 : mxxMat1 := MxxRuntime.matrixMul w_1198_0 w_551_0
  let w_1200_0 : mxxMat1 := MxxRuntime.matrixMul w_1199_0 w_553_0
  let w_1201_0 := witness.w_1201_0
  let w_1202_0 : mxxMat1 := MxxRuntime.matrixMul w_1201_0 w_551_0
  let w_1203_0 : mxxMat1 := MxxRuntime.matrixMul w_1202_0 w_563_0
  let w_1204_0 : mxxMat1 := MxxRuntime.matrixAdd w_1200_0 w_1203_0
  let w_1205_0 : mxxMat1 := MxxRuntime.matrixMul w_1199_0 w_566_0
  let w_1206_0 : mxxMat1 := MxxRuntime.matrixMul w_1202_0 w_568_0
  let w_1207_0 : mxxMat1 := MxxRuntime.matrixAdd w_1205_0 w_1206_0
  let w_1208_0 := witness.w_1208_0
  let w_1209_0 : mxxMat1 := MxxRuntime.matrixMul w_1208_0 w_591_0
  let w_1210_0 : mxxMat1 := MxxRuntime.matrixMul w_1209_0 w_593_0
  let w_1211_0 := witness.w_1211_0
  let w_1212_0 : mxxMat1 := MxxRuntime.matrixMul w_1211_0 w_591_0
  let w_1213_0 : mxxMat1 := MxxRuntime.matrixMul w_1212_0 w_603_0
  let w_1214_0 : mxxMat1 := MxxRuntime.matrixAdd w_1210_0 w_1213_0
  let w_1215_0 : mxxMat1 := MxxRuntime.matrixMul w_1209_0 w_606_0
  let w_1216_0 : mxxMat1 := MxxRuntime.matrixMul w_1212_0 w_608_0
  let w_1217_0 : mxxMat1 := MxxRuntime.matrixAdd w_1215_0 w_1216_0
  let w_1218_0 := witness.w_1218_0
  let w_1219_0 : mxxMat1 := MxxRuntime.matrixMul w_1218_0 w_631_0
  let w_1220_0 : mxxMat1 := MxxRuntime.matrixMul w_1219_0 w_633_0
  let w_1221_0 := witness.w_1221_0
  let w_1222_0 : mxxMat1 := MxxRuntime.matrixMul w_1221_0 w_631_0
  let w_1223_0 : mxxMat1 := MxxRuntime.matrixMul w_1222_0 w_643_0
  let w_1224_0 : mxxMat1 := MxxRuntime.matrixAdd w_1220_0 w_1223_0
  let w_1225_0 : mxxMat1 := MxxRuntime.matrixMul w_1219_0 w_646_0
  let w_1226_0 : mxxMat1 := MxxRuntime.matrixMul w_1222_0 w_648_0
  let w_1227_0 : mxxMat1 := MxxRuntime.matrixAdd w_1225_0 w_1226_0
  let w_1228_0 : mxxMat1 := MxxRuntime.matrixMul w_54_3 w_655_0
  let w_1229_0 : mxxMat1 := MxxRuntime.matrixMul w_1228_0 w_661_0
  let w_1230_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_661_0
  let w_1231_0 := witness.w_1231_0
  let w_1232_0 : mxxMat1 := MxxRuntime.matrixAdd w_675_5 w_1231_0
  let w_1233_0 : mxxMat1 := MxxRuntime.matrixMul w_1232_0 w_711_0
  let w_1234_0 : mxxMat1 := MxxRuntime.matrixMul w_1233_0 w_717_0
  let w_1235_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_717_0
  let w_1236_0 := witness.w_1236_0
  let w_1237_0 : mxxMat1 := MxxRuntime.matrixAdd w_731_5 w_1236_0
  let w_1238_0 : mxxMat1 := MxxRuntime.matrixMul w_1237_0 w_767_0
  let w_1239_0 : mxxMat1 := MxxRuntime.matrixMul w_1238_0 w_773_0
  let w_1240_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_773_0
  let w_1241_0 := witness.w_1241_0
  let w_1242_0 : mxxMat1 := MxxRuntime.matrixAdd w_787_5 w_1241_0
  let w_1243_0 : mxxMat1 := MxxRuntime.matrixMul w_1242_0 w_823_0
  let w_1244_0 : mxxMat1 := MxxRuntime.matrixMul w_1243_0 w_829_0
  let w_1245_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_829_0
  let w_1246_0 := witness.w_1246_0
  let w_1247_0 : mxxMat1 := MxxRuntime.matrixAdd w_843_5 w_1246_0
  let w_1248_0 : mxxMat1 := MxxRuntime.matrixMul w_1247_0 w_879_0
  let w_1249_0 : mxxMat1 := MxxRuntime.matrixMul w_1248_0 w_885_0
  let w_1250_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_885_0
  let w_1251_0 := witness.w_1251_0
  let w_1252_0 : mxxMat1 := MxxRuntime.matrixAdd w_899_5 w_1251_0
  let w_1253_0 : mxxMat1 := MxxRuntime.matrixMul w_1252_0 w_935_0
  let w_1254_0 : mxxMat1 := MxxRuntime.matrixMul w_1253_0 w_941_0
  let w_1255_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_941_0
  let w_1256_0 := witness.w_1256_0
  let w_1257_0 : mxxMat1 := MxxRuntime.matrixAdd w_955_5 w_1256_0
  let w_1258_0 : mxxMat1 := MxxRuntime.matrixMul w_1257_0 w_991_0
  let w_1259_0 : mxxMat1 := MxxRuntime.matrixMul w_1258_0 w_997_0
  let w_1260_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_997_0
  let w_1261_0 := witness.w_1261_0
  let w_1262_0 : mxxMat1 := MxxRuntime.matrixAdd w_1011_5 w_1261_0
  let w_1263_0 : mxxMat1 := MxxRuntime.matrixMul w_1262_0 w_1047_0
  let w_1264_0 : mxxMat1 := MxxRuntime.matrixMul w_1263_0 w_1053_0
  let w_1265_0 : mxxMat1 := MxxRuntime.matrixMul w_408_0 w_1053_0
  let w_1266_0 := witness.w_1266_0
  let w_1267_0 := witness.w_1267_0
  let w_1268_0 := witness.w_1268_0
  let w_1269_0 := witness.w_1269_0
  let w_1270_0 := witness.w_1270_0
  let w_1271_0 := witness.w_1271_0
  let w_1272_0 := witness.w_1272_0
  let w_1273_0 := witness.w_1273_0
  let w_1274_0 : mxxMat1 := MxxRuntime.matrixAdd w_1272_0 w_1273_0
  let w_1275_0 : mxxMat1 := MxxRuntime.matrixAdd w_315_0 w_1274_0
  let w_1276_0 := witness.w_1276_0
  let w_1277_0 := witness.w_1277_0
  let w_1278_0 : mxxMat1 := MxxRuntime.matrixAdd w_1276_0 w_1277_0
  let w_1279_0 : mxxMat1 := MxxRuntime.matrixAdd w_315_1 w_1278_0
  let w_1280_0 := witness.w_1280_0
  let w_1281_0 := witness.w_1281_0
  let w_1282_0 := witness.w_1282_0
  let w_1283_0 : mxxMat0 := MxxRuntime.matrixMul w_974_0 w_1282_0
  let w_1284_0 := witness.w_1284_0
  let w_1285_0 := witness.w_1285_0
  let w_1286_0 := witness.w_1286_0
  let w_1287_0 := witness.w_1287_0
  let w_1288_0 := witness.w_1288_0
  let w_1289_0 : mxxMat0 := MxxRuntime.matrixMul w_965_0 w_1288_0
  let w_1290_0 := witness.w_1290_0
  let w_1291_0 : mxxMat1 := MxxRuntime.matrixAdd w_955_2 w_1290_0
  let w_1292_0 := witness.w_1292_0
  let w_1293_0 : mxxMat1 := MxxRuntime.matrixAdd w_955_4 w_1292_0
  let w_1294_0 := witness.w_1294_0
  let w_1295_0 := witness.w_1295_0
  let w_1296_0 : mxxMat1 := MxxRuntime.matrixAdd w_1294_0 w_1295_0
  let w_1297_0 : mxxMat1 := MxxRuntime.matrixAdd w_955_0 w_1296_0
  let w_1298_0 := witness.w_1298_0
  let w_1299_0 := witness.w_1299_0
  let w_1300_0 : mxxMat1 := MxxRuntime.matrixAdd w_1298_0 w_1299_0
  let w_1301_0 : mxxMat1 := MxxRuntime.matrixAdd w_955_1 w_1300_0
  let w_1302_0 := witness.w_1302_0
  let w_1303_0 := witness.w_1303_0
  let w_1304_0 := witness.w_1304_0
  let w_1305_0 : mxxMat0 := MxxRuntime.matrixMul w_1030_0 w_1304_0
  let w_1306_0 := witness.w_1306_0
  let w_1307_0 := witness.w_1307_0
  let w_1308_0 := witness.w_1308_0
  let w_1309_0 := witness.w_1309_0
  let w_1310_0 := witness.w_1310_0
  let w_1311_0 : mxxMat0 := MxxRuntime.matrixMul w_1021_0 w_1310_0
  let w_1312_0 := witness.w_1312_0
  let w_1313_0 : mxxMat1 := MxxRuntime.matrixAdd w_1011_2 w_1312_0
  let w_1314_0 := witness.w_1314_0
  let w_1315_0 : mxxMat1 := MxxRuntime.matrixAdd w_1011_4 w_1314_0
  let w_1316_0 := witness.w_1316_0
  let w_1317_0 := witness.w_1317_0
  let w_1318_0 : mxxMat1 := MxxRuntime.matrixAdd w_1316_0 w_1317_0
  let w_1319_0 : mxxMat1 := MxxRuntime.matrixAdd w_1011_0 w_1318_0
  let w_1320_0 := witness.w_1320_0
  let w_1321_0 := witness.w_1321_0
  let w_1322_0 : mxxMat1 := MxxRuntime.matrixAdd w_1320_0 w_1321_0
  let w_1323_0 : mxxMat1 := MxxRuntime.matrixAdd w_1011_1 w_1322_0
  let w_1324_0 := witness.w_1324_0
  let w_1325_0 := witness.w_1325_0
  let w_1326_0 := witness.w_1326_0
  let w_1327_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1325_0 w_1326_0
  let w_1328_0 : mxxMat1 := MxxRuntime.matrixAdd w_1324_0 w_1327_0
  let w_1329_0 := witness.w_1329_0
  let w_1330_0 := witness.w_1330_0
  let w_1331_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1330_0 w_1326_0
  let w_1332_0 : mxxMat1 := MxxRuntime.matrixAdd w_1329_0 w_1331_0
  let w_1333_0 := witness.w_1333_0
  let w_1334_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1326_0 w_1333_0
  let w_1335_0 := witness.w_1335_0
  let w_1336_0 := witness.w_1336_0
  let w_1337_0 := witness.w_1337_0
  let w_1338_0 := witness.w_1338_0
  let w_1339_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1337_0 w_1338_0
  let w_1340_0 : mxxMat1 := MxxRuntime.matrixAdd w_1336_0 w_1339_0
  let w_1341_0 := witness.w_1341_0
  let w_1342_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_431_0 w_1341_0
  let w_1343_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1340_0 w_1342_0
  let w_1344_0 : mxxMat1 := MxxRuntime.matrixAdd w_1335_0 w_1343_0
  let w_1345_0 := witness.w_1345_0
  let w_1346_0 := witness.w_1346_0
  let w_1347_0 := witness.w_1347_0
  let w_1348_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1347_0 w_1338_0
  let w_1349_0 : mxxMat1 := MxxRuntime.matrixAdd w_1346_0 w_1348_0
  let w_1350_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1349_0 w_1342_0
  let w_1351_0 : mxxMat1 := MxxRuntime.matrixAdd w_1345_0 w_1350_0
  let w_1352_0 := witness.w_1352_0
  let w_1353_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1338_0 w_1352_0
  let w_1354_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1342_0 w_1353_0
  let w_1355_0 := witness.w_1355_0
  let w_1356_0 := witness.w_1356_0
  let w_1357_0 := witness.w_1357_0
  let w_1358_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_520_0 w_1357_0
  let w_1359_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1356_0 w_1358_0
  let w_1360_0 : mxxMat1 := MxxRuntime.matrixAdd w_1355_0 w_1359_0
  let w_1361_0 := witness.w_1361_0
  let w_1362_0 := witness.w_1362_0
  let w_1363_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1362_0 w_1358_0
  let w_1364_0 : mxxMat1 := MxxRuntime.matrixAdd w_1361_0 w_1363_0
  let w_1365_0 := witness.w_1365_0
  let w_1366_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1358_0 w_1365_0
  let w_1367_0 := witness.w_1367_0
  let w_1368_0 := witness.w_1368_0
  let w_1369_0 := witness.w_1369_0
  let w_1370_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1368_0 w_1369_0
  let w_1371_0 : mxxMat1 := MxxRuntime.matrixAdd w_1367_0 w_1370_0
  let w_1372_0 := witness.w_1372_0
  let w_1373_0 := witness.w_1373_0
  let w_1374_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1373_0 w_1369_0
  let w_1375_0 : mxxMat1 := MxxRuntime.matrixAdd w_1372_0 w_1374_0
  let w_1376_0 := witness.w_1376_0
  let w_1377_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1369_0 w_1376_0
  let w_1378_0 := witness.w_1378_0
  let w_1379_0 := witness.w_1379_0
  let w_1380_0 := witness.w_1380_0
  let w_1381_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1379_0 w_1380_0
  let w_1382_0 : mxxMat1 := MxxRuntime.matrixAdd w_1378_0 w_1381_0
  let w_1383_0 := witness.w_1383_0
  let w_1384_0 := witness.w_1384_0
  let w_1385_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_1384_0 w_1380_0
  let w_1386_0 : mxxMat1 := MxxRuntime.matrixAdd w_1383_0 w_1385_0
  let w_1387_0 := witness.w_1387_0
  let w_1388_0 : mxxMat5 := MxxRuntime.matrixMulScalarLeft w_1380_0 w_1387_0
  let w_1389_0 := witness.w_1389_0
  let w_1390_0 := witness.w_1390_0
  let w_1391_0 := witness.w_1391_0
  let w_1392_0 := witness.w_1392_0
  let w_1393_0 := witness.w_1393_0
  let w_1394_0 := witness.w_1394_0
  let w_1395_0 := witness.w_1395_0
  let w_1396_0 := witness.w_1396_0
  let w_1397_0 : mxxMat0 := MxxRuntime.matrixMul w_694_0 w_1396_0
  let w_1398_0 := witness.w_1398_0
  let w_1399_0 := witness.w_1399_0
  let w_1400_0 := witness.w_1400_0
  let w_1401_0 := witness.w_1401_0
  let w_1402_0 := witness.w_1402_0
  let w_1403_0 : mxxMat0 := MxxRuntime.matrixMul w_685_0 w_1402_0
  let w_1404_0 := witness.w_1404_0
  let w_1405_0 : mxxMat1 := MxxRuntime.matrixAdd w_675_2 w_1404_0
  let w_1406_0 := witness.w_1406_0
  let w_1407_0 : mxxMat1 := MxxRuntime.matrixAdd w_675_4 w_1406_0
  let w_1408_0 := witness.w_1408_0
  let w_1409_0 := witness.w_1409_0
  let w_1410_0 : mxxMat1 := MxxRuntime.matrixAdd w_1408_0 w_1409_0
  let w_1411_0 : mxxMat1 := MxxRuntime.matrixAdd w_675_0 w_1410_0
  let w_1412_0 := witness.w_1412_0
  let w_1413_0 := witness.w_1413_0
  let w_1414_0 : mxxMat1 := MxxRuntime.matrixAdd w_1412_0 w_1413_0
  let w_1415_0 : mxxMat1 := MxxRuntime.matrixAdd w_675_1 w_1414_0
  let w_1416_0 := witness.w_1416_0
  let w_1417_0 := witness.w_1417_0
  let w_1418_0 := witness.w_1418_0
  let w_1419_0 : mxxMat0 := MxxRuntime.matrixMul w_750_0 w_1418_0
  let w_1420_0 := witness.w_1420_0
  let w_1421_0 := witness.w_1421_0
  let w_1422_0 := witness.w_1422_0
  let w_1423_0 := witness.w_1423_0
  let w_1424_0 := witness.w_1424_0
  let w_1425_0 : mxxMat0 := MxxRuntime.matrixMul w_741_0 w_1424_0
  let w_1426_0 := witness.w_1426_0
  let w_1427_0 : mxxMat1 := MxxRuntime.matrixAdd w_731_2 w_1426_0
  let w_1428_0 := witness.w_1428_0
  let w_1429_0 : mxxMat1 := MxxRuntime.matrixAdd w_731_4 w_1428_0
  let w_1430_0 := witness.w_1430_0
  let w_1431_0 := witness.w_1431_0
  let w_1432_0 : mxxMat1 := MxxRuntime.matrixAdd w_1430_0 w_1431_0
  let w_1433_0 : mxxMat1 := MxxRuntime.matrixAdd w_731_0 w_1432_0
  let w_1434_0 := witness.w_1434_0
  let w_1435_0 := witness.w_1435_0
  let w_1436_0 : mxxMat1 := MxxRuntime.matrixAdd w_1434_0 w_1435_0
  let w_1437_0 : mxxMat1 := MxxRuntime.matrixAdd w_731_1 w_1436_0
  let w_1438_0 := witness.w_1438_0
  let w_1439_0 := witness.w_1439_0
  let w_1440_0 := witness.w_1440_0
  let w_1441_0 : mxxMat0 := MxxRuntime.matrixMul w_806_0 w_1440_0
  let w_1442_0 := witness.w_1442_0
  let w_1443_0 := witness.w_1443_0
  let w_1444_0 := witness.w_1444_0
  let w_1445_0 := witness.w_1445_0
  let w_1446_0 := witness.w_1446_0
  let w_1447_0 : mxxMat0 := MxxRuntime.matrixMul w_797_0 w_1446_0
  let w_1448_0 := witness.w_1448_0
  let w_1449_0 : mxxMat1 := MxxRuntime.matrixAdd w_787_2 w_1448_0
  let w_1450_0 := witness.w_1450_0
  let w_1451_0 : mxxMat1 := MxxRuntime.matrixAdd w_787_4 w_1450_0
  let w_1452_0 := witness.w_1452_0
  let w_1453_0 := witness.w_1453_0
  let w_1454_0 : mxxMat1 := MxxRuntime.matrixAdd w_1452_0 w_1453_0
  let w_1455_0 : mxxMat1 := MxxRuntime.matrixAdd w_787_0 w_1454_0
  let w_1456_0 := witness.w_1456_0
  let w_1457_0 := witness.w_1457_0
  let w_1458_0 : mxxMat1 := MxxRuntime.matrixAdd w_1456_0 w_1457_0
  let w_1459_0 : mxxMat1 := MxxRuntime.matrixAdd w_787_1 w_1458_0
  let w_1460_0 := witness.w_1460_0
  let w_1461_0 := witness.w_1461_0
  let w_1462_0 := witness.w_1462_0
  let w_1463_0 : mxxMat0 := MxxRuntime.matrixMul w_862_0 w_1462_0
  let w_1464_0 := witness.w_1464_0
  let w_1465_0 := witness.w_1465_0
  let w_1466_0 := witness.w_1466_0
  let w_1467_0 := witness.w_1467_0
  let w_1468_0 := witness.w_1468_0
  let w_1469_0 : mxxMat0 := MxxRuntime.matrixMul w_853_0 w_1468_0
  let w_1470_0 := witness.w_1470_0
  let w_1471_0 : mxxMat1 := MxxRuntime.matrixAdd w_843_2 w_1470_0
  let w_1472_0 := witness.w_1472_0
  let w_1473_0 : mxxMat1 := MxxRuntime.matrixAdd w_843_4 w_1472_0
  let w_1474_0 := witness.w_1474_0
  let w_1475_0 := witness.w_1475_0
  let w_1476_0 : mxxMat1 := MxxRuntime.matrixAdd w_1474_0 w_1475_0
  let w_1477_0 : mxxMat1 := MxxRuntime.matrixAdd w_843_0 w_1476_0
  let w_1478_0 := witness.w_1478_0
  let w_1479_0 := witness.w_1479_0
  let w_1480_0 : mxxMat1 := MxxRuntime.matrixAdd w_1478_0 w_1479_0
  let w_1481_0 : mxxMat1 := MxxRuntime.matrixAdd w_843_1 w_1480_0
  let w_1482_0 := witness.w_1482_0
  let w_1483_0 := witness.w_1483_0
  let w_1484_0 := witness.w_1484_0
  let w_1485_0 : mxxMat0 := MxxRuntime.matrixMul w_918_0 w_1484_0
  let w_1486_0 := witness.w_1486_0
  let w_1487_0 := witness.w_1487_0
  let w_1488_0 := witness.w_1488_0
  let w_1489_0 := witness.w_1489_0
  let w_1490_0 := witness.w_1490_0
  let w_1491_0 : mxxMat0 := MxxRuntime.matrixMul w_909_0 w_1490_0
  let w_1492_0 := witness.w_1492_0
  let w_1493_0 : mxxMat1 := MxxRuntime.matrixAdd w_899_2 w_1492_0
  let w_1494_0 := witness.w_1494_0
  let w_1495_0 : mxxMat1 := MxxRuntime.matrixAdd w_899_4 w_1494_0
  let w_1496_0 := witness.w_1496_0
  let w_1497_0 := witness.w_1497_0
  let w_1498_0 : mxxMat1 := MxxRuntime.matrixAdd w_1496_0 w_1497_0
  let w_1499_0 : mxxMat1 := MxxRuntime.matrixAdd w_899_0 w_1498_0
  let w_1500_0 := witness.w_1500_0
  let w_1501_0 := witness.w_1501_0
  let w_1502_0 : mxxMat1 := MxxRuntime.matrixAdd w_1500_0 w_1501_0
  let w_1503_0 : mxxMat1 := MxxRuntime.matrixAdd w_899_1 w_1502_0
  generatedRoot.constraints_0 backend hashModel params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 w_25_0 w_26_0 w_29_0 w_30_0 w_17_0 w_18_0 w_21_0 w_22_0 w_27_0 w_27_1 w_27_2 w_27_3 w_27_4 w_27_5 w_28_0 w_28_1 w_28_2 w_28_3 w_28_4 w_28_5 w_31_0 w_31_1 w_31_2 w_31_3 w_31_4 w_31_5 w_32_0 w_32_1 w_32_2 w_32_3 w_32_4 w_32_5 w_33_0 w_33_1 w_33_2 w_33_3 w_33_4 w_33_5 w_34_0 w_34_1 w_34_2 w_34_3 w_34_4 w_34_5 w_35_0 w_35_1 w_35_2 w_35_3 w_35_4 w_35_5 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 ∧
    generatedRoot.constraints_1 backend hashModel w_5_0 w_69_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 w_65_0 w_66_0 w_67_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 ∧
    generatedRoot.constraints_2 hashModel w_69_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 ∧
    generatedRoot.constraints_3 hashModel w_69_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 ∧
    generatedRoot.constraints_4 hashModel w_69_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 ∧
    generatedRoot.constraints_5 hashModel w_69_0 w_214_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 ∧
    generatedRoot.constraints_6 backend hashModel w_5_0 w_245_0 w_247_0 w_249_0 w_251_0 w_255_0 w_260_0 w_73_0 w_77_0 w_81_0 w_85_0 w_89_0 w_93_0 w_97_0 w_101_0 w_105_0 w_109_0 w_113_0 w_117_0 w_121_0 w_125_0 w_129_0 w_133_0 w_138_0 w_143_0 w_148_0 w_153_0 w_158_0 w_163_0 w_168_0 w_173_0 w_178_0 w_183_0 w_188_0 w_193_0 w_198_0 w_203_0 w_208_0 w_213_0 w_216_0 w_218_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_227_0 w_228_0 w_229_0 w_230_0 w_232_0 w_233_0 w_234_0 w_237_0 w_239_0 w_240_0 w_241_0 w_246_0 w_248_0 w_250_0 w_252_0 w_256_0 w_257_0 w_258_0 w_259_0 w_259_concat_1 w_261_0 w_262_0 w_263_0 w_264_concat_1 ∧
    generatedRoot.constraints_7 backend hashModel w_5_0 w_274_0 w_259_0 w_263_0 w_264_0 w_264_concat_1 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_280_concat_1 w_281_0 ∧
    generatedRoot.constraints_8 backend hashModel w_5_0 w_24_0 w_296_0 w_280_0 w_281_0 w_282_0 w_283_0 w_284_0 w_285_0 w_286_0 w_287_0 w_290_0 w_291_0 w_292_0 w_293_0 w_293_concat_1 w_294_0 w_295_0 w_297_0 w_298_0 w_299_0 ∧
    generatedRoot.constraints_9 backend hashModel w_24_0 w_306_0 w_299_0 w_300_0 w_301_0 w_302_0 w_303_0 w_304_0 w_305_0 w_307_0 w_308_0 w_309_0 w_310_0 ∧
    generatedRoot.constraints_10 backend hashModel params w_5_0 w_245_0 w_247_0 w_249_0 w_251_0 w_253_0 w_254_0 w_244_0 w_246_0 w_248_0 w_250_0 w_252_0 w_302_0 w_303_0 w_304_0 w_311_0 w_312_0 w_313_0 w_314_0 w_315_0 w_315_1 w_315_2 w_315_3 w_315_4 w_315_5 w_315_6 w_315_7 w_315_8 w_315_9 w_316_0 w_317_0 w_318_0 w_319_0 w_320_0 w_320_concat_1 w_321_0 w_322_0 w_323_0 w_324_0 w_324_concat_1 w_325_0 ∧
    generatedRoot.constraints_11 backend w_302_0 w_303_0 w_304_0 w_312_0 w_313_0 w_314_0 w_315_6 w_315_8 w_326_0 w_327_0 w_328_0 w_329_0 w_329_concat_1 w_330_0 w_331_0 w_332_0 w_333_0 w_333_concat_1 w_334_0 w_335_0 w_336_0 w_337_0 w_338_0 w_339_0 ∧
    generatedRoot.constraints_12 backend w_229_0 w_315_7 w_315_9 w_338_0 w_339_0 w_340_0 w_341_0 w_342_0 w_343_0 w_344_0 w_345_0 w_347_0 w_349_0 w_350_0 w_351_0 w_353_0 w_354_0 w_355_0 w_356_0 w_357_0 ∧
    generatedRoot.constraints_13 backend hashModel w_5_0 w_366_0 w_371_0 w_356_0 w_357_0 w_358_0 w_359_0 w_360_0 w_361_0 w_362_0 w_363_0 w_367_0 w_368_0 w_369_0 w_370_0 w_370_concat_1 w_372_0 w_373_0 w_374_0 w_375_0 w_375_concat_1 w_376_0 w_377_0 w_378_0 w_379_0 w_380_0 w_381_0 ∧
    generatedRoot.constraints_14 backend hashModel w_5_0 w_366_0 w_371_0 w_55_0 w_242_0 w_243_0 w_382_0 w_383_0 w_384_0 w_386_0 w_387_0 w_388_0 w_390_0 w_392_0 w_393_0 w_394_0 w_395_0 w_396_0 w_397_0 w_398_0 w_401_0 w_402_0 w_405_0 w_407_0 w_409_0 w_410_0 w_413_0 w_414_0 w_415_0 w_416_0 w_416_concat_1 w_417_0 ∧
    generatedRoot.constraints_15 backend hashModel w_5_0 w_366_0 w_430_0 w_416_0 w_417_0 w_418_0 w_419_0 w_420_0 w_420_concat_1 w_421_0 w_422_0 w_423_0 w_424_0 w_425_0 w_426_0 w_427_0 w_428_0 w_429_0 w_431_0 w_434_0 ∧
    generatedRoot.constraints_16 backend hashModel w_5_0 w_371_0 w_427_0 w_433_0 w_434_0 w_435_0 w_436_0 w_436_concat_1 w_437_0 w_438_0 w_439_0 w_440_0 w_441_0 w_442_0 w_443_0 w_444_0 w_446_0 w_447_0 w_448_0 w_449_0 w_449_concat_1 ∧
    generatedRoot.constraints_17 backend hashModel w_5_0 w_366_0 w_371_0 w_436_0 w_449_0 w_450_0 w_451_0 w_452_0 w_453_0 w_454_0 w_454_concat_1 w_455_0 w_456_0 w_457_0 w_458_0 w_458_concat_1 w_459_0 w_460_0 w_461_0 w_462_0 w_463_0 w_464_0 w_465_0 w_466_0 w_467_0 w_468_0 ∧
    generatedRoot.constraints_18 backend hashModel w_5_0 w_55_0 w_242_0 w_468_0 w_469_0 w_470_0 w_471_0 w_472_0 w_473_0 w_475_0 w_476_0 w_477_0 w_479_0 w_481_0 w_482_0 w_483_0 w_484_0 w_485_0 w_486_0 w_487_0 w_488_0 ∧
    generatedRoot.constraints_19 backend hashModel w_5_0 w_366_0 w_371_0 w_243_0 w_401_0 w_409_0 w_487_0 w_488_0 w_489_0 w_490_0 w_491_0 w_492_0 w_493_0 w_495_0 w_498_0 w_500_0 w_503_0 w_504_0 w_505_0 w_506_0 w_506_concat_1 w_507_0 w_508_0 w_509_0 w_510_0 w_510_concat_1 w_511_0 w_512_0 w_513_0 w_514_0 w_515_0 w_516_0 ∧
    generatedRoot.constraints_20 backend hashModel w_5_0 w_366_0 w_430_0 w_517_0 w_518_0 w_519_0 w_520_0 w_522_0 w_523_0 w_524_0 w_525_0 w_525_concat_1 w_526_0 w_527_0 w_528_0 w_529_0 w_530_0 ∧
    generatedRoot.constraints_21 backend hashModel w_5_0 w_371_0 w_525_0 w_531_0 w_532_0 w_533_0 w_535_0 w_536_0 w_537_0 w_538_0 w_538_concat_1 w_539_0 w_540_0 w_541_0 w_542_0 w_543_0 w_544_0 w_545_0 w_546_0 ∧
    generatedRoot.constraints_22 backend hashModel w_5_0 w_366_0 w_371_0 w_55_0 w_242_0 w_243_0 w_401_0 w_409_0 w_547_0 w_549_0 w_550_0 w_551_0 w_553_0 w_555_0 w_556_0 w_557_0 w_558_0 w_559_0 w_560_0 w_561_0 w_563_0 w_566_0 w_568_0 w_571_0 w_572_0 w_573_0 w_574_0 w_574_concat_1 w_575_0 w_576_0 w_577_0 w_578_0 w_578_concat_1 w_579_0 w_580_0 w_581_0 ∧
    generatedRoot.constraints_23 backend hashModel w_5_0 w_366_0 w_55_0 w_242_0 w_243_0 w_401_0 w_409_0 w_580_0 w_581_0 w_582_0 w_583_0 w_584_0 w_585_0 w_586_0 w_587_0 w_589_0 w_590_0 w_591_0 w_593_0 w_595_0 w_596_0 w_597_0 w_598_0 w_599_0 w_600_0 w_601_0 w_603_0 w_606_0 w_608_0 w_611_0 w_612_0 w_613_0 ∧
    generatedRoot.constraints_24 backend hashModel w_5_0 w_371_0 w_55_0 w_242_0 w_611_0 w_612_0 w_613_0 w_614_0 w_614_concat_1 w_615_0 w_616_0 w_617_0 w_618_0 w_618_concat_1 w_619_0 w_620_0 w_621_0 w_622_0 w_623_0 w_624_0 w_625_0 w_626_0 w_627_0 w_629_0 w_630_0 w_631_0 w_633_0 w_635_0 w_636_0 w_637_0 w_638_0 w_639_0 ∧
    generatedRoot.constraints_25 backend hashModel params w_5_0 w_665_0 w_667_0 w_669_0 w_671_0 w_673_0 w_674_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_55_0 w_229_0 w_243_0 w_401_0 w_409_0 w_640_0 w_641_0 w_643_0 w_646_0 w_648_0 w_651_0 w_653_0 w_654_0 w_655_0 w_657_0 w_659_0 w_660_0 w_661_0 w_664_0 w_666_0 w_668_0 w_670_0 w_672_0 w_675_0 w_675_1 w_675_2 w_675_3 w_675_4 w_675_5 w_675_6 w_675_7 w_675_8 w_675_9 w_676_0 w_677_0 w_678_0 w_679_0 w_680_0 w_680_concat_1 w_681_0 w_682_0 w_683_0 w_684_0 w_684_concat_1 w_685_0 ∧
    generatedRoot.constraints_26 backend w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_675_7 w_675_9 w_686_0 w_687_0 w_688_0 w_689_0 w_689_concat_1 w_690_0 w_691_0 w_692_0 w_693_0 w_693_concat_1 w_694_0 w_695_0 w_696_0 w_697_0 ∧
    generatedRoot.constraints_27 backend hashModel params w_5_0 w_721_0 w_723_0 w_725_0 w_727_0 w_729_0 w_730_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_55_0 w_229_0 w_698_0 w_699_0 w_700_0 w_701_0 w_702_0 w_703_0 w_704_0 w_705_0 w_707_0 w_709_0 w_710_0 w_711_0 w_713_0 w_715_0 w_716_0 w_717_0 w_720_0 w_722_0 w_724_0 w_726_0 w_728_0 w_731_0 w_731_1 w_731_2 w_731_3 w_731_4 w_731_5 w_731_6 w_731_7 w_731_8 w_731_9 w_732_0 ∧
    generatedRoot.constraints_28 hashModel w_5_0 w_730_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_732_0 w_733_0 w_734_0 w_735_0 w_736_0 w_736_concat_1 w_737_0 w_738_0 w_739_0 w_740_0 w_740_concat_1 w_741_0 w_742_0 w_743_0 w_744_0 w_745_0 w_745_concat_1 w_746_0 w_747_0 w_748_0 w_749_concat_1 ∧
    generatedRoot.constraints_29 backend w_55_0 w_731_7 w_731_9 w_745_0 w_748_0 w_749_0 w_749_concat_1 w_750_0 w_751_0 w_752_0 w_753_0 w_754_0 w_755_0 w_756_0 w_757_0 w_758_0 w_759_0 w_760_0 w_761_0 w_763_0 w_765_0 w_766_0 w_767_0 ∧
    generatedRoot.constraints_30 backend hashModel params w_5_0 w_777_0 w_779_0 w_781_0 w_783_0 w_785_0 w_786_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_229_0 w_769_0 w_771_0 w_772_0 w_773_0 w_776_0 w_778_0 w_780_0 w_782_0 w_784_0 w_787_0 w_787_1 w_787_2 w_787_3 w_787_4 w_787_5 w_787_6 w_787_7 w_787_8 w_787_9 w_788_0 w_789_0 w_790_0 w_791_0 w_792_0 w_792_concat_1 w_793_0 w_794_0 w_795_0 w_796_0 w_796_concat_1 w_797_0 w_798_0 w_799_0 w_800_0 w_801_concat_1 ∧
    generatedRoot.constraints_31 backend w_54_1 w_54_3 w_54_5 w_787_7 w_787_9 w_800_0 w_801_0 w_801_concat_1 w_802_0 w_803_0 w_804_0 w_805_0 w_805_concat_1 w_806_0 w_807_0 w_808_0 w_809_0 w_810_0 w_811_0 w_812_0 w_813_0 w_814_0 ∧
    generatedRoot.constraints_32 backend hashModel params w_5_0 w_833_0 w_835_0 w_837_0 w_839_0 w_841_0 w_842_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_55_0 w_229_0 w_814_0 w_815_0 w_816_0 w_817_0 w_819_0 w_821_0 w_822_0 w_823_0 w_825_0 w_827_0 w_828_0 w_829_0 w_832_0 w_834_0 w_836_0 w_838_0 w_840_0 w_843_0 w_843_1 w_843_2 w_843_3 w_843_4 w_843_5 w_843_6 w_843_7 w_843_8 w_843_9 w_844_0 w_845_0 w_846_0 w_847_0 w_848_0 w_848_concat_1 w_849_0 w_850_0 w_851_0 w_852_0 w_852_concat_1 ∧
    generatedRoot.constraints_33 backend w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_843_7 w_843_9 w_848_0 w_852_0 w_853_0 w_854_0 w_855_0 w_856_0 w_857_0 w_857_concat_1 w_858_0 w_859_0 w_860_0 w_861_0 w_861_concat_1 w_862_0 w_863_0 w_864_0 w_865_0 ∧
    generatedRoot.constraints_34 backend hashModel params w_5_0 w_889_0 w_891_0 w_893_0 w_895_0 w_897_0 w_898_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_55_0 w_229_0 w_866_0 w_867_0 w_868_0 w_869_0 w_870_0 w_871_0 w_872_0 w_873_0 w_875_0 w_877_0 w_878_0 w_879_0 w_881_0 w_883_0 w_884_0 w_885_0 w_888_0 w_890_0 w_892_0 w_894_0 w_896_0 w_899_0 w_899_1 w_899_2 w_899_3 w_899_4 w_899_5 w_899_6 w_899_7 w_899_8 w_899_9 w_900_0 ∧
    generatedRoot.constraints_35 hashModel w_5_0 w_898_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_900_0 w_901_0 w_902_0 w_903_0 w_904_0 w_904_concat_1 w_905_0 w_906_0 w_907_0 w_908_0 w_908_concat_1 w_909_0 w_910_0 w_911_0 w_912_0 w_913_0 w_913_concat_1 w_914_0 w_915_0 w_916_0 ∧
    generatedRoot.constraints_36 backend w_55_0 w_899_7 w_899_9 w_913_0 w_914_0 w_915_0 w_916_0 w_917_0 w_917_concat_1 w_918_0 w_919_0 w_920_0 w_921_0 w_922_0 w_923_0 w_924_0 w_925_0 w_926_0 w_927_0 w_928_0 w_929_0 w_931_0 w_933_0 w_934_0 ∧
    generatedRoot.constraints_37 backend hashModel params w_5_0 w_945_0 w_947_0 w_949_0 w_951_0 w_953_0 w_954_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_229_0 w_934_0 w_935_0 w_937_0 w_939_0 w_940_0 w_941_0 w_944_0 w_946_0 w_948_0 w_950_0 w_952_0 w_955_0 w_955_1 w_955_2 w_955_3 w_955_4 w_955_5 w_955_6 w_955_7 w_955_8 w_955_9 w_956_0 w_957_0 w_958_0 w_959_0 w_960_0 w_960_concat_1 w_961_0 w_962_0 w_963_0 w_964_0 w_964_concat_1 w_965_0 w_966_0 w_967_0 w_968_0 ∧
    generatedRoot.constraints_38 backend w_54_1 w_54_3 w_54_5 w_955_7 w_955_9 w_966_0 w_967_0 w_968_0 w_969_0 w_969_concat_1 w_970_0 w_971_0 w_972_0 w_973_0 w_973_concat_1 w_974_0 w_975_0 w_976_0 w_977_0 w_978_0 w_979_0 w_980_0 w_981_0 ∧
    generatedRoot.constraints_39 backend hashModel params w_5_0 w_1001_0 w_1003_0 w_1005_0 w_1007_0 w_1009_0 w_1010_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_55_0 w_229_0 w_981_0 w_982_0 w_983_0 w_984_0 w_985_0 w_987_0 w_989_0 w_990_0 w_991_0 w_993_0 w_995_0 w_996_0 w_997_0 w_1000_0 w_1002_0 w_1004_0 w_1006_0 w_1008_0 w_1011_0 w_1011_1 w_1011_2 w_1011_3 w_1011_4 w_1011_5 w_1011_6 w_1011_7 w_1011_8 w_1011_9 w_1012_0 w_1013_0 w_1014_0 w_1015_0 w_1016_0 w_1016_concat_1 w_1017_0 w_1018_0 w_1019_0 w_1020_concat_1 ∧
    generatedRoot.constraints_40 backend w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_1011_7 w_1011_9 w_1016_0 w_1019_0 w_1020_0 w_1020_concat_1 w_1021_0 w_1022_0 w_1023_0 w_1024_0 w_1025_0 w_1025_concat_1 w_1026_0 w_1027_0 w_1028_0 w_1029_0 w_1029_concat_1 w_1030_0 w_1031_0 w_1032_0 w_1033_0 ∧
    generatedRoot.constraints_41 backend hashModel w_5_0 w_55_0 w_229_0 w_235_0 w_236_0 w_242_0 w_243_0 w_352_0 w_365_0 w_404_0 w_412_0 w_497_0 w_502_0 w_565_0 w_570_0 w_605_0 w_610_0 w_645_0 w_650_0 w_662_0 w_663_0 w_718_0 w_719_0 w_774_0 w_775_0 w_830_0 w_831_0 w_886_0 w_887_0 w_942_0 w_943_0 w_998_0 w_999_0 w_1034_0 w_1035_0 w_1036_0 w_1037_0 w_1038_0 w_1039_0 w_1040_0 w_1041_0 w_1043_0 w_1045_0 w_1046_0 w_1047_0 w_1049_0 w_1051_0 w_1052_0 w_1053_0 w_1054_0 w_1055_0 w_1056_0 w_1057_0 w_1067_0 w_1068_0 w_1071_0 ∧
    generatedRoot.constraints_42 backend hashModel params w_5_0 w_11_0 w_12_0 w_24_0 w_1059_0 w_1060_0 w_1061_0 w_1062_0 w_1063_0 w_1064_0 w_1065_0 w_1066_0 w_1069_0 w_1070_0 w_1073_0 w_1074_0 w_1075_0 w_1078_0 w_1079_0 w_1058_0 w_1067_0 w_1068_0 w_1071_0 w_1072_0 w_1076_0 w_1076_1 w_1076_2 w_1076_3 w_1076_4 w_1076_5 w_1077_0 w_1077_1 w_1077_2 w_1077_3 w_1077_4 w_1077_5 w_1080_0 w_1080_1 w_1080_2 w_1080_3 w_1080_4 w_1080_5 w_1081_0 w_1081_1 w_1081_2 w_1081_3 w_1081_4 w_1081_5 w_1082_0 w_1082_1 w_1082_2 w_1082_3 w_1082_4 w_1082_5 w_1083_0 w_1083_1 w_1083_2 w_1083_3 w_1083_4 w_1083_5 w_1084_0 w_1085_0 w_1086_0 w_1087_0 w_1088_0 w_1089_0 w_1090_0 w_1091_0 w_1092_0 w_1093_0 w_1094_0 w_1095_0 w_1096_0 w_1097_0 w_1098_0 w_1099_0 w_1100_0 w_1101_0 w_1102_0 w_1102_1 w_1102_2 w_1102_3 w_1102_4 w_1102_5 w_1103_0 w_1104_0 w_1105_0 ∧
    generatedRoot.constraints_43 backend hashModel params w_5_0 w_11_0 w_12_0 w_24_0 w_255_0 w_260_0 w_274_0 w_1116_0 w_1117_0 w_1118_0 w_1119_0 w_1120_0 w_1121_0 w_1124_0 w_1127_0 w_1128_0 w_1129_0 w_1132_0 w_1133_0 w_1083_0 w_1083_1 w_1083_2 w_1083_3 w_1083_4 w_1083_5 w_1102_2 w_1102_3 w_1102_4 w_1102_5 w_1103_0 w_1105_0 w_1106_0 w_1107_0 w_1108_0 w_1109_0 w_1110_0 w_1111_0 w_1112_0 w_1113_0 w_1114_0 w_1114_1 w_1114_2 w_1114_3 w_1114_4 w_1114_5 w_1122_0 w_1123_0 w_1125_0 w_1126_0 w_1130_0 w_1130_1 w_1130_2 w_1130_3 w_1130_4 w_1130_5 w_1131_0 w_1131_1 w_1131_2 w_1131_3 w_1131_4 w_1131_5 w_1134_0 w_1134_1 w_1134_2 w_1134_3 w_1134_4 w_1134_5 w_1135_0 w_1135_1 w_1135_2 w_1135_3 w_1135_4 w_1135_5 w_1136_0 w_1136_1 w_1136_2 w_1136_3 w_1136_4 w_1136_5 w_1137_0 w_1137_1 w_1137_2 w_1137_3 w_1137_4 w_1137_5 w_1138_0 ∧
    generatedRoot.constraints_44 backend hashModel params w_1115_0 w_1137_0 w_1137_1 w_1137_2 w_1137_3 w_1137_4 w_1137_5 w_1139_0 w_1140_0 w_1141_0 w_1142_0 w_1143_0 w_1144_0 w_1145_0 w_1146_0 w_1147_0 w_1148_0 w_1149_0 w_1150_0 w_1151_0 w_1152_0 w_1153_0 w_1154_0 w_1155_0 w_1156_0 w_1156_1 w_1156_2 w_1156_3 w_1156_4 w_1156_5 w_1157_0 w_1158_0 w_1159_0 w_1160_0 w_1161_0 w_1162_0 w_1163_0 w_1164_0 w_1165_0 w_1166_0 w_1167_0 w_1168_0 w_1168_1 w_1168_2 w_1168_3 w_1168_4 w_1168_5 ∧
    generatedRoot.constraints_45 w_344_0 w_362_0 w_382_0 w_397_0 w_1172_0 w_1175_0 w_1178_0 w_1181_0 ∧
    generatedRoot.constraints_46 w_471_0 w_492_0 w_545_0 w_560_0 w_585_0 w_1188_0 w_1191_0 w_1198_0 w_1201_0 w_1208_0 ∧
    generatedRoot.constraints_47 w_600_0 w_625_0 w_640_0 w_704_0 w_1211_0 w_1218_0 w_1221_0 w_1231_0 ∧
    generatedRoot.constraints_48 w_760_0 w_816_0 w_872_0 w_928_0 w_984_0 w_1236_0 w_1241_0 w_1246_0 w_1251_0 w_1256_0 ∧
    generatedRoot.constraints_49 backend hashModel params w_55_0 w_61_0 w_68_0 w_225_0 w_338_0 w_344_0 w_400_0 w_401_0 w_408_0 w_409_0 w_1040_0 w_1056_0 w_1057_0 w_1114_1 w_1114_3 w_1168_1 w_1168_3 w_1169_0 w_1170_0 w_1171_0 w_1174_0 w_1177_0 w_1184_0 w_1187_0 w_1194_0 w_1197_0 w_1204_0 w_1207_0 w_1214_0 w_1217_0 w_1224_0 w_1227_0 w_1229_0 w_1230_0 w_1234_0 w_1235_0 w_1239_0 w_1240_0 w_1244_0 w_1245_0 w_1249_0 w_1250_0 w_1254_0 w_1255_0 w_1259_0 w_1260_0 w_1261_0 w_1264_0 w_1265_0 w_1266_0 w_1267_0 w_1268_0 w_1269_0 w_1270_0 w_1271_0 w_1272_0 w_1273_0 ∧
    generatedRoot.constraints_50 backend w_356_0 w_362_0 w_955_6 w_955_8 w_1276_0 w_1277_0 w_1280_0 w_1281_0 w_1282_0 w_1283_0 w_1284_0 w_1285_0 w_1286_0 w_1287_0 w_1288_0 ∧
    generatedRoot.constraints_51 w_1283_0 w_1289_0 w_1290_0 w_1292_0 w_1294_0 w_1295_0 ∧
    generatedRoot.constraints_52 backend w_978_0 w_984_0 w_1011_6 w_1011_8 w_1298_0 w_1299_0 w_1302_0 w_1303_0 w_1304_0 w_1305_0 w_1306_0 w_1307_0 w_1308_0 w_1309_0 w_1310_0 ∧
    generatedRoot.constraints_53 w_1040_0 w_1305_0 w_1311_0 w_1312_0 w_1314_0 w_1316_0 w_1317_0 w_1320_0 ∧
    generatedRoot.constraints_54 w_366_0 w_371_0 w_430_0 w_382_0 w_397_0 w_471_0 w_1034_0 w_1321_0 w_1324_0 w_1325_0 w_1326_0 w_1329_0 w_1330_0 w_1333_0 w_1335_0 ∧
    generatedRoot.constraints_55 w_366_0 w_371_0 w_430_0 w_465_0 w_486_0 w_492_0 w_1336_0 w_1337_0 w_1338_0 w_1341_0 w_1345_0 w_1346_0 w_1347_0 w_1352_0 ∧
    generatedRoot.constraints_56 w_255_0 w_260_0 w_274_0 w_366_0 w_371_0 w_430_0 w_545_0 w_560_0 w_585_0 w_600_0 w_1355_0 w_1356_0 w_1357_0 w_1361_0 w_1362_0 w_1365_0 w_1367_0 w_1368_0 w_1369_0 w_1372_0 w_1373_0 w_1376_0 ∧
    generatedRoot.constraints_57 backend hashModel w_5_0 w_14_0 w_15_0 w_20_0 w_366_0 w_371_0 w_430_0 w_625_0 w_640_0 w_675_6 w_675_8 w_1378_0 w_1379_0 w_1380_0 w_1383_0 w_1384_0 w_1387_0 w_1389_0 w_1390_0 w_1391_0 w_1392_0 w_1393_0 w_1394_0 w_1395_0 w_1396_0 w_1397_0 w_1398_0 ∧
    generatedRoot.constraints_58 backend w_1397_0 w_1398_0 w_1399_0 w_1400_0 w_1401_0 w_1402_0 w_1403_0 w_1404_0 w_1406_0 w_1408_0 ∧
    generatedRoot.constraints_59 backend w_698_0 w_704_0 w_731_6 w_731_8 w_1397_0 w_1409_0 w_1412_0 w_1413_0 w_1416_0 w_1417_0 w_1418_0 w_1419_0 w_1420_0 ∧
    generatedRoot.constraints_60 backend w_1419_0 w_1420_0 w_1421_0 w_1422_0 w_1423_0 w_1424_0 w_1425_0 w_1426_0 w_1428_0 w_1430_0 ∧
    generatedRoot.constraints_61 backend w_754_0 w_760_0 w_787_6 w_787_8 w_1419_0 w_1431_0 w_1434_0 w_1435_0 w_1438_0 w_1439_0 w_1440_0 w_1441_0 w_1442_0 ∧
    generatedRoot.constraints_62 backend w_1441_0 w_1442_0 w_1443_0 w_1444_0 w_1445_0 w_1446_0 w_1447_0 w_1448_0 w_1450_0 w_1452_0 ∧
    generatedRoot.constraints_63 backend w_810_0 w_816_0 w_843_6 w_843_8 w_1441_0 w_1453_0 w_1456_0 w_1457_0 w_1460_0 w_1461_0 w_1462_0 w_1463_0 w_1464_0 w_1465_0 ∧
    generatedRoot.constraints_64 backend w_1463_0 w_1464_0 w_1465_0 w_1466_0 w_1467_0 w_1468_0 w_1469_0 w_1470_0 w_1472_0 w_1474_0 w_1475_0 ∧
    generatedRoot.constraints_65 backend w_866_0 w_872_0 w_899_6 w_899_8 w_1478_0 w_1479_0 w_1482_0 w_1483_0 w_1484_0 w_1485_0 w_1486_0 w_1487_0 w_1488_0 w_1489_0 ∧
    generatedRoot.constraints_66 w_1485_0 w_1489_0 w_1490_0 w_1491_0 w_1492_0 w_1494_0 w_1496_0 w_1497_0 ∧
    generatedRoot.constraints_67 w_19_0 w_23_0 w_17_0 w_18_0 w_21_0 w_22_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_62_0 w_346_0 w_364_0 w_383_0 w_398_0 w_407_0 w_472_0 w_493_0 w_546_0 w_561_0 w_586_0 w_601_0 w_626_0 w_641_0 w_706_0 w_762_0 w_818_0 w_874_0 w_922_0 w_928_0 w_930_0 w_986_0 w_1042_0 w_1114_0 w_1114_1 w_1114_2 w_1114_3 w_1114_4 w_1114_5 w_1168_0 w_1168_1 w_1168_2 w_1168_3 w_1168_4 w_1168_5 w_1173_0 w_1176_0 w_1178_0 w_1181_0 w_1188_0 w_1191_0 w_1198_0 w_1201_0 w_1208_0 w_1211_0 w_1218_0 w_1221_0 w_1232_0 w_1237_0 w_1242_0 w_1247_0 w_1252_0 w_1257_0 w_1262_0 w_1267_0 w_1268_0 w_1269_0 w_1270_0 w_1271_0 w_1275_0 w_1279_0 w_1291_0 w_1293_0 w_1297_0 w_1301_0 w_1313_0 w_1315_0 w_1319_0 w_1323_0 w_1328_0 w_1332_0 w_1334_0 w_1344_0 w_1351_0 w_1354_0 w_1360_0 w_1364_0 w_1366_0 w_1371_0 w_1375_0 w_1377_0 w_1382_0 w_1386_0 w_1388_0 w_1389_0 w_1390_0 w_1391_0 w_1392_0 w_1393_0 w_1405_0 w_1407_0 w_1411_0 w_1415_0 w_1427_0 w_1429_0 w_1433_0 w_1437_0 w_1449_0 w_1451_0 w_1455_0 w_1459_0 w_1471_0 w_1473_0 w_1477_0 w_1481_0 w_1493_0 w_1495_0 w_1499_0 w_1500_0 w_1501_0 w_1503_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 16 → Int) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 16 → Int) × mxxMat1 × (Fin 2 → mxxMat5) × mxxMat1 × ByteArray × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × ByteArray × mxxMat75 × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × mxxMat1 × (Fin 2 → mxxMat5) × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat5) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 14400 → Int) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 14400 → Int) × mxxMat1 × (Fin 2 → mxxMat5) × mxxMat1 × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 72000 → Int) × (Fin 72000 → Int) × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat8 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat5 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end W_Stage_keys_online_0_0
