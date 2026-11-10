import MxxIR
import MxxRuntime
import MxxSecurity.Sampling

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Eval

structure Params where
  unit : Unit

local notation "mxxNat0" => 421249166578543632464236954685046510773404157767471124412817604609
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 2 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 4 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix mxxNat0 8192 3 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 4 1
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 1 1
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 3 1
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix 23384026193386519304488586269947798086183317045249 8192 2 3
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix 23384026193386519304488586269947798086183317045249 8192 2 2
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix 23384026193386519304488586269947798086183317045249 8192 2 1
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix 18014398507892737 8192 2 1
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix mxxNat0 8192 2 3
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix mxxNat0 8192 2 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : mxxMat0
  w_4_0 : mxxMat1
  w_5_0 : mxxMat2
  w_6_0 : mxxMat2
  w_7_0 : mxxMat2
  w_9_0 : mxxMat2
  w_10_0 : mxxMat3
  w_10_concat_1 : mxxMat4
  w_11_0 : mxxMat2
  w_12_0 : mxxMat2
  w_13_0 : mxxMat4
  w_15_0 : mxxMat2
  w_16_0 : mxxMat5
  w_18_0 : mxxMat4
  w_20_0 : mxxMat0
  w_21_0 : mxxMat6
  w_22_0 : mxxMat7
  w_23_0 : mxxMat7
  w_24_0 : mxxMat7
  w_26_0 : mxxMat7
  w_27_0 : mxxMat8
  w_27_concat_1 : mxxMat0
  w_28_0 : mxxMat7
  w_29_0 : mxxMat7
  w_30_0 : mxxMat0
  w_31_0 : mxxMat9
  w_32_0 : mxxMat10
  w_33_0 : mxxMat7
  w_35_0 : mxxMat11
  w_37_0 : mxxMat0
  w_39_0 : mxxMat12

abbrev generatedRoot.constraints_3 (w_14_0 : mxxMat13) (w_27_0 : mxxMat8) (w_28_0 : mxxMat7) (w_29_0 : mxxMat7) (w_30_0 : mxxMat0) (w_31_0 : mxxMat9) (w_32_0 : mxxMat10) (w_33_0 : mxxMat7) (w_34_0 : mxxMat7) (w_35_0 : mxxMat11) (w_36_0 : mxxMat11) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat12) (outputs : mxxMat12) : Prop :=
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_27_0 2 3 0 1 w_29_0 ∧
  MxxRuntime.concatRows w_28_0 w_29_0 w_30_0 ∧
  MxxRuntime.modulusReduceRuns w_14_0 w_31_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 3 ∧
  MxxRuntime.sliceMatrix w_31_0 0 2 0 2 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_27_0 0 1 0 1 w_33_0 ∧
  MxxRuntime.rnsModUpRuns [18014398507892737, 18014398508138497] 1 true w_34_0 w_35_0 ∧
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497, 72057594037616641] (1032193) w_36_0 w_37_0 ∧
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497] (1032193) w_38_0 w_39_0 ∧
  outputs = w_39_0

abbrev generatedRoot.constraints_2 (w_14_0 : mxxMat13) (w_21_0 : mxxMat6) (w_22_0 : mxxMat7) (w_23_0 : mxxMat7) (w_24_0 : mxxMat7) (w_25_0 : mxxMat7) (w_26_0 : mxxMat7) (w_27_0 : mxxMat8) (w_27_concat_1 : mxxMat0) (w_28_0 : mxxMat7) (w_29_0 : mxxMat7) (w_30_0 : mxxMat0) (w_31_0 : mxxMat9) (w_32_0 : mxxMat10) (w_33_0 : mxxMat7) (w_34_0 : mxxMat7) (w_35_0 : mxxMat11) (w_36_0 : mxxMat11) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat12) (outputs : mxxMat12) : Prop :=
  MxxRuntime.sliceMatrix w_21_0 0 1 0 1 w_22_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_21_0 1 2 0 1 w_23_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_21_0 2 3 0 1 w_24_0 ∧
  0 ≤ 3 ∧
  3 < 4 ∧
  4 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_21_0 3 4 0 1 w_26_0 ∧
  MxxRuntime.concatRows w_22_0 w_25_0 w_27_concat_1 ∧
  MxxRuntime.concatRows w_27_concat_1 w_26_0 w_27_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_27_0 1 2 0 1 w_28_0 ∧
  0 ≤ 2 ∧
  generatedRoot.constraints_3 w_14_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 outputs

abbrev generatedRoot.constraints_1 (w_14_0 : mxxMat13) (w_1_0 : mxxMat0) (w_10_0 : mxxMat3) (w_11_0 : mxxMat2) (w_12_0 : mxxMat2) (w_13_0 : mxxMat4) (w_15_0 : mxxMat2) (w_16_0 : mxxMat5) (w_17_0 : mxxMat14) (w_18_0 : mxxMat4) (w_19_0 : mxxMat4) (w_20_0 : mxxMat0) (w_21_0 : mxxMat6) (w_22_0 : mxxMat7) (w_23_0 : mxxMat7) (w_24_0 : mxxMat7) (w_25_0 : mxxMat7) (w_26_0 : mxxMat7) (w_27_0 : mxxMat8) (w_27_concat_1 : mxxMat0) (w_28_0 : mxxMat7) (w_29_0 : mxxMat7) (w_30_0 : mxxMat0) (w_31_0 : mxxMat9) (w_32_0 : mxxMat10) (w_33_0 : mxxMat7) (w_34_0 : mxxMat7) (w_35_0 : mxxMat11) (w_36_0 : mxxMat11) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat12) (outputs : mxxMat12) : Prop :=
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_10_0 1 2 0 1 w_11_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_10_0 2 3 0 1 w_12_0 ∧
  MxxRuntime.concatRows w_11_0 w_12_0 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 0 1 w_15_0 ∧
  MxxRuntime.rnsModUpRuns [18014398507892737, 18014398508138497, 18014398508400641] 1 true w_15_0 w_16_0 ∧
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497, 18014398508400641, 72057594037616641] (1032193) w_17_0 w_18_0 ∧
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497, 18014398508400641] (1032193) w_19_0 w_20_0 ∧
  MxxRuntime.tensorRuns w_1_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  generatedRoot.constraints_2 w_14_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 outputs

abbrev generatedRoot.constraints_0 (w_0_0 : mxxMat4) (w_2_0 : mxxMat4) (w_3_0 : mxxMat4) (w_14_0 : mxxMat13) (w_1_0 : mxxMat0) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_6_0 : mxxMat2) (w_7_0 : mxxMat2) (w_8_0 : mxxMat2) (w_9_0 : mxxMat2) (w_10_0 : mxxMat3) (w_10_concat_1 : mxxMat4) (w_11_0 : mxxMat2) (w_12_0 : mxxMat2) (w_13_0 : mxxMat4) (w_15_0 : mxxMat2) (w_16_0 : mxxMat5) (w_17_0 : mxxMat14) (w_18_0 : mxxMat4) (w_19_0 : mxxMat4) (w_20_0 : mxxMat0) (w_21_0 : mxxMat6) (w_22_0 : mxxMat7) (w_23_0 : mxxMat7) (w_24_0 : mxxMat7) (w_25_0 : mxxMat7) (w_26_0 : mxxMat7) (w_27_0 : mxxMat8) (w_27_concat_1 : mxxMat0) (w_28_0 : mxxMat7) (w_29_0 : mxxMat7) (w_30_0 : mxxMat0) (w_31_0 : mxxMat9) (w_32_0 : mxxMat10) (w_33_0 : mxxMat7) (w_34_0 : mxxMat7) (w_35_0 : mxxMat11) (w_36_0 : mxxMat11) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat12) (outputs : mxxMat12) : Prop :=
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497, 18014398508400641] (1032193) w_0_0 w_1_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_3_0 w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 0 1 w_5_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_4_0 1 2 0 1 w_6_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_4_0 2 3 0 1 w_7_0 ∧
  0 ≤ 3 ∧
  3 < 4 ∧
  4 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_4_0 3 4 0 1 w_9_0 ∧
  MxxRuntime.concatRows w_5_0 w_8_0 w_10_concat_1 ∧
  MxxRuntime.concatRows w_10_concat_1 w_9_0 w_10_0 ∧
  generatedRoot.constraints_1 w_14_0 w_1_0 w_10_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 outputs

abbrev generatedRoot.body (params : Params) (inputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat13 × Unit) (outputs : mxxMat12) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : mxxMat4 := inputs.1
  let w_2_0 : mxxMat4 := inputs.2.1
  let w_3_0 : mxxMat4 := inputs.2.2.1
  let w_14_0 : mxxMat13 := inputs.2.2.2.1
  let w_1_0 := witness.w_1_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat2 := MxxRuntime.matrixAdd w_6_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_10_concat_1 := witness.w_10_concat_1
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat14 := MxxRuntime.matrixMul w_14_0 w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat4 := MxxRuntime.matrixAdd w_13_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat7 := MxxRuntime.matrixAdd w_23_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_27_concat_1 := witness.w_27_concat_1
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat7 := MxxRuntime.matrixScale (149176486870899040271091973769539) w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat11 := MxxRuntime.matrixMul w_32_0 w_35_0
  let w_37_0 := witness.w_37_0
  let w_38_0 : mxxMat0 := MxxRuntime.matrixAdd w_30_0 w_37_0
  let w_39_0 := witness.w_39_0
  generatedRoot.constraints_0 w_0_0 w_2_0 w_3_0 w_14_0 w_1_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_10_concat_1 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 outputs

def generatedRoot (params : Params) (inputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat13 × Unit) (outputs : mxxMat12) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness

noncomputable def generatedRoot.comp (params : Params) (inputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat13 × Unit) : MxxSecurity.Comp (mxxMat12) := do
  pure (MxxSecurity.determined fun outputs : mxxMat12 ↦
    ∃ witness : generatedRoot.Witness, generatedRoot.body params inputs outputs witness)


end Eval
