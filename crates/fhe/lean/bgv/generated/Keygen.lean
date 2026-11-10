import MxxIR
import MxxRuntime
import MxxSecurity.Sampling

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Keygen

structure Params where
  unit : Unit

local notation "mxxNat0" => 421249166578543632464236954685046510773404157767471124412817604609
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 8192 1 3
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix 18014398507892737 8192 1 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix mxxNat0 8192 1 1
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix mxxNat0 8192 1 2
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix mxxNat0 8192 2 3
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_0 : mxxMat0
  w_1_0 : mxxMat1
  w_2_0 : mxxMat2
  w_3_0 : mxxMat3
  w_5_0 : mxxMat0
  w_8_0 : mxxMat1
  w_10_0 : mxxMat3
  w_14_0 : mxxMat0
  w_14_concat_1 : mxxMat4
  w_16_0 : mxxMat5
  w_17_0 : mxxMat1
  w_19_0 : mxxMat1
  w_22_0 : mxxMat6

abbrev generatedRoot.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat2) (w_3_0 : mxxMat3) (w_5_0 : mxxMat0) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat3) (w_11_0 : mxxMat3) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_14_concat_1 : mxxMat4) (w_15_0 : mxxMat0) (w_16_0 : mxxMat5) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat6) (outputs : mxxMat5 × mxxMat6 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.uniformResidueSample w_0_0 ∧
  MxxRuntime.uniformIntervalSample (-1) (1) w_1_0 ∧
  MxxRuntime.modulusReduceRuns w_1_0 w_2_0 ∧
  MxxRuntime.centeredRebaseRuns w_2_0 w_3_0 ∧
  MxxRuntime.gaussianSample ((3602879701896397/1125899906842624)) (20) w_5_0 ∧
  MxxRuntime.modulusReduceRuns w_1_0 w_8_0 ∧
  MxxRuntime.rnsModUpRuns [18014398507892737, 18014398508138497, 18014398508400641] 3 false w_9_0 w_10_0 ∧
  MxxRuntime.concatColumns w_11_0 w_12_0 w_14_concat_1 ∧
  MxxRuntime.concatColumns w_14_concat_1 w_13_0 w_14_0 ∧
  MxxRuntime.concatRows w_0_0 w_15_0 w_16_0 ∧
  MxxRuntime.uniformResidueSample w_17_0 ∧
  MxxRuntime.gaussianSample ((3602879701896397/1125899906842624)) (20) w_19_0 ∧
  MxxRuntime.concatRows w_17_0 w_21_0 w_22_0 ∧
  outputs = (w_16_0, (w_22_0, (w_1_0, ())))

abbrev generatedRoot.body (params : Params) (_ : Unit) (outputs : mxxMat5 × mxxMat6 × mxxMat1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 := witness.w_0_0
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_3_0 w_0_0
  let w_5_0 := witness.w_5_0
  let w_6_0 : mxxMat0 := MxxRuntime.matrixScale (1032193) w_5_0
  let w_7_0 : mxxMat0 := MxxRuntime.matrixAdd w_4_0 w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_8_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat3 := MxxRuntime.matrixScale (23384026194045816390345886928489845842762721017857) w_10_0
  let w_12_0 : mxxMat3 := MxxRuntime.matrixScale (23384026193726801671378034657273812936317868343297) w_10_0
  let w_13_0 : mxxMat3 := MxxRuntime.matrixScale (23384026193386519304488586269947798086183317045249) w_10_0
  let w_14_0 := witness.w_14_0
  let w_14_concat_1 := witness.w_14_concat_1
  let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_7_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := MxxRuntime.matrixScale (1032193) w_19_0
  let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_0 w_20_0
  let w_22_0 := witness.w_22_0
  generatedRoot.constraints_0 w_0_0 w_1_0 w_2_0 w_3_0 w_5_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_14_concat_1 w_15_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 outputs

def generatedRoot (params : Params) (_ : Unit) (outputs : mxxMat5 × mxxMat6 × mxxMat1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params () outputs witness

noncomputable def generatedRoot.comp (params : Params) (_ : Unit) : MxxSecurity.Comp (mxxMat5 × mxxMat6 × mxxMat1 × Unit) := do
  let w_0_0 : mxxMat0 ← MxxSecurity.sampleResidue
  let w_1_0 : mxxMat1 ← MxxSecurity.sampleInterval (-1) (1)
  let w_5_0 : mxxMat0 ← MxxSecurity.sampleGaussian ((3602879701896397/1125899906842624)) (20)
  let w_17_0 : mxxMat1 ← MxxSecurity.sampleResidue
  let w_19_0 : mxxMat1 ← MxxSecurity.sampleGaussian ((3602879701896397/1125899906842624)) (20)
  pure (MxxSecurity.determined fun outputs : mxxMat5 × mxxMat6 × mxxMat1 × Unit ↦
    ∃ witness : generatedRoot.Witness, witness.w_0_0 = w_0_0 ∧ witness.w_1_0 = w_1_0 ∧ witness.w_5_0 = w_5_0 ∧ witness.w_17_0 = w_17_0 ∧ witness.w_19_0 = w_19_0 ∧ generatedRoot.body params () outputs witness)


end Keygen
