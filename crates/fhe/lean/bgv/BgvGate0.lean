import MxxIR
import MxxRuntime
import MxxSecurity.Sampling

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

/-! The graph of one slot-wise multiplication of two level-0 ciphertexts, from level 0 to
level 1, as the exported evaluation of a circuit contains it. It is
not part of any statement: the correctness proof reads the circuit's graph as instances of it
(`BgvDepth.eval_spec`). -/

namespace Gate0

structure Params where
  unit : Unit

local notation "mxxNat0" => 421249166578543632464236954685046510773404157767471124412817604609
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 4 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix mxxNat0 8192 3 1
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 2 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix mxxNat0 8192 2 1
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix mxxNat0 8192 2 3

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : mxxMat0
  w_3_0 : mxxMat1
  w_4_0 : mxxMat1
  w_5_0 : mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat2
  w_8_concat_1 : mxxMat3
  w_9_0 : mxxMat1
  w_10_0 : mxxMat1
  w_11_0 : mxxMat3
  w_13_0 : mxxMat1
  w_14_0 : mxxMat4
  w_16_0 : mxxMat3
  w_18_0 : mxxMat5

abbrev generatedRoot.constraints_1 (w_8_0 : mxxMat2) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat3) (w_13_0 : mxxMat1) (w_14_0 : mxxMat4) (w_15_0 : mxxMat6) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat5) (outputs : mxxMat5) : Prop :=
  1 < 2 ∧
  2 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_8_0 1 2 0 1 w_9_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_8_0 2 3 0 1 w_10_0 ∧
  MxxRuntime.concatRows w_9_0 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 3 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 0 1 w_13_0 ∧
  MxxRuntime.rnsModUpRuns [18014398507892737, 18014398508138497, 18014398508400641] 1 true w_13_0 w_14_0 ∧
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497, 18014398508400641, 72057594037616641] (1032193) w_15_0 w_16_0 ∧
  MxxRuntime.rnsModDownRuns [18014398507892737, 18014398508138497, 18014398508400641] (1032193) w_17_0 w_18_0 ∧
  outputs = w_18_0

abbrev generatedRoot.constraints_0 (w_0_0 : mxxMat3) (w_1_0 : mxxMat3) (w_2_0 : mxxMat0) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat2) (w_8_concat_1 : mxxMat3) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat3) (w_13_0 : mxxMat1) (w_14_0 : mxxMat4) (w_15_0 : mxxMat6) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat5) (outputs : mxxMat5) : Prop :=
  MxxRuntime.tensorRuns w_0_0 w_1_0 w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 0 1 w_3_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_2_0 1 2 0 1 w_4_0 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_2_0 2 3 0 1 w_5_0 ∧
  0 ≤ 3 ∧
  3 < 4 ∧
  4 ≤ 4 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  MxxRuntime.sliceMatrix w_2_0 3 4 0 1 w_7_0 ∧
  MxxRuntime.concatRows w_3_0 w_6_0 w_8_concat_1 ∧
  MxxRuntime.concatRows w_8_concat_1 w_7_0 w_8_0 ∧
  0 ≤ 1 ∧
  generatedRoot.constraints_1 w_8_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 outputs

abbrev generatedRoot.body (params : Params) (inputs : mxxMat3 × mxxMat3 × mxxMat7 × Unit) (outputs : mxxMat5) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : mxxMat3 := inputs.1
  let w_1_0 : mxxMat3 := inputs.2.1
  let w_12_0 : mxxMat7 := inputs.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_6_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_8_concat_1 := witness.w_8_concat_1
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat6 := MxxRuntime.matrixMul w_12_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat3 := MxxRuntime.matrixAdd w_11_0 w_16_0
  let w_18_0 := witness.w_18_0
  generatedRoot.constraints_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_8_concat_1 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 outputs

def generatedRoot (params : Params) (inputs : mxxMat3 × mxxMat3 × mxxMat7 × Unit) (outputs : mxxMat5) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness

noncomputable def generatedRoot.comp (params : Params) (inputs : mxxMat3 × mxxMat3 × mxxMat7 × Unit) : MxxSecurity.Comp (mxxMat5) := do
  pure (MxxSecurity.determined fun outputs : mxxMat5 ↦
    ∃ witness : generatedRoot.Witness, generatedRoot.body params inputs outputs witness)


/-- The parameters of the graph. -/
def params : Params := { «unit» := () }

end Gate0
