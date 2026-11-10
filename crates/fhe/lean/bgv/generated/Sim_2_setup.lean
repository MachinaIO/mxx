import MxxIR
import MxxRuntime
import MxxSecurity.Sampling

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Sim_2_setup

structure Params where
  unit : Unit

local notation "mxxNat0" => 421249166578543632464236954685046510773404157767471124412817604609
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 8192 2 3
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 2

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : mxxMat0
  w_3_0 : mxxMat1
  w_5_0 : mxxMat1
  w_7_0 : mxxMat2

abbrev generatedRoot.constraints_0 (w_0_0 : mxxMat3) (w_2_0 : mxxMat3) (w_1_0 : mxxMat0) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat2) (outputs : mxxMat3 × mxxMat0 × mxxMat2 × Unit) : Prop :=
  MxxRuntime.uniformResidueSample w_1_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 0 1 w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 1 2 w_5_0 ∧
  MxxRuntime.concatRows w_4_0 w_6_0 w_7_0 ∧
  outputs = (w_0_0, (w_1_0, (w_7_0, ())))

abbrev generatedRoot.body (params : Params) (inputs : mxxMat3 × mxxMat3 × Unit) (outputs : mxxMat3 × mxxMat0 × mxxMat2 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : mxxMat3 := inputs.1
  let w_2_0 : mxxMat3 := inputs.2.1
  let w_1_0 := witness.w_1_0
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat1 := MxxRuntime.matrixScale (1032193) w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 : mxxMat1 := MxxRuntime.matrixScale (1032193) w_5_0
  let w_7_0 := witness.w_7_0
  generatedRoot.constraints_0 w_0_0 w_2_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 outputs

def generatedRoot (params : Params) (inputs : mxxMat3 × mxxMat3 × Unit) (outputs : mxxMat3 × mxxMat0 × mxxMat2 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness

noncomputable def generatedRoot.comp (params : Params) (inputs : mxxMat3 × mxxMat3 × Unit) : MxxSecurity.Comp (mxxMat3 × mxxMat0 × mxxMat2 × Unit) := do
  let w_1_0 : mxxMat0 ← MxxSecurity.sampleResidue
  pure (MxxSecurity.determined fun outputs : mxxMat3 × mxxMat0 × mxxMat2 × Unit ↦
    ∃ witness : generatedRoot.Witness, witness.w_1_0 = w_1_0 ∧ generatedRoot.body params inputs outputs witness)


end Sim_2_setup
