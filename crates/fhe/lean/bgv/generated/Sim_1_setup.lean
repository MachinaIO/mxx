import MxxIR
import MxxRuntime
import MxxSecurity.Sampling

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Sim_1_setup

structure Params where
  unit : Unit

local notation "mxxNat0" => 421249166578543632464236954685046510773404157767471124412817604609
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 8192 2 3
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_0 : mxxMat0
  w_5_0 : mxxMat1

abbrev generatedRoot.constraints_0 (w_0_0 : mxxMat0) (w_2_0 : mxxMat2) (w_4_0 : mxxMat2) (w_5_0 : mxxMat1) (outputs : mxxMat0 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.uniformResidueSample w_0_0 ∧
  MxxRuntime.concatRows w_2_0 w_4_0 w_5_0 ∧
  outputs = (w_0_0, (w_5_0, ()))

abbrev generatedRoot.body (params : Params) (inputs : mxxMat2 × mxxMat2 × Unit) (outputs : mxxMat0 × mxxMat1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : mxxMat2 := inputs.1
  let w_3_0 : mxxMat2 := inputs.2.1
  let w_0_0 := witness.w_0_0
  let w_2_0 : mxxMat2 := MxxRuntime.matrixScale (1032193) w_1_0
  let w_4_0 : mxxMat2 := MxxRuntime.matrixScale (1032193) w_3_0
  let w_5_0 := witness.w_5_0
  generatedRoot.constraints_0 w_0_0 w_2_0 w_4_0 w_5_0 outputs

def generatedRoot (params : Params) (inputs : mxxMat2 × mxxMat2 × Unit) (outputs : mxxMat0 × mxxMat1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness

noncomputable def generatedRoot.comp (params : Params) (inputs : mxxMat2 × mxxMat2 × Unit) : MxxSecurity.Comp (mxxMat0 × mxxMat1 × Unit) := do
  let w_0_0 : mxxMat0 ← MxxSecurity.sampleResidue
  pure (MxxSecurity.determined fun outputs : mxxMat0 × mxxMat1 × Unit ↦
    ∃ witness : generatedRoot.Witness, witness.w_0_0 = w_0_0 ∧ generatedRoot.body params inputs outputs witness)


end Sim_1_setup
