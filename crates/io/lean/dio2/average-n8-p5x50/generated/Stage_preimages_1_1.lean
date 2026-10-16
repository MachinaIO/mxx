import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_preimages_1_1

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 54 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 2 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : mxxMat0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : MxxRuntime.TrapdoorValue (mxxMat1) Unit) (w_2_0 : mxxMat2) (w_1_0 : mxxMat1) (w_3_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ (22011706) ∧
  MxxRuntime.preimageRunsDispatched backend w_1_0 w_0_0 w_2_0 (Int.toNat (22011706)) w_3_0 ∧
  outputs = w_3_0

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (inputs : MxxRuntime.TrapdoorValue (mxxMat1) Unit × mxxMat2 × Unit) (outputs : mxxMat0) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : MxxRuntime.TrapdoorValue (mxxMat1) Unit := inputs.1
  let w_2_0 : mxxMat2 := inputs.2.1
  let w_1_0 : mxxMat1 := MxxRuntime.trapdoorPublic w_0_0
  let w_3_0 := witness.w_3_0
  generatedRoot.constraints_0 backend w_0_0 w_2_0 w_1_0 w_3_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : MxxRuntime.TrapdoorValue (mxxMat1) Unit × mxxMat2 × Unit) (outputs : mxxMat0) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend tape path params inputs outputs witness


end Stage_preimages_1_1
