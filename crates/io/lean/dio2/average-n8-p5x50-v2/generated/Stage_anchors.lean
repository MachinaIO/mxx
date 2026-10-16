import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_anchors

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit
  w_0_0 : mxxMat0
  w_1_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit
  w_1_0 : mxxMat0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (w_0_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_0_0 : mxxMat0) (w_1_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_1_0 : mxxMat0) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop :=
  MxxRuntime.trapdoorSampleAt backend tape (path ++ [0]) (4/1) 4096 25 22011706 w_0_0 w_0_1 ∧
  MxxRuntime.trapdoorSampleAt backend tape (path ++ [1]) (4/1) 4096 25 22011706 w_1_0 w_1_1 ∧
  outputs = (w_0_0, (w_0_1, (w_1_0, (w_1_1, ()))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_1 := witness.w_0_1
  let w_0_0 := witness.w_0_0
  let w_1_1 := witness.w_1_1
  let w_1_0 := witness.w_1_0
  generatedRoot.constraints_0 backend tape path w_0_1 w_0_0 w_1_1 w_1_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend tape path params () outputs witness


end Stage_anchors
