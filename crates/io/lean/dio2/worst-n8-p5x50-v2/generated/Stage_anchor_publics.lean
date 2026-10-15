import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_anchor_publics

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where

abbrev generatedRoot.constraints_0 (w_1_0 : mxxMat0) (w_3_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_1_0, (w_3_0, ()))

abbrev generatedRoot.body (params : Params) (inputs : MxxRuntime.TrapdoorValue (mxxMat0) Unit × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (_witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.1
  let w_2_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_1_0 : mxxMat0 := MxxRuntime.trapdoorPublic w_0_0
  let w_3_0 : mxxMat0 := MxxRuntime.trapdoorPublic w_2_0
  generatedRoot.constraints_0 w_1_0 w_3_0 outputs

def generatedRoot (params : Params) (inputs : MxxRuntime.TrapdoorValue (mxxMat0) Unit × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Stage_anchor_publics
