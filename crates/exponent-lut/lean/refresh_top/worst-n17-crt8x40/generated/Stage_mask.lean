import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_mask

structure Params where
  unit : Unit

local notation "mxxNat0" => 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_0 : mxxMat0

abbrev generatedRoot.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  MxxRuntime.uniformIntervalSample (-1) (1) w_0_0 ∧
  outputs = w_0_0

abbrev generatedRoot.body (params : Params) (_ : Unit) (outputs : mxxMat0) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 := witness.w_0_0
  generatedRoot.constraints_0 w_0_0 outputs

def generatedRoot (params : Params) (_ : Unit) (outputs : mxxMat0) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params () outputs witness


end Stage_mask
