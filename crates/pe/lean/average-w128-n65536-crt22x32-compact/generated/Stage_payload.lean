import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_payload

structure Params where
  unit : Unit

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_0 : mxxMat0

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [0]) (-1) (1) w_0_0 ∧
  outputs = w_0_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : mxxMat0) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 := witness.w_0_0
  generatedRoot.constraints_0 tape path w_0_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : mxxMat0) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params () outputs witness


end Stage_payload
