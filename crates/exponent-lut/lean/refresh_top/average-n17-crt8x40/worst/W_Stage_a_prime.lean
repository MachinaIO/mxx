import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Stage_a_prime

structure Params where
  unit : Unit

local notation "mxxNat0" => 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 131072 2 28
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 131072 2 56

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : mxxMat0
  w_2_0 : mxxMat0
  w_3_0 : mxxMat1

abbrev generatedRoot.constraints_0 (hashModel : MxxRuntime.HashModel) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (w_3_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112, 0]) ([]) (w_0_0) w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112, 1]) ([]) (w_0_0) w_2_0 ∧
  MxxRuntime.concatColumns w_1_0 w_2_0 w_3_0 ∧
  outputs = w_3_0

abbrev generatedRoot.body (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray) (outputs : mxxMat1) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : ByteArray := inputs
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  generatedRoot.constraints_0 hashModel w_0_0 w_1_0 w_2_0 w_3_0 outputs

def generatedRoot (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray) (outputs : mxxMat1) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body hashModel params inputs outputs witness


end W_Stage_a_prime
