import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_setup_trapdoor

structure Params where
  unit : Unit

local notation "mxxNat0" => 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 295

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit
  w_0_0 : mxxMat0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (w_0_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_0_0 : mxxMat0) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop :=
  MxxRuntime.statTrapdoorSampleAt backend tape (path ++ [0]) 265 (5154369773525533/1125899906842624) 4096 30 2610114297 w_0_0 w_0_1 ∧
  outputs = (w_0_0, (w_0_1, ()))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_1 := witness.w_0_1
  let w_0_0 := witness.w_0_0
  generatedRoot.constraints_0 backend tape path w_0_1 w_0_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend tape path params () outputs witness


end Stage_setup_trapdoor
