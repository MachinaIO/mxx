import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_keygen_helpers

structure Params where
  unit : Unit

local notation "mxxNat0" => 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489
local notation "mxxNat1" => 2007780228562889017327695763426836351403150354206288662473911661538034308791619296312261592598031117
local notation "mxxNat2" => 5000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 60 45

abbrev parallel_generatedRoot_0.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  MxxRuntime.gaussianSampleAt tape (path ++ [0]) ((mxxNat1/mxxNat2)) (2610114297) w_0_0 ∧
  outputs = w_0_0

def parallel_generatedRoot_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  ∃ (w_0_0 : mxxMat0),
    parallel_generatedRoot_0.constraints_0 tape path w_0_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_0 : Fin 256 → mxxMat0

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 256 → mxxMat0) (outputs : Fin 256 → mxxMat0) : Prop :=
  (256) = 256 ∧
  (∀ i : Fin 256, parallel_generatedRoot_0 tape (path ++ [0, i.val]) params i () (w_0_0 i)) ∧
  outputs = w_0_0

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : Fin 256 → mxxMat0) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 := witness.w_0_0
  generatedRoot.constraints_0 tape path params w_0_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : Fin 256 → mxxMat0) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params () outputs witness


end Stage_keygen_helpers
