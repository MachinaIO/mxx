import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_preimages_1_1

structure Params where
  unit : Unit

local notation "mxxNat0" => 46622222787362522255596712427438640158977719430215665481763441408391999111509151608999934968273383011642417378633647840562514917807804816922347656847676979519702813451900357865140701070628284347817487928891263004147023796456197114198655078174334397712252374369987792302056981468300367512375728844419128305889690907960192125350868940719263879317724644989970506210974573641462431438389817409111571168135085524040305727541538565027353777260752527223173138698768973043853522644067239267926017

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 544 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 544
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 1

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : mxxMat0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : MxxRuntime.TrapdoorValue (mxxMat1) Unit) (w_2_0 : mxxMat2) (w_1_0 : mxxMat1) (w_3_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_1_0 w_0_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
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
