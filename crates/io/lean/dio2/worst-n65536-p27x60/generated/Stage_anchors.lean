import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_anchors

structure Params where
  unit : Unit

local notation "mxxNat0" => 46622222787362522255596712427438640158977719430215665481763441408391999111509151608999934968273383011642417378633647840562514917807804816922347656847676979519702813451900357865140701070628284347817487928891263004147023796456197114198655078174334397712252374369987792302056981468300367512375728844419128305889690907960192125350868940719263879317724644989970506210974573641462431438389817409111571168135085524040305727541538565027353777260752527223173138698768973043853522644067239267926017

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 544

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_0_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit
  w_0_0 : mxxMat0
  w_1_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit
  w_1_0 : mxxMat0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_0_0 : mxxMat0) (w_1_1 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_1_0 : mxxMat0) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop :=
  MxxRuntime.trapdoorSample backend (4/1) 64 270 76848720 w_0_0 w_0_1 ∧
  MxxRuntime.trapdoorSample backend (4/1) 64 270 76848720 w_1_0 w_1_1 ∧
  outputs = (w_0_0, (w_0_1, (w_1_0, (w_1_1, ()))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (params : Params) (_ : Unit) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_1 := witness.w_0_1
  let w_0_0 := witness.w_0_0
  let w_1_1 := witness.w_1_1
  let w_1_0 := witness.w_1_0
  generatedRoot.constraints_0 backend w_0_1 w_0_0 w_1_1 w_1_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (params : Params) (_ : Unit) (outputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend params () outputs witness


end Stage_anchors
