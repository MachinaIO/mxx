import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_preimages_0_0

structure Params where
  unit : Unit

local notation "mxxNat0" => 46622222787362522255596712427438640158977719430215665481763441408391999111509151608999934968273383011642417378633647840562514917807804816922347656847676979519702813451900357865140701070628284347817487928891263004147023796456197114198655078174334397712252374369987792302056981468300367512375728844419128305889690907960192125350868940719263879317724644989970506210974573641462431438389817409111571168135085524040305727541538565027353777260752527223173138698768973043853522644067239267926017

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 544
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 270
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 65536 544 270

abbrev parallel_generatedRoot_3.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_3 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_3.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_5.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_5.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_7.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_7.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_9.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_9 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_9.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_11.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_11 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_11.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_13.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_13.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_15.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_15 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_15.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_17.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_17.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_19.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_19.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_21.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_21 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_21.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_23.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_23 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_23.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_25.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_25 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_25.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_27.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_27 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_27.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_29.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_29 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_29.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_31.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_31 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_31.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_33.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_33 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_33.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_35.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_35 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_35.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_37.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_37 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_37.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_39.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_39 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_39.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_41.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_41 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_41.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_43.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_43 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_43.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_45.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_45 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_45.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_47.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_47 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_47.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_49.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_49 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_49.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_51.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_51 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_51.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_53.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_53 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_53.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_55.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (76848720) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (76848720)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_55 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_55.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev generatedRoot.InputType : Type :=
  MxxRuntime.TrapdoorValue (mxxMat0) Unit × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × (Fin 16779536 → mxxMat1) × Unit

abbrev generatedRoot.OutputType : Type :=
  (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × (Fin 16779536 → mxxMat2) × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Fin 16779536 → mxxMat2
  w_5_0 : Fin 16779536 → mxxMat2
  w_7_0 : Fin 16779536 → mxxMat2
  w_9_0 : Fin 16779536 → mxxMat2
  w_11_0 : Fin 16779536 → mxxMat2
  w_13_0 : Fin 16779536 → mxxMat2
  w_15_0 : Fin 16779536 → mxxMat2
  w_17_0 : Fin 16779536 → mxxMat2
  w_19_0 : Fin 16779536 → mxxMat2
  w_21_0 : Fin 16779536 → mxxMat2
  w_23_0 : Fin 16779536 → mxxMat2
  w_25_0 : Fin 16779536 → mxxMat2
  w_27_0 : Fin 16779536 → mxxMat2
  w_29_0 : Fin 16779536 → mxxMat2
  w_31_0 : Fin 16779536 → mxxMat2
  w_33_0 : Fin 16779536 → mxxMat2
  w_35_0 : Fin 16779536 → mxxMat2
  w_37_0 : Fin 16779536 → mxxMat2
  w_39_0 : Fin 16779536 → mxxMat2
  w_41_0 : Fin 16779536 → mxxMat2
  w_43_0 : Fin 16779536 → mxxMat2
  w_45_0 : Fin 16779536 → mxxMat2
  w_47_0 : Fin 16779536 → mxxMat2
  w_49_0 : Fin 16779536 → mxxMat2
  w_51_0 : Fin 16779536 → mxxMat2
  w_53_0 : Fin 16779536 → mxxMat2
  w_55_0 : Fin 16779536 → mxxMat2

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_34_0 : Fin 16779536 → mxxMat1) (w_36_0 : Fin 16779536 → mxxMat1) (w_38_0 : Fin 16779536 → mxxMat1) (w_40_0 : Fin 16779536 → mxxMat1) (w_42_0 : Fin 16779536 → mxxMat1) (w_44_0 : Fin 16779536 → mxxMat1) (w_46_0 : Fin 16779536 → mxxMat1) (w_48_0 : Fin 16779536 → mxxMat1) (w_50_0 : Fin 16779536 → mxxMat1) (w_52_0 : Fin 16779536 → mxxMat1) (w_54_0 : Fin 16779536 → mxxMat1) (w_1_0 : mxxMat0) (w_3_0 : Fin 16779536 → mxxMat2) (w_5_0 : Fin 16779536 → mxxMat2) (w_7_0 : Fin 16779536 → mxxMat2) (w_9_0 : Fin 16779536 → mxxMat2) (w_11_0 : Fin 16779536 → mxxMat2) (w_13_0 : Fin 16779536 → mxxMat2) (w_15_0 : Fin 16779536 → mxxMat2) (w_17_0 : Fin 16779536 → mxxMat2) (w_19_0 : Fin 16779536 → mxxMat2) (w_21_0 : Fin 16779536 → mxxMat2) (w_23_0 : Fin 16779536 → mxxMat2) (w_25_0 : Fin 16779536 → mxxMat2) (w_27_0 : Fin 16779536 → mxxMat2) (w_29_0 : Fin 16779536 → mxxMat2) (w_31_0 : Fin 16779536 → mxxMat2) (w_33_0 : Fin 16779536 → mxxMat2) (w_35_0 : Fin 16779536 → mxxMat2) (w_37_0 : Fin 16779536 → mxxMat2) (w_39_0 : Fin 16779536 → mxxMat2) (w_41_0 : Fin 16779536 → mxxMat2) (w_43_0 : Fin 16779536 → mxxMat2) (w_45_0 : Fin 16779536 → mxxMat2) (w_47_0 : Fin 16779536 → mxxMat2) (w_49_0 : Fin 16779536 → mxxMat2) (w_51_0 : Fin 16779536 → mxxMat2) (w_53_0 : Fin 16779536 → mxxMat2) (w_55_0 : Fin 16779536 → mxxMat2) (outputs : (generatedRoot.OutputType)) : Prop :=
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_35 backend tape (path ++ [35, i.val]) params i (w_1_0, (w_0_0, ((w_34_0 i), ()))) (w_35_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_37 backend tape (path ++ [37, i.val]) params i (w_1_0, (w_0_0, ((w_36_0 i), ()))) (w_37_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_39 backend tape (path ++ [39, i.val]) params i (w_1_0, (w_0_0, ((w_38_0 i), ()))) (w_39_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_41 backend tape (path ++ [41, i.val]) params i (w_1_0, (w_0_0, ((w_40_0 i), ()))) (w_41_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_43 backend tape (path ++ [43, i.val]) params i (w_1_0, (w_0_0, ((w_42_0 i), ()))) (w_43_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_45 backend tape (path ++ [45, i.val]) params i (w_1_0, (w_0_0, ((w_44_0 i), ()))) (w_45_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_47 backend tape (path ++ [47, i.val]) params i (w_1_0, (w_0_0, ((w_46_0 i), ()))) (w_47_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_49 backend tape (path ++ [49, i.val]) params i (w_1_0, (w_0_0, ((w_48_0 i), ()))) (w_49_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_51 backend tape (path ++ [51, i.val]) params i (w_1_0, (w_0_0, ((w_50_0 i), ()))) (w_51_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_53 backend tape (path ++ [53, i.val]) params i (w_1_0, (w_0_0, ((w_52_0 i), ()))) (w_53_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_55 backend tape (path ++ [55, i.val]) params i (w_1_0, (w_0_0, ((w_54_0 i), ()))) (w_55_0 i)) ∧
  outputs = (w_3_0, (w_5_0, (w_7_0, (w_9_0, (w_11_0, (w_13_0, (w_15_0, (w_17_0, (w_19_0, (w_21_0, (w_23_0, (w_25_0, (w_27_0, (w_29_0, (w_31_0, (w_33_0, (w_35_0, (w_37_0, (w_39_0, (w_41_0, (w_43_0, (w_45_0, (w_47_0, (w_49_0, (w_51_0, (w_53_0, (w_55_0, ())))))))))))))))))))))))))))

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : Fin 16779536 → mxxMat1) (w_4_0 : Fin 16779536 → mxxMat1) (w_6_0 : Fin 16779536 → mxxMat1) (w_8_0 : Fin 16779536 → mxxMat1) (w_10_0 : Fin 16779536 → mxxMat1) (w_12_0 : Fin 16779536 → mxxMat1) (w_14_0 : Fin 16779536 → mxxMat1) (w_16_0 : Fin 16779536 → mxxMat1) (w_18_0 : Fin 16779536 → mxxMat1) (w_20_0 : Fin 16779536 → mxxMat1) (w_22_0 : Fin 16779536 → mxxMat1) (w_24_0 : Fin 16779536 → mxxMat1) (w_26_0 : Fin 16779536 → mxxMat1) (w_28_0 : Fin 16779536 → mxxMat1) (w_30_0 : Fin 16779536 → mxxMat1) (w_32_0 : Fin 16779536 → mxxMat1) (w_1_0 : mxxMat0) (w_3_0 : Fin 16779536 → mxxMat2) (w_5_0 : Fin 16779536 → mxxMat2) (w_7_0 : Fin 16779536 → mxxMat2) (w_9_0 : Fin 16779536 → mxxMat2) (w_11_0 : Fin 16779536 → mxxMat2) (w_13_0 : Fin 16779536 → mxxMat2) (w_15_0 : Fin 16779536 → mxxMat2) (w_17_0 : Fin 16779536 → mxxMat2) (w_19_0 : Fin 16779536 → mxxMat2) (w_21_0 : Fin 16779536 → mxxMat2) (w_23_0 : Fin 16779536 → mxxMat2) (w_25_0 : Fin 16779536 → mxxMat2) (w_27_0 : Fin 16779536 → mxxMat2) (w_29_0 : Fin 16779536 → mxxMat2) (w_31_0 : Fin 16779536 → mxxMat2) (w_33_0 : Fin 16779536 → mxxMat2) : Prop :=
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_3 backend tape (path ++ [3, i.val]) params i (w_1_0, (w_0_0, ((w_2_0 i), ()))) (w_3_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_5 backend tape (path ++ [5, i.val]) params i (w_1_0, (w_0_0, ((w_4_0 i), ()))) (w_5_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_7 backend tape (path ++ [7, i.val]) params i (w_1_0, (w_0_0, ((w_6_0 i), ()))) (w_7_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_9 backend tape (path ++ [9, i.val]) params i (w_1_0, (w_0_0, ((w_8_0 i), ()))) (w_9_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_11 backend tape (path ++ [11, i.val]) params i (w_1_0, (w_0_0, ((w_10_0 i), ()))) (w_11_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_13 backend tape (path ++ [13, i.val]) params i (w_1_0, (w_0_0, ((w_12_0 i), ()))) (w_13_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_15 backend tape (path ++ [15, i.val]) params i (w_1_0, (w_0_0, ((w_14_0 i), ()))) (w_15_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_17 backend tape (path ++ [17, i.val]) params i (w_1_0, (w_0_0, ((w_16_0 i), ()))) (w_17_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_19 backend tape (path ++ [19, i.val]) params i (w_1_0, (w_0_0, ((w_18_0 i), ()))) (w_19_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_21 backend tape (path ++ [21, i.val]) params i (w_1_0, (w_0_0, ((w_20_0 i), ()))) (w_21_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_23 backend tape (path ++ [23, i.val]) params i (w_1_0, (w_0_0, ((w_22_0 i), ()))) (w_23_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_25 backend tape (path ++ [25, i.val]) params i (w_1_0, (w_0_0, ((w_24_0 i), ()))) (w_25_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_27 backend tape (path ++ [27, i.val]) params i (w_1_0, (w_0_0, ((w_26_0 i), ()))) (w_27_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_29 backend tape (path ++ [29, i.val]) params i (w_1_0, (w_0_0, ((w_28_0 i), ()))) (w_29_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_31 backend tape (path ++ [31, i.val]) params i (w_1_0, (w_0_0, ((w_30_0 i), ()))) (w_31_0 i)) ∧
  (16779536) = 16779536 ∧
  (∀ i : Fin 16779536, parallel_generatedRoot_33 backend tape (path ++ [33, i.val]) params i (w_1_0, (w_0_0, ((w_32_0 i), ()))) (w_33_0 i))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (generatedRoot.InputType)) (outputs : (generatedRoot.OutputType)) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.1
  let w_2_0 : Fin 16779536 → mxxMat1 := inputs.2.1
  let w_4_0 : Fin 16779536 → mxxMat1 := inputs.2.2.1
  let w_6_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.1
  let w_8_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.1
  let w_10_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_12_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_16_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_24_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_28_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_30_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_34_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_36_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_40_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_42_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_44_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_50_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_52_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_54_0 : Fin 16779536 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := MxxRuntime.trapdoorPublic w_0_0
  let w_3_0 := witness.w_3_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_11_0 := witness.w_11_0
  let w_13_0 := witness.w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 := witness.w_17_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_23_0 := witness.w_23_0
  let w_25_0 := witness.w_25_0
  let w_27_0 := witness.w_27_0
  let w_29_0 := witness.w_29_0
  let w_31_0 := witness.w_31_0
  let w_33_0 := witness.w_33_0
  let w_35_0 := witness.w_35_0
  let w_37_0 := witness.w_37_0
  let w_39_0 := witness.w_39_0
  let w_41_0 := witness.w_41_0
  let w_43_0 := witness.w_43_0
  let w_45_0 := witness.w_45_0
  let w_47_0 := witness.w_47_0
  let w_49_0 := witness.w_49_0
  let w_51_0 := witness.w_51_0
  let w_53_0 := witness.w_53_0
  let w_55_0 := witness.w_55_0
  generatedRoot.constraints_0 backend tape path params w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 ∧
    generatedRoot.constraints_1 backend tape path params w_0_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (generatedRoot.InputType)) (outputs : (generatedRoot.OutputType)) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend tape path params inputs outputs witness


end Stage_preimages_0_0
