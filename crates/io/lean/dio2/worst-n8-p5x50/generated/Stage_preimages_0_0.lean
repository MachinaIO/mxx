import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_preimages_0_0

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 54 25

abbrev parallel_generatedRoot_3.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (22011706) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (22011706)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_3 (backend : MxxRuntime.BackendContext) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_3.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_5.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (22011706) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (22011706)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_5.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_7.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (22011706) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (22011706)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_7 (backend : MxxRuntime.BackendContext) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_7.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_9.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (22011706) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (22011706)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_9 (backend : MxxRuntime.BackendContext) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_9.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

abbrev parallel_generatedRoot_11.constraints_0 (backend : MxxRuntime.BackendContext) (w_0_0 : mxxMat0) (w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ (22011706) ∧
  MxxRuntime.preimageRunsDispatched backend w_0_0 w_1_0 w_2_0 (Int.toNat (22011706)) w_3_0 ∧
  outputs = w_3_0

def parallel_generatedRoot_11 (backend : MxxRuntime.BackendContext) (params : Params) (_ : Nat) (inputs : mxxMat0 × MxxRuntime.TrapdoorValue (mxxMat0) Unit × mxxMat1 × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  ∃ (w_3_0 : mxxMat2),
    parallel_generatedRoot_11.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Fin 36 → mxxMat2
  w_5_0 : Fin 36 → mxxMat2
  w_7_0 : Fin 36 → mxxMat2
  w_9_0 : Fin 36 → mxxMat2
  w_11_0 : Fin 36 → mxxMat2

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (params : Params) (w_0_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit) (w_2_0 : Fin 36 → mxxMat1) (w_4_0 : Fin 36 → mxxMat1) (w_6_0 : Fin 36 → mxxMat1) (w_8_0 : Fin 36 → mxxMat1) (w_10_0 : Fin 36 → mxxMat1) (w_1_0 : mxxMat0) (w_3_0 : Fin 36 → mxxMat2) (w_5_0 : Fin 36 → mxxMat2) (w_7_0 : Fin 36 → mxxMat2) (w_9_0 : Fin 36 → mxxMat2) (w_11_0 : Fin 36 → mxxMat2) (outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × Unit) : Prop :=
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_3 backend params i (w_1_0, (w_0_0, ((w_2_0 i), ()))) (w_3_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_5 backend params i (w_1_0, (w_0_0, ((w_4_0 i), ()))) (w_5_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_7 backend params i (w_1_0, (w_0_0, ((w_6_0 i), ()))) (w_7_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_9 backend params i (w_1_0, (w_0_0, ((w_8_0 i), ()))) (w_9_0 i)) ∧
  (36) = 36 ∧
  (∀ i : Fin 36, parallel_generatedRoot_11 backend params i (w_1_0, (w_0_0, ((w_10_0 i), ()))) (w_11_0 i)) ∧
  outputs = (w_3_0, (w_5_0, (w_7_0, (w_9_0, (w_11_0, ())))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (params : Params) (inputs : MxxRuntime.TrapdoorValue (mxxMat0) Unit × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × Unit) (outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : MxxRuntime.TrapdoorValue (mxxMat0) Unit := inputs.1
  let w_2_0 : Fin 36 → mxxMat1 := inputs.2.1
  let w_4_0 : Fin 36 → mxxMat1 := inputs.2.2.1
  let w_6_0 : Fin 36 → mxxMat1 := inputs.2.2.2.1
  let w_8_0 : Fin 36 → mxxMat1 := inputs.2.2.2.2.1
  let w_10_0 : Fin 36 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := MxxRuntime.trapdoorPublic w_0_0
  let w_3_0 := witness.w_3_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_11_0 := witness.w_11_0
  generatedRoot.constraints_0 backend params w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (params : Params) (inputs : MxxRuntime.TrapdoorValue (mxxMat0) Unit × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × (Fin 36 → mxxMat1) × Unit) (outputs : (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × (Fin 36 → mxxMat2) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend params inputs outputs witness


end Stage_preimages_0_0
