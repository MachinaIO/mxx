import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_import_1

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1

abbrev parallel_generatedRoot_6.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_5_0 : mxxMat1) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, (w_5_0, ())))

def parallel_generatedRoot_6 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Int × Int × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : Int := inputs.2.2.1
  let w_4_0 : Int := inputs.2.2.2.1
  let w_3_0 : mxxMat1 := MxxRuntime.liftInteger w_2_0
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  parallel_generatedRoot_6.constraints_0 w_0_0 w_1_0 w_5_0 outputs

abbrev parallel_generatedRoot_11.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_11 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_11.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_14.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_14 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_14.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_15.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_15 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_15.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_16.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_16 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_16.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_17.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_17 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_17.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_18.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_18 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_18.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_19.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_19 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_19.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_20.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_20 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_20.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_21.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_21 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_21.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_22.constraints_0 (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_22 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  parallel_generatedRoot_22.constraints_0 w_0_0 w_1_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_6_0 : Fin 2 → mxxMat0
  w_6_1 : Fin 2 → mxxMat0
  w_6_2 : Fin 2 → mxxMat1
  w_11_0 : Fin 8 → mxxMat0
  w_11_1 : Fin 8 → mxxMat0
  w_14_0 : Fin 4 → mxxMat0
  w_14_1 : Fin 4 → mxxMat0
  w_15_0 : Fin 8 → mxxMat0
  w_15_1 : Fin 8 → mxxMat0
  w_16_0 : Fin 4 → mxxMat0
  w_16_1 : Fin 4 → mxxMat0
  w_17_0 : Fin 8 → mxxMat0
  w_17_1 : Fin 8 → mxxMat0
  w_18_0 : Fin 4 → mxxMat0
  w_18_1 : Fin 4 → mxxMat0
  w_19_0 : Fin 8 → mxxMat0
  w_19_1 : Fin 8 → mxxMat0
  w_20_0 : Fin 4 → mxxMat0
  w_20_1 : Fin 4 → mxxMat0
  w_21_0 : Fin 8 → mxxMat0
  w_21_1 : Fin 8 → mxxMat0
  w_22_0 : Fin 4 → mxxMat0
  w_22_1 : Fin 4 → mxxMat0

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_2_0 : Fin 2 → mxxMat0) (w_3_0 : Fin 2 → mxxMat0) (w_4_0 : Fin 2 → Int) (w_5_0 : Fin 2 → Int) (w_7_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : Fin 8 → mxxMat0) (w_10_0 : Fin 8 → mxxMat0) (w_12_0 : Fin 4 → mxxMat0) (w_13_0 : Fin 4 → mxxMat0) (w_6_0 : Fin 2 → mxxMat0) (w_6_1 : Fin 2 → mxxMat0) (w_6_2 : Fin 2 → mxxMat1) (w_11_0 : Fin 8 → mxxMat0) (w_11_1 : Fin 8 → mxxMat0) (w_14_0 : Fin 4 → mxxMat0) (w_14_1 : Fin 4 → mxxMat0) (w_15_0 : Fin 8 → mxxMat0) (w_15_1 : Fin 8 → mxxMat0) (w_16_0 : Fin 4 → mxxMat0) (w_16_1 : Fin 4 → mxxMat0) (w_17_0 : Fin 8 → mxxMat0) (w_17_1 : Fin 8 → mxxMat0) (w_18_0 : Fin 4 → mxxMat0) (w_18_1 : Fin 4 → mxxMat0) (w_19_0 : Fin 8 → mxxMat0) (w_19_1 : Fin 8 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_20_1 : Fin 4 → mxxMat0) (w_21_0 : Fin 8 → mxxMat0) (w_21_1 : Fin 8 → mxxMat0) (w_22_0 : Fin 4 → mxxMat0) (w_22_1 : Fin 4 → mxxMat0) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_6 tape (path ++ [6, i.val]) params i ((w_2_0 i), ((w_3_0 i), ((w_4_0 i), ((w_5_0 i), ())))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ())))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_11 tape (path ++ [11, i.val]) params i ((w_9_0 i), ((w_10_0 i), ())) ((w_11_0 i), ((w_11_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_14 tape (path ++ [14, i.val]) params i ((w_12_0 i), ((w_13_0 i), ())) ((w_14_0 i), ((w_14_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_15 tape (path ++ [15, i.val]) params i ((w_9_0 i), ((w_10_0 i), ())) ((w_15_0 i), ((w_15_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_16 tape (path ++ [16, i.val]) params i ((w_12_0 i), ((w_13_0 i), ())) ((w_16_0 i), ((w_16_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_17 tape (path ++ [17, i.val]) params i ((w_9_0 i), ((w_10_0 i), ())) ((w_17_0 i), ((w_17_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_18 tape (path ++ [18, i.val]) params i ((w_12_0 i), ((w_13_0 i), ())) ((w_18_0 i), ((w_18_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_19 tape (path ++ [19, i.val]) params i ((w_9_0 i), ((w_10_0 i), ())) ((w_19_0 i), ((w_19_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_20 tape (path ++ [20, i.val]) params i ((w_12_0 i), ((w_13_0 i), ())) ((w_20_0 i), ((w_20_1 i), ()))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_21 tape (path ++ [21, i.val]) params i ((w_9_0 i), ((w_10_0 i), ())) ((w_21_0 i), ((w_21_1 i), ()))) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_22 tape (path ++ [22, i.val]) params i ((w_12_0 i), ((w_13_0 i), ())) ((w_22_0 i), ((w_22_1 i), ()))) ∧
  outputs = (w_0_0, (w_1_0, (w_6_0, (w_6_1, (w_6_2, (w_7_0, (w_8_0, (w_11_0, (w_11_1, (w_14_0, (w_14_1, (w_15_0, (w_15_1, (w_16_0, (w_16_1, (w_17_0, (w_17_1, (w_18_0, (w_18_1, (w_19_0, (w_19_1, (w_20_0, (w_20_1, (w_21_0, (w_21_1, (w_22_0, (w_22_1, ())))))))))))))))))))))))))))

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → Int) × (Fin 2 → Int) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : Fin 2 → mxxMat0 := inputs.2.2.1
  let w_3_0 : Fin 2 → mxxMat0 := inputs.2.2.2.1
  let w_4_0 : Fin 2 → Int := inputs.2.2.2.2.1
  let w_5_0 : Fin 2 → Int := inputs.2.2.2.2.2.1
  let w_7_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_10_0 : Fin 8 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_12_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_13_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_15_0 := witness.w_15_0
  let w_15_1 := witness.w_15_1
  let w_16_0 := witness.w_16_0
  let w_16_1 := witness.w_16_1
  let w_17_0 := witness.w_17_0
  let w_17_1 := witness.w_17_1
  let w_18_0 := witness.w_18_0
  let w_18_1 := witness.w_18_1
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_20_1 := witness.w_20_1
  let w_21_0 := witness.w_21_0
  let w_21_1 := witness.w_21_1
  let w_22_0 := witness.w_22_0
  let w_22_1 := witness.w_22_1
  generatedRoot.constraints_0 tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_14_0 w_14_1 w_15_0 w_15_1 w_16_0 w_16_1 w_17_0 w_17_1 w_18_0 w_18_1 w_19_0 w_19_1 w_20_0 w_20_1 w_21_0 w_21_1 w_22_0 w_22_1 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → Int) × (Fin 2 → Int) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat1) × mxxMat0 × mxxMat0 × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Stage_import_1
