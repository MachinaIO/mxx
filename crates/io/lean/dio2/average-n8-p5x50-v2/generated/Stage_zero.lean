import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_zero

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54

abbrev parallel_generatedRoot_1.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_1 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_1.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_2.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_2 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_2.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_3.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_3 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_3.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_4.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_4 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_4.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_5.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_5 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_5.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_6.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_6 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_6.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_7.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_7 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_7.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_8.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_8 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_8.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_9.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_9 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_9.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_10.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_10 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_10.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_11.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_11 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_11.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_12.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_12 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_12.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_13.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_13 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_13.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_14.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_14 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_14.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_15.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_15 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_15.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_16.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_16 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_16.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_17.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_17 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_17.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_18.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_18 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_18.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_19.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_19 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_19.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_20.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_20 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_20.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_21.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_21 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_21.constraints_0 w_0_0 outputs

abbrev parallel_generatedRoot_22.constraints_0 (w_0_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  outputs = w_0_0

def parallel_generatedRoot_22 (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (_ : Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  parallel_generatedRoot_22.constraints_0 w_0_0 outputs

abbrev generatedRoot.OutputType : Type :=
  mxxMat1 × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 1 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 8 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 2 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : Fin 1 → mxxMat0
  w_2_0 : Fin 1 → mxxMat0
  w_3_0 : Fin 1 → mxxMat0
  w_4_0 : Fin 1 → mxxMat0
  w_5_0 : Fin 1 → mxxMat0
  w_6_0 : Fin 1 → mxxMat0
  w_7_0 : Fin 1 → mxxMat0
  w_8_0 : Fin 1 → mxxMat0
  w_9_0 : Fin 1 → mxxMat0
  w_10_0 : Fin 1 → mxxMat0
  w_11_0 : Fin 1 → mxxMat0
  w_12_0 : Fin 1 → mxxMat0
  w_13_0 : Fin 1 → mxxMat0
  w_14_0 : Fin 1 → mxxMat0
  w_15_0 : Fin 8 → mxxMat0
  w_16_0 : Fin 8 → mxxMat0
  w_17_0 : Fin 4 → mxxMat0
  w_18_0 : Fin 4 → mxxMat0
  w_19_0 : Fin 2 → mxxMat0
  w_20_0 : Fin 2 → mxxMat0
  w_21_0 : Fin 4 → mxxMat0
  w_22_0 : Fin 4 → mxxMat0

abbrev generatedRoot.constraints_1 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat1) (w_1_0 : Fin 1 → mxxMat0) (w_2_0 : Fin 1 → mxxMat0) (w_3_0 : Fin 1 → mxxMat0) (w_4_0 : Fin 1 → mxxMat0) (w_5_0 : Fin 1 → mxxMat0) (w_6_0 : Fin 1 → mxxMat0) (w_7_0 : Fin 1 → mxxMat0) (w_8_0 : Fin 1 → mxxMat0) (w_9_0 : Fin 1 → mxxMat0) (w_10_0 : Fin 1 → mxxMat0) (w_11_0 : Fin 1 → mxxMat0) (w_12_0 : Fin 1 → mxxMat0) (w_13_0 : Fin 1 → mxxMat0) (w_14_0 : Fin 1 → mxxMat0) (w_15_0 : Fin 8 → mxxMat0) (w_16_0 : Fin 8 → mxxMat0) (w_17_0 : Fin 4 → mxxMat0) (w_18_0 : Fin 4 → mxxMat0) (w_19_0 : Fin 2 → mxxMat0) (w_20_0 : Fin 2 → mxxMat0) (w_21_0 : Fin 4 → mxxMat0) (w_22_0 : Fin 4 → mxxMat0) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : mxxMat0) (outputs : (generatedRoot.OutputType)) : Prop :=
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_17 tape (path ++ [17, i.val]) params i () (w_17_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_18 tape (path ++ [18, i.val]) params i () (w_18_0 i)) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_19 tape (path ++ [19, i.val]) params i () (w_19_0 i)) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_20 tape (path ++ [20, i.val]) params i () (w_20_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_21 tape (path ++ [21, i.val]) params i () (w_21_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_22 tape (path ++ [22, i.val]) params i () (w_22_0 i)) ∧
  outputs = (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, (w_6_0, (w_7_0, (w_8_0, (w_9_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_18_0, (w_19_0, (w_20_0, (w_21_0, (w_22_0, (w_23_0, (w_24_0, (w_25_0, (w_26_0, ())))))))))))))))))))))))))))

abbrev generatedRoot.constraints_0 (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_1_0 : Fin 1 → mxxMat0) (w_2_0 : Fin 1 → mxxMat0) (w_3_0 : Fin 1 → mxxMat0) (w_4_0 : Fin 1 → mxxMat0) (w_5_0 : Fin 1 → mxxMat0) (w_6_0 : Fin 1 → mxxMat0) (w_7_0 : Fin 1 → mxxMat0) (w_8_0 : Fin 1 → mxxMat0) (w_9_0 : Fin 1 → mxxMat0) (w_10_0 : Fin 1 → mxxMat0) (w_11_0 : Fin 1 → mxxMat0) (w_12_0 : Fin 1 → mxxMat0) (w_13_0 : Fin 1 → mxxMat0) (w_14_0 : Fin 1 → mxxMat0) (w_15_0 : Fin 8 → mxxMat0) (w_16_0 : Fin 8 → mxxMat0) : Prop :=
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_1 tape (path ++ [1, i.val]) params i () (w_1_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_2 tape (path ++ [2, i.val]) params i () (w_2_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_3 tape (path ++ [3, i.val]) params i () (w_3_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_4 tape (path ++ [4, i.val]) params i () (w_4_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_5 tape (path ++ [5, i.val]) params i () (w_5_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_6 tape (path ++ [6, i.val]) params i () (w_6_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_7 tape (path ++ [7, i.val]) params i () (w_7_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_8 tape (path ++ [8, i.val]) params i () (w_8_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_9 tape (path ++ [9, i.val]) params i () (w_9_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_10 tape (path ++ [10, i.val]) params i () (w_10_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_11 tape (path ++ [11, i.val]) params i () (w_11_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_12 tape (path ++ [12, i.val]) params i () (w_12_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_13 tape (path ++ [13, i.val]) params i () (w_13_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_14 tape (path ++ [14, i.val]) params i () (w_14_0 i)) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_15 tape (path ++ [15, i.val]) params i () (w_15_0 i)) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_16 tape (path ++ [16, i.val]) params i () (w_16_0 i))

abbrev generatedRoot.body (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : (generatedRoot.OutputType)) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat0 := (0 : mxxMat0)
  let w_24_0 : mxxMat0 := (0 : mxxMat0)
  let w_25_0 : mxxMat0 := (0 : mxxMat0)
  let w_26_0 : mxxMat0 := (0 : mxxMat0)
  generatedRoot.constraints_0 tape path params w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 ∧
    generatedRoot.constraints_1 tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (_ : Unit) (outputs : (generatedRoot.OutputType)) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params () outputs witness


end Stage_zero
