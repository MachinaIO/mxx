import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Stage_decode

structure Params where
  unit : Unit

local notation "mxxNat0" => 2135831716638487324604933616719561441822403870837824617769981942184435508314978270190567457292289
local notation "mxxNat1" => 1942566534251559791942281995946931586214205534210157066971588917963081260105189818369
local notation "mxxNat2" => 1067915858319243662302466808359780720911201935418912308884990971092217754157489135095283728646144

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

/-- The modulus of a matrix type. -/
abbrev modulus_1 : Nat := mxxNat1

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 28
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 131072 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 131072 2 28
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_1 131072 1 28
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 131072 2 56

abbrev sequential_generatedRoot_4.constraints_0 (w_1_0 : Fin 16 → mxxMat0) (w_8_0 : Int) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_8_0 ∧
  w_8_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_8_0 w_9_0 ∧
  outputs = w_10_0

def sequential_generatedRoot_4 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 16 → mxxMat0 := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 : Int := 1
  let w_4_0 : Int := (w_2_0 + w_3_0)
  let w_5_0 : Int := 2
  let w_6_0 : Int := (w_4_0 * w_5_0)
  let w_7_0 : Int := 0
  let w_8_0 : Int := (w_6_0 + w_7_0)
  ∃ (w_9_0 : mxxMat0),
    let w_10_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_9_0
    sequential_generatedRoot_4.constraints_0 w_1_0 w_8_0 w_9_0 w_10_0 outputs

abbrev sequential_sequential_generatedRoot_28_6.constraints_0 (w_1_0 : Fin 131072 → Int) (w_8_0 : Int) (w_2_0 : Int) (w_3_0 : Int) (w_10_0 : Int) (w_12_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 131072 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  w_8_0 ≠ 0 ∧
  w_10_0 ≠ 0 ∧
  outputs = w_12_0

def sequential_sequential_generatedRoot_28_6 (backend : MxxRuntime.BackendContext) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 131072 → Int) × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 131072 → Int := inputs.2.1
  let w_6_0 : Int := inputs.2.2.1
  let w_8_0 : Int := inputs.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := 2
    let w_5_0 : Int := (w_3_0 * w_4_0)
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_9_0 : Int := (w_7_0 / w_8_0)
    let w_10_0 : Int := 2
    let w_11_0 : Int := (w_9_0 % w_10_0)
    let w_12_0 : Int := (w_0_0 + w_11_0)
    sequential_sequential_generatedRoot_28_6.constraints_0 w_1_0 w_8_0 w_2_0 w_3_0 w_10_0 w_12_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_28.Witness where
  w_2_0 : mxxMat1
  w_3_0 : Fin 131072 → Int
  w_6_0 : Int

abbrev sequential_generatedRoot_28.constraints_0 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (w_0_0 : Int) (w_1_0 : mxxMat0) (w_4_0 : Int) (w_5_0 : Int) (w_2_0 : mxxMat1) (w_3_0 : Fin 131072 → Int) (w_6_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 28 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_2_0 ∧
  MxxRuntime.polynomialValues false w_2_0 w_3_0 ∧
  0 ≤ (131072) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_28_6 backend params i_0 i (current, (w_3_0, (w_4_0, (w_5_0, ())))) next) (Int.toNat (131072)) w_0_0 w_6_0 ∧
  outputs = w_6_0

abbrev sequential_generatedRoot_28.body (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat0 × Int × Int × Unit) (outputs : Int) (witness : sequential_generatedRoot_28.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : Int := inputs.2.2.1
  let w_5_0 : Int := inputs.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_6_0 := witness.w_6_0
  sequential_generatedRoot_28.constraints_0 backend params i_0 w_0_0 w_1_0 w_4_0 w_5_0 w_2_0 w_3_0 w_6_0 outputs

def sequential_generatedRoot_28 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat0 × Int × Int × Unit) (outputs : Int) : Prop :=
  ∃ witness : sequential_generatedRoot_28.Witness,
    sequential_generatedRoot_28.body backend params i_0 inputs outputs witness

abbrev sequential_generatedRoot_32.constraints_0 (w_1_0 : Fin 16 → mxxMat0) (w_8_0 : Int) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_8_0 ∧
  w_8_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_8_0 w_9_0 ∧
  outputs = w_10_0

def sequential_generatedRoot_32 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × (Fin 16 → mxxMat0) × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 16 → mxxMat0 := inputs.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 : Int := 1
  let w_4_0 : Int := (w_2_0 + w_3_0)
  let w_5_0 : Int := 2
  let w_6_0 : Int := (w_4_0 * w_5_0)
  let w_7_0 : Int := 1
  let w_8_0 : Int := (w_6_0 + w_7_0)
  ∃ (w_9_0 : mxxMat0),
    let w_10_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_9_0
    sequential_generatedRoot_32.constraints_0 w_1_0 w_8_0 w_9_0 w_10_0 outputs

abbrev sequential_sequential_generatedRoot_46_6.constraints_0 (w_1_0 : Fin 131072 → Int) (w_8_0 : Int) (w_2_0 : Int) (w_3_0 : Int) (w_10_0 : Int) (w_12_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 131072 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  w_8_0 ≠ 0 ∧
  w_10_0 ≠ 0 ∧
  outputs = w_12_0

def sequential_sequential_generatedRoot_46_6 (backend : MxxRuntime.BackendContext) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : Int × (Fin 131072 → Int) × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs.1
  let w_1_0 : Fin 131072 → Int := inputs.2.1
  let w_6_0 : Int := inputs.2.2.1
  let w_8_0 : Int := inputs.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := 2
    let w_5_0 : Int := (w_3_0 * w_4_0)
    let w_7_0 : Int := (w_5_0 + w_6_0)
    let w_9_0 : Int := (w_7_0 / w_8_0)
    let w_10_0 : Int := 2
    let w_11_0 : Int := (w_9_0 % w_10_0)
    let w_12_0 : Int := (w_0_0 + w_11_0)
    sequential_sequential_generatedRoot_46_6.constraints_0 w_1_0 w_8_0 w_2_0 w_3_0 w_10_0 w_12_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_46.Witness where
  w_2_0 : mxxMat1
  w_3_0 : Fin 131072 → Int
  w_6_0 : Int

abbrev sequential_generatedRoot_46.constraints_0 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (w_0_0 : Int) (w_1_0 : mxxMat0) (w_4_0 : Int) (w_5_0 : Int) (w_2_0 : mxxMat1) (w_3_0 : Fin 131072 → Int) (w_6_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 28 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_2_0 ∧
  MxxRuntime.polynomialValues false w_2_0 w_3_0 ∧
  0 ≤ (131072) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_sequential_generatedRoot_46_6 backend params i_0 i (current, (w_3_0, (w_4_0, (w_5_0, ())))) next) (Int.toNat (131072)) w_0_0 w_6_0 ∧
  outputs = w_6_0

abbrev sequential_generatedRoot_46.body (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat0 × Int × Int × Unit) (outputs : Int) (witness : sequential_generatedRoot_46.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : Int := inputs.2.2.1
  let w_5_0 : Int := inputs.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_6_0 := witness.w_6_0
  sequential_generatedRoot_46.constraints_0 backend params i_0 w_0_0 w_1_0 w_4_0 w_5_0 w_2_0 w_3_0 w_6_0 outputs

def sequential_generatedRoot_46 (backend : MxxRuntime.BackendContext) (params : Params) (i_0 : Nat) (inputs : Int × mxxMat0 × Int × Int × Unit) (outputs : Int) : Prop :=
  ∃ witness : sequential_generatedRoot_46.Witness,
    sequential_generatedRoot_46.body backend params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : mxxMat0
  w_4_0 : mxxMat0
  w_6_0 : mxxMat1
  w_8_0 : mxxMat2
  w_9_0 : mxxMat0
  w_11_0 : mxxMat0
  w_21_0 : mxxMat3
  w_22_0 : mxxMat0
  w_28_0 : Int
  w_31_0 : mxxMat0
  w_32_0 : mxxMat0
  w_33_0 : mxxMat2
  w_34_0 : mxxMat0
  w_36_0 : mxxMat0
  w_46_0 : Int

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (params : Params) (w_7_0 : mxxMat4) (w_30_0 : Int) (w_33_0 : mxxMat2) (w_34_0 : mxxMat0) (w_36_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : Int) (w_45_0 : Int) (w_46_0 : Int) (w_47_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 28 ∧
  28 < 56 ∧
  56 ≤ 56 ∧
  MxxRuntime.sliceMatrix w_7_0 0 2 28 56 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 28 ∧
  28 ≤ 28 ∧
  MxxRuntime.sliceMatrix w_33_0 0 1 0 28 w_34_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 28 ∧
  28 ≤ 28 ∧
  MxxRuntime.sliceMatrix w_33_0 1 2 0 28 w_36_0 ∧
  0 ≤ (28) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_46 backend params i (current, (w_43_0, (w_44_0, (w_45_0, ())))) next) (Int.toNat (28)) w_30_0 w_46_0 ∧
  outputs = w_47_0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (params : Params) (w_2_0 : Fin 16 → mxxMat0) (w_5_0 : Fin 131072 → Int) (w_7_0 : mxxMat4) (w_1_0 : Int) (w_3_0 : mxxMat0) (w_4_0 : mxxMat0) (w_6_0 : mxxMat1) (w_8_0 : mxxMat2) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_21_0 : mxxMat3) (w_22_0 : mxxMat0) (w_25_0 : mxxMat0) (w_26_0 : Int) (w_27_0 : Int) (w_28_0 : Int) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) : Prop :=
  MxxRuntime.familyGetStatic w_2_0 (0) w_3_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_4 backend params i (current, (w_2_0, ())) next) (Int.toNat (7)) w_3_0 w_4_0 ∧
  MxxRuntime.polynomialFromValues false w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 28 ∧
  28 ≤ 56 ∧
  MxxRuntime.sliceMatrix w_7_0 0 2 0 28 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 28 ∧
  28 ≤ 28 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 0 28 w_9_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 28 ∧
  28 ≤ 28 ∧
  MxxRuntime.sliceMatrix w_8_0 1 2 0 28 w_11_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 1024 28 w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_21_0 w_22_0 ∧
  0 ≤ (28) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_28 backend params i (current, (w_25_0, (w_26_0, (w_27_0, ())))) next) (Int.toNat (28)) w_1_0 w_28_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (1) w_31_0 ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_generatedRoot_32 backend params i (current, (w_2_0, ())) next) (Int.toNat (7)) w_31_0 w_32_0

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (params : Params) (inputs : (Fin 16 → mxxMat0) × (Fin 131072 → Int) × mxxMat4 × mxxMat1 × Int × Int × Unit) (outputs : Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_2_0 : Fin 16 → mxxMat0 := inputs.1
  let w_5_0 : Fin 131072 → Int := inputs.2.1
  let w_7_0 : mxxMat4 := inputs.2.2.1
  let w_14_0 : mxxMat1 := inputs.2.2.2.1
  let w_16_0 : Int := inputs.2.2.2.2.1
  let w_18_0 : Int := inputs.2.2.2.2.2.1
  let w_0_0 : Int := 0
  let w_1_0 : Int := 0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_6_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_10_0 w_11_0
  let w_13_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1099489607681)] : mxxMat1)
  let w_15_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_13_0 w_14_0
  let w_17_0 : mxxMat1 := MxxRuntime.liftInteger w_16_0
  let w_19_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_17_0 w_18_0
  let w_20_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_15_0 w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_20_0 w_22_0
  let w_24_0 : mxxMat0 := MxxRuntime.matrixSub w_12_0 w_23_0
  let w_25_0 : mxxMat0 := MxxRuntime.matrixSub w_4_0 w_24_0
  let w_26_0 : Int := mxxNat2
  let w_27_0 : Int := mxxNat0
  let w_28_0 := witness.w_28_0
  let w_29_0 : Int := (w_0_0 + w_28_0)
  let w_30_0 : Int := 0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_6_0 w_34_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat0 := MxxRuntime.matrixAdd w_35_0 w_36_0
  let w_38_0 : mxxMat1 := (1 : mxxMat1)
  let w_39_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_13_0 w_38_0
  let w_40_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_39_0 w_19_0
  let w_41_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_40_0 w_22_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_37_0 w_41_0
  let w_43_0 : mxxMat0 := MxxRuntime.matrixSub w_32_0 w_42_0
  let w_44_0 : Int := mxxNat2
  let w_45_0 : Int := mxxNat0
  let w_46_0 := witness.w_46_0
  let w_47_0 : Int := (w_29_0 + w_46_0)
  generatedRoot.constraints_0 backend params w_2_0 w_5_0 w_7_0 w_1_0 w_3_0 w_4_0 w_6_0 w_8_0 w_9_0 w_11_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_31_0 w_32_0 ∧
    generatedRoot.constraints_1 backend params w_7_0 w_30_0 w_33_0 w_34_0 w_36_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (params : Params) (inputs : (Fin 16 → mxxMat0) × (Fin 131072 → Int) × mxxMat4 × mxxMat1 × Int × Int × Unit) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend params inputs outputs witness


end W_Stage_decode
