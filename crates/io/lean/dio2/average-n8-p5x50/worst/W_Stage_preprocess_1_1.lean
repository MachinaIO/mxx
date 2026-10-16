import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Stage_preprocess_1_1

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993
local notation "mxxNat1" => 1356938545745524308187737599098762797340022013559604731140326430414605958745
local notation "mxxNat2" => 1583094970036445026219027198948556596896692349152872186330380835483706951869
local notation "mxxNat3" => 36550556566445887949135852474979514307313305075533710697673106717124183540354
local notation "mxxNat4" => 97925267724003222340535519505145373065489256943849140928644983866610518685471

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 25 25

abbrev parallel_parallel_generatedRoot_3_12.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 4 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_3_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(0), (1), (1), (0)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_3_12.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_3_13.constraints_0 (i_0 : Nat) (w_7_0 : Fin 4 → Int) (w_2_0 : Int) (w_5_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : Int) (w_12_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_5_0 w_6_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_2_0 w_8_0 ∧
  outputs = w_12_0

def sequential_parallel_generatedRoot_3_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × Int × (Fin 4 → Int) × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_3_0 : Int := inputs.2.2.1
  let w_7_0 : Fin 4 → Int := inputs.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_1_0 w_4_0
  ∃ (w_6_0 : mxxMat0),
    ∃ (w_8_0 : Int),
      let w_9_0 : Int := 2
      let w_10_0 : Int := (w_8_0 * w_9_0)
      let w_11_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_6_0 w_10_0
      let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_11_0
      sequential_parallel_generatedRoot_3_13.constraints_0 i_0 w_7_0 w_2_0 w_5_0 w_6_0 w_8_0 w_12_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_3.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_12_0 : Fin 4 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1

abbrev parallel_generatedRoot_3.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : Int) (w_12_0 : Fin 4 → Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_22_0 : mxxMat1) (w_31_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_3_0 w_4_0 ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_parallel_generatedRoot_3_12 backend hashModel params i_0 i () (w_12_0 i)) ∧
  0 ≤ (4) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_3_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (4)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  MxxRuntime.centeredRebaseRuns w_17_0 w_18_0 ∧
  outputs = (w_1_0, (w_2_0, (w_22_0, (w_31_0, ()))))

abbrev parallel_generatedRoot_3.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_3.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_3_0 : mxxMat0 := inputs.2.1
  let w_26_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_4_0 := witness.w_4_0
  let w_5_0 : mxxMat0 := MxxRuntime.matrixNeg w_4_0
  let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_1_0
  let w_7_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(mxxNat1)] : mxxMat0)
  let w_9_0 : mxxMat0 := (0 : mxxMat0)
  let w_10_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
  let w_11_0 : Int := -2
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_13_0
  let w_15_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_7_0 w_14_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_3_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_16_0 w_18_0
  let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_6_0 w_19_0
  let w_21_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 : mxxMat1 := MxxRuntime.matrixAdd w_20_0 w_21_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixNeg w_4_0
  let w_24_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_23_0 w_2_0
  let w_25_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_7_0 w_14_0
  let w_27_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_27_0 w_18_0
  let w_29_0 : mxxMat1 := MxxRuntime.matrixAdd w_24_0 w_28_0
  let w_30_0 : mxxMat1 := (0 : mxxMat1)
  let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_29_0 w_30_0
  parallel_generatedRoot_3.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs

def parallel_generatedRoot_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_3.Witness,
    parallel_generatedRoot_3.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_4_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_4_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (1), (3), (3), (5), (5), (7), (7)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_4_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_4_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 8 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_4_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 8 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 8 → Int := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_6_0 : Int := inputs.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  ∃ (w_3_0 : Int),
    let w_4_0 : mxxMat0 := MxxRuntime.liftInteger w_3_0
    let w_7_0 : Int := (w_2_0 * w_6_0)
    let w_8_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_5_0 w_7_0
    ∃ (w_9_0 : mxxMat0),
      let w_10_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_4_0 w_9_0
      let w_11_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_10_0
      sequential_parallel_generatedRoot_4_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_4.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 8 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_4.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 8 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 48, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_3_0 w_4_0 ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_parallel_generatedRoot_4_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_4_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (8)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_4.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_4.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_3_0 : mxxMat0 := inputs.2.1
  let w_25_0 : mxxMat0 := inputs.2.2.1
  let w_1_0 := witness.w_1_0
  let w_2_0 := witness.w_2_0
  let w_4_0 := witness.w_4_0
  let w_5_0 : mxxMat0 := MxxRuntime.matrixNeg w_4_0
  let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_1_0
  let w_7_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat0)
  let w_9_0 : mxxMat0 := (0 : mxxMat0)
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
  let w_12_0 : Int := -1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_13_0
  let w_15_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_7_0 w_14_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_3_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_16_0 w_17_0
  let w_19_0 : mxxMat1 := MxxRuntime.matrixAdd w_6_0 w_18_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_20_0
  let w_22_0 : mxxMat0 := MxxRuntime.matrixNeg w_4_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_22_0 w_2_0
  let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_7_0 w_14_0
  let w_26_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_24_0 w_25_0
  let w_27_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_26_0 w_17_0
  let w_28_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_27_0
  let w_29_0 : mxxMat1 := (0 : mxxMat1)
  let w_30_0 : mxxMat1 := MxxRuntime.matrixAdd w_28_0 w_29_0
  parallel_generatedRoot_4.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_4.Witness,
    parallel_generatedRoot_4.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_33.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_17_0 : mxxMat1) (w_25_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_4_0 ∧
  MxxRuntime.centeredRebaseRuns w_4_0 w_5_0 ∧
  outputs = (w_1_0, (w_2_0, (w_17_0, (w_25_0, ()))))

def parallel_generatedRoot_33 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_3_0 : mxxMat0 := inputs.2.1
  let w_9_0 : Int := inputs.2.2.1
  let w_11_0 : Int := inputs.2.2.2.1
  let w_18_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_1_0 : mxxMat1),
    ∃ (w_2_0 : mxxMat1),
      ∃ (w_4_0 : mxxMat1),
        ∃ (w_5_0 : mxxMat1),
          let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_3_0 w_5_0
          let w_7_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_6_0 w_7_0
          let w_10_0 : mxxMat0 := MxxRuntime.liftInteger w_9_0
          let w_12_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_10_0 w_11_0
          let w_13_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_8_0 w_12_0
          let w_14_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_3_0 w_1_0
          let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
          let w_16_0 : mxxMat1 := (0 : mxxMat1)
          let w_17_0 : mxxMat1 := MxxRuntime.matrixAdd w_15_0 w_16_0
          let w_19_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_18_0 w_5_0
          let w_20_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_19_0 w_7_0
          let w_21_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_20_0 w_12_0
          let w_22_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_3_0 w_2_0
          let w_23_0 : mxxMat1 := MxxRuntime.matrixSub w_21_0 w_22_0
          let w_24_0 : mxxMat1 := (0 : mxxMat1)
          let w_25_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_24_0
          parallel_generatedRoot_33.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_4_0 w_5_0 w_17_0 w_25_0 outputs

abbrev sequential_generatedRoot_66.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_39_0 1 2 0 25 w_66_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_62_0 1 2 0 25 w_69_0 ∧
  outputs = (w_41_0, (w_64_0, (w_67_0, (w_70_0, ()))))

abbrev sequential_generatedRoot_66.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 4 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
  MxxRuntime.familyGetDynamic w_48_0 w_12_0 w_49_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_53_0 0 1 0 25 w_54_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_55_0 w_56_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_53_0 1 2 0 25 w_58_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_62_0 0 1 0 25 w_63_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0

abbrev sequential_generatedRoot_66.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 4 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_27_0 0 1 0 25 w_28_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_28_0 w_29_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_29_0 w_30_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_32_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_33_0 ∧
  MxxRuntime.concatRows w_32_0 w_33_0 w_34_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_27_0 1 2 0 25 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_35_0 w_36_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_36_0 w_37_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_39_0 0 1 0 25 w_40_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 4

abbrev sequential_generatedRoot_66.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 4 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 4 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_generatedRoot_66 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_5_0 : mxxMat2),
        ∃ (w_7_0 : mxxMat1),
          ∃ (w_9_0 : mxxMat1),
            ∃ (w_10_0 : mxxMat2),
              let w_12_0 : Int := (Int.ofNat i_0)
              ∃ (w_13_0 : mxxMat1),
                ∃ (w_14_0 : mxxMat3),
                  ∃ (w_15_0 : mxxMat3),
                    let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_15_0
                    ∃ (w_18_0 : mxxMat1),
                      ∃ (w_20_0 : mxxMat1),
                        ∃ (w_21_0 : mxxMat2),
                          ∃ (w_23_0 : mxxMat1),
                            ∃ (w_24_0 : mxxMat3),
                              ∃ (w_25_0 : mxxMat3),
                                let w_26_0 : mxxMat2 := MxxRuntime.matrixMul w_21_0 w_25_0
                                let w_27_0 : mxxMat2 := MxxRuntime.matrixAdd w_16_0 w_26_0
                                ∃ (w_28_0 : mxxMat1),
                                  ∃ (w_29_0 : mxxMat3),
                                    ∃ (w_30_0 : mxxMat3),
                                      let w_31_0 : mxxMat2 := MxxRuntime.matrixMul w_5_0 w_30_0
                                      ∃ (w_32_0 : mxxMat1),
                                        ∃ (w_33_0 : mxxMat1),
                                          ∃ (w_34_0 : mxxMat2),
                                            ∃ (w_35_0 : mxxMat1),
                                              ∃ (w_36_0 : mxxMat3),
                                                ∃ (w_37_0 : mxxMat3),
                                                  let w_38_0 : mxxMat2 := MxxRuntime.matrixMul w_34_0 w_37_0
                                                  let w_39_0 : mxxMat2 := MxxRuntime.matrixAdd w_31_0 w_38_0
                                                  ∃ (w_40_0 : mxxMat1),
                                                    let w_41_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_40_0
                                                    ∃ (w_44_0 : mxxMat1),
                                                      ∃ (w_45_0 : mxxMat3),
                                                        ∃ (w_46_0 : mxxMat3),
                                                          let w_47_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_46_0
                                                          ∃ (w_49_0 : mxxMat1),
                                                            ∃ (w_50_0 : mxxMat3),
                                                              ∃ (w_51_0 : mxxMat3),
                                                                let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_21_0 w_51_0
                                                                let w_53_0 : mxxMat2 := MxxRuntime.matrixAdd w_47_0 w_52_0
                                                                ∃ (w_54_0 : mxxMat1),
                                                                  ∃ (w_55_0 : mxxMat3),
                                                                    ∃ (w_56_0 : mxxMat3),
                                                                      let w_57_0 : mxxMat2 := MxxRuntime.matrixMul w_5_0 w_56_0
                                                                      ∃ (w_58_0 : mxxMat1),
                                                                        ∃ (w_59_0 : mxxMat3),
                                                                          ∃ (w_60_0 : mxxMat3),
                                                                            let w_61_0 : mxxMat2 := MxxRuntime.matrixMul w_34_0 w_60_0
                                                                            let w_62_0 : mxxMat2 := MxxRuntime.matrixAdd w_57_0 w_61_0
                                                                            ∃ (w_63_0 : mxxMat1),
                                                                              let w_64_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_63_0
                                                                              ∃ (w_66_0 : mxxMat1),
                                                                                let w_67_0 : mxxMat1 := MxxRuntime.matrixAdd w_65_0 w_66_0
                                                                                ∃ (w_69_0 : mxxMat1),
                                                                                  let w_70_0 : mxxMat1 := MxxRuntime.matrixAdd w_68_0 w_69_0
                                                                                  sequential_generatedRoot_66.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_66.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_66.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_66.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev parallel_generatedRoot_75.constraints_2 (backend : MxxRuntime.BackendContext) (w_19_0 : mxxMat1) (w_47_0 : mxxMat1) (w_51_0 : mxxMat2) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat3) (w_55_0 : mxxMat3) (w_64_0 : mxxMat1) (w_65_0 : mxxMat3) (w_66_0 : mxxMat3) (w_68_0 : mxxMat2) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.compactCenteredRebaseRuns 2048 w_54_0 w_55_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_64_0 w_65_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_65_0 w_66_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_68_0 0 1 0 25 w_69_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_51_0 1 2 0 25 w_70_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_68_0 1 2 0 25 w_71_0 ∧
  outputs = (w_52_0, (w_69_0, (w_70_0, (w_71_0, (w_19_0, (w_53_0, (w_47_0, (w_64_0, ()))))))))

abbrev parallel_generatedRoot_75.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_18_0 : ByteArray) (w_23_0 : mxxMat1) (w_26_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_24_0 : mxxMat3) (w_27_0 : mxxMat3) (w_29_0 : mxxMat2) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_32_0 : mxxMat2) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat3) (w_49_0 : mxxMat3) (w_51_0 : mxxMat2) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat3) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_19_0 w_20_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_20_0 w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_26_0 w_27_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_29_0 0 1 0 25 w_30_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_29_0 1 2 0 25 w_31_0 ∧
  MxxRuntime.concatRows w_30_0 w_31_0 w_32_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_34_0 ∧
  MxxRuntime.centeredRebaseRuns w_34_0 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_47_0 w_48_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_48_0 w_49_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_51_0 0 1 0 25 w_52_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_18_0) w_53_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_53_0 w_54_0

abbrev parallel_generatedRoot_75.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_1_0 : Int) (w_5_0 : mxxMat1) (w_11_0 : mxxMat1) (w_18_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_6_0 : mxxMat3) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : mxxMat3) (w_14_0 : mxxMat2) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat2) (w_19_0 : mxxMat1) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_3_0 ∧
  MxxRuntime.concatRows w_2_0 w_3_0 w_4_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_5_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_8_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_9_0 ∧
  MxxRuntime.concatRows w_8_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 25 w_15_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_14_0 1 2 0 25 w_16_0 ∧
  MxxRuntime.concatRows w_15_0 w_16_0 w_17_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_18_0) w_19_0

def parallel_generatedRoot_75 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Int × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_1_0 : Int := inputs.2.1
  let w_5_0 : mxxMat1 := inputs.2.2.1
  let w_11_0 : mxxMat1 := inputs.2.2.2.1
  let w_18_0 : ByteArray := inputs.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_26_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_33_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_39_0 : Int := inputs.2.2.2.2.2.2.2.2.1
  let w_41_0 : Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_57_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_3_0 : mxxMat1),
      ∃ (w_4_0 : mxxMat2),
        ∃ (w_6_0 : mxxMat3),
          let w_7_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_6_0
          ∃ (w_8_0 : mxxMat1),
            ∃ (w_9_0 : mxxMat1),
              ∃ (w_10_0 : mxxMat2),
                ∃ (w_12_0 : mxxMat3),
                  let w_13_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_12_0
                  let w_14_0 : mxxMat2 := MxxRuntime.matrixAdd w_7_0 w_13_0
                  ∃ (w_15_0 : mxxMat1),
                    ∃ (w_16_0 : mxxMat1),
                      ∃ (w_17_0 : mxxMat2),
                        ∃ (w_19_0 : mxxMat1),
                          ∃ (w_20_0 : mxxMat3),
                            ∃ (w_21_0 : mxxMat3),
                              let w_22_0 : mxxMat2 := MxxRuntime.matrixMul w_17_0 w_21_0
                              ∃ (w_24_0 : mxxMat3),
                                let w_25_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_24_0
                                ∃ (w_27_0 : mxxMat3),
                                  let w_28_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_27_0
                                  let w_29_0 : mxxMat2 := MxxRuntime.matrixAdd w_25_0 w_28_0
                                  ∃ (w_30_0 : mxxMat1),
                                    ∃ (w_31_0 : mxxMat1),
                                      ∃ (w_32_0 : mxxMat2),
                                        ∃ (w_34_0 : mxxMat1),
                                          ∃ (w_35_0 : mxxMat1),
                                            let w_36_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_33_0 w_35_0
                                            let w_37_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
                                            let w_38_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_36_0 w_37_0
                                            let w_40_0 : mxxMat0 := MxxRuntime.liftInteger w_39_0
                                            let w_42_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_40_0 w_41_0
                                            let w_43_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_38_0 w_42_0
                                            let w_44_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_33_0 w_19_0
                                            let w_45_0 : mxxMat1 := MxxRuntime.matrixSub w_43_0 w_44_0
                                            let w_46_0 : mxxMat1 := (0 : mxxMat1)
                                            let w_47_0 : mxxMat1 := MxxRuntime.matrixAdd w_45_0 w_46_0
                                            ∃ (w_48_0 : mxxMat3),
                                              ∃ (w_49_0 : mxxMat3),
                                                let w_50_0 : mxxMat2 := MxxRuntime.matrixMul w_32_0 w_49_0
                                                let w_51_0 : mxxMat2 := MxxRuntime.matrixAdd w_22_0 w_50_0
                                                ∃ (w_52_0 : mxxMat1),
                                                  ∃ (w_53_0 : mxxMat1),
                                                    ∃ (w_54_0 : mxxMat3),
                                                      ∃ (w_55_0 : mxxMat3),
                                                        let w_56_0 : mxxMat2 := MxxRuntime.matrixMul w_17_0 w_55_0
                                                        let w_58_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_57_0 w_35_0
                                                        let w_59_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_58_0 w_37_0
                                                        let w_60_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_59_0 w_42_0
                                                        let w_61_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_33_0 w_53_0
                                                        let w_62_0 : mxxMat1 := MxxRuntime.matrixSub w_60_0 w_61_0
                                                        let w_63_0 : mxxMat1 := (0 : mxxMat1)
                                                        let w_64_0 : mxxMat1 := MxxRuntime.matrixAdd w_62_0 w_63_0
                                                        ∃ (w_65_0 : mxxMat3),
                                                          ∃ (w_66_0 : mxxMat3),
                                                            let w_67_0 : mxxMat2 := MxxRuntime.matrixMul w_32_0 w_66_0
                                                            let w_68_0 : mxxMat2 := MxxRuntime.matrixAdd w_56_0 w_67_0
                                                            ∃ (w_69_0 : mxxMat1),
                                                              ∃ (w_70_0 : mxxMat1),
                                                                ∃ (w_71_0 : mxxMat1),
                                                                  parallel_generatedRoot_75.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_5_0 w_11_0 w_18_0 w_2_0 w_3_0 w_4_0 w_6_0 w_8_0 w_9_0 w_10_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_19_0 ∧
    parallel_generatedRoot_75.constraints_1 backend hashModel i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_54_0 ∧
    parallel_generatedRoot_75.constraints_2 backend w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs

abbrev parallel_parallel_generatedRoot_76_4.constraints_0 (w_0_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_5_0 : Fin 16 → mxxMat1) (w_7_0 : Fin 16 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, ()))))

def parallel_parallel_generatedRoot_76_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat1 := inputs.1
  let w_3_0 : Fin 16 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 16 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 16 → mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          parallel_parallel_generatedRoot_76_4.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 outputs

abbrev sequential_parallel_generatedRoot_76_17.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_parallel_generatedRoot_76_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_parallel_generatedRoot_76_17.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_76.Witness where
  w_4_0 : Fin 2 → mxxMat1
  w_4_1 : Fin 2 → mxxMat1
  w_4_2 : Fin 2 → mxxMat1
  w_4_3 : Fin 2 → mxxMat1
  w_5_0 : mxxMat1
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_9_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat1
  w_17_0 : mxxMat1
  w_17_1 : mxxMat1
  w_17_2 : mxxMat1
  w_17_3 : mxxMat1

abbrev parallel_generatedRoot_76.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat1) (w_1_0 : Fin 16 → mxxMat1) (w_2_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_4_0 : Fin 2 → mxxMat1) (w_4_1 : Fin 2 → mxxMat1) (w_4_2 : Fin 2 → mxxMat1) (w_4_3 : Fin 2 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_17_1 : mxxMat1) (w_17_2 : mxxMat1) (w_17_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_76_4 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_4_0 (0) w_5_0 ∧
  MxxRuntime.familyGetStatic w_4_0 (1) w_6_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (1) w_9_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (0) w_11_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (1) w_12_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (0) w_14_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (1) w_15_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_76_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_4_0, (w_4_1, (w_4_2, (w_4_3, ())))))))) next) (Int.toNat (0)) (w_7_0, (w_10_0, (w_13_0, (w_16_0, ())))) (w_17_0, (w_17_1, (w_17_2, (w_17_3, ())))) ∧
  outputs = (w_17_0, (w_17_1, (w_17_2, (w_17_3, ()))))

abbrev parallel_generatedRoot_76.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_76.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat1 := inputs.1
  let w_1_0 : Fin 16 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 16 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 16 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_4_2 := witness.w_4_2
  let w_4_3 := witness.w_4_3
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_11_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_17_1 := witness.w_17_1
  let w_17_2 := witness.w_17_2
  let w_17_3 := witness.w_17_3
  parallel_generatedRoot_76.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs

def parallel_generatedRoot_76 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_76.Witness,
    parallel_generatedRoot_76.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_79.constraints_3 (w_22_0 : mxxMat1) (w_52_0 : mxxMat1) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_69_0 : mxxMat1) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) (w_76_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_73_0 1 2 0 25 w_76_0 ∧
  outputs = (w_57_0, (w_74_0, (w_75_0, (w_76_0, (w_22_0, (w_58_0, (w_52_0, (w_69_0, ()))))))))

abbrev parallel_generatedRoot_79.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_21_0 : ByteArray) (w_39_0 : mxxMat1) (w_40_0 : mxxMat1) (w_52_0 : mxxMat1) (w_53_0 : mxxMat3) (w_54_0 : mxxMat3) (w_56_0 : mxxMat2) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_69_0 : mxxMat1) (w_70_0 : mxxMat3) (w_71_0 : mxxMat3) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_39_0 ∧
  MxxRuntime.centeredRebaseRuns w_39_0 w_40_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_52_0 w_53_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_53_0 w_54_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_56_0 0 1 0 25 w_57_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_21_0) w_58_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_69_0 w_70_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_70_0 w_71_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_73_0 0 1 0 25 w_74_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_56_0 1 2 0 25 w_75_0 ∧
  0 ≤ 1 ∧
  1 < 2

abbrev parallel_generatedRoot_79.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_21_0 : ByteArray) (w_26_0 : Fin 8 → mxxMat1) (w_30_0 : Fin 8 → mxxMat1) (w_6_0 : Int) (w_17_0 : mxxMat2) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat2) (w_22_0 : mxxMat1) (w_23_0 : mxxMat3) (w_24_0 : mxxMat3) (w_27_0 : mxxMat1) (w_28_0 : mxxMat3) (w_31_0 : mxxMat1) (w_32_0 : mxxMat3) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat2) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_17_0 1 2 0 25 w_19_0 ∧
  MxxRuntime.concatRows w_18_0 w_19_0 w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_21_0) w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_22_0 w_23_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_23_0 w_24_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_26_0 w_6_0 w_27_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_27_0 w_28_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_30_0 w_6_0 w_31_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_31_0 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_34_0 0 1 0 25 w_35_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_34_0 1 2 0 25 w_36_0 ∧
  MxxRuntime.concatRows w_35_0 w_36_0 w_37_0

abbrev parallel_generatedRoot_79.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_0_0 : ByteArray) (w_1_0 : Int) (w_5_0 : Fin 8 → mxxMat1) (w_13_0 : Fin 8 → mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_6_0 : Int) (w_7_0 : mxxMat1) (w_8_0 : mxxMat3) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat2) (w_14_0 : mxxMat1) (w_15_0 : mxxMat3) (w_17_0 : mxxMat2) (w_18_0 : mxxMat1) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_3_0 ∧
  MxxRuntime.concatRows w_2_0 w_3_0 w_4_0 ∧
  (2) ≠ 0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_7_0 w_8_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_10_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_11_0 ∧
  MxxRuntime.concatRows w_10_0 w_11_0 w_12_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_6_0 w_14_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_14_0 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_17_0 0 1 0 25 w_18_0 ∧
  0 ≤ 1 ∧
  1 < 2

def parallel_generatedRoot_79 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Int × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_1_0 : Int := inputs.2.1
  let w_5_0 : Fin 8 → mxxMat1 := inputs.2.2.1
  let w_13_0 : Fin 8 → mxxMat1 := inputs.2.2.2.1
  let w_21_0 : ByteArray := inputs.2.2.2.2.1
  let w_26_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_38_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_44_0 : Int := inputs.2.2.2.2.2.2.2.2.1
  let w_46_0 : Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_62_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_3_0 : mxxMat1),
      ∃ (w_4_0 : mxxMat2),
        let w_6_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (2))
        ∃ (w_7_0 : mxxMat1),
          ∃ (w_8_0 : mxxMat3),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_8_0
            ∃ (w_10_0 : mxxMat1),
              ∃ (w_11_0 : mxxMat1),
                ∃ (w_12_0 : mxxMat2),
                  ∃ (w_14_0 : mxxMat1),
                    ∃ (w_15_0 : mxxMat3),
                      let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_12_0 w_15_0
                      let w_17_0 : mxxMat2 := MxxRuntime.matrixAdd w_9_0 w_16_0
                      ∃ (w_18_0 : mxxMat1),
                        ∃ (w_19_0 : mxxMat1),
                          ∃ (w_20_0 : mxxMat2),
                            ∃ (w_22_0 : mxxMat1),
                              ∃ (w_23_0 : mxxMat3),
                                ∃ (w_24_0 : mxxMat3),
                                  let w_25_0 : mxxMat2 := MxxRuntime.matrixMul w_20_0 w_24_0
                                  ∃ (w_27_0 : mxxMat1),
                                    ∃ (w_28_0 : mxxMat3),
                                      let w_29_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_28_0
                                      ∃ (w_31_0 : mxxMat1),
                                        ∃ (w_32_0 : mxxMat3),
                                          let w_33_0 : mxxMat2 := MxxRuntime.matrixMul w_12_0 w_32_0
                                          let w_34_0 : mxxMat2 := MxxRuntime.matrixAdd w_29_0 w_33_0
                                          ∃ (w_35_0 : mxxMat1),
                                            ∃ (w_36_0 : mxxMat1),
                                              ∃ (w_37_0 : mxxMat2),
                                                ∃ (w_39_0 : mxxMat1),
                                                  ∃ (w_40_0 : mxxMat1),
                                                    let w_41_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_38_0 w_40_0
                                                    let w_42_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
                                                    let w_43_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_41_0 w_42_0
                                                    let w_45_0 : mxxMat0 := MxxRuntime.liftInteger w_44_0
                                                    let w_47_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_45_0 w_46_0
                                                    let w_48_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_43_0 w_47_0
                                                    let w_49_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_38_0 w_22_0
                                                    let w_50_0 : mxxMat1 := MxxRuntime.matrixSub w_48_0 w_49_0
                                                    let w_51_0 : mxxMat1 := (0 : mxxMat1)
                                                    let w_52_0 : mxxMat1 := MxxRuntime.matrixAdd w_50_0 w_51_0
                                                    ∃ (w_53_0 : mxxMat3),
                                                      ∃ (w_54_0 : mxxMat3),
                                                        let w_55_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_54_0
                                                        let w_56_0 : mxxMat2 := MxxRuntime.matrixAdd w_25_0 w_55_0
                                                        ∃ (w_57_0 : mxxMat1),
                                                          ∃ (w_58_0 : mxxMat1),
                                                            ∃ (w_59_0 : mxxMat3),
                                                              ∃ (w_60_0 : mxxMat3),
                                                                let w_61_0 : mxxMat2 := MxxRuntime.matrixMul w_20_0 w_60_0
                                                                let w_63_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_62_0 w_40_0
                                                                let w_64_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_63_0 w_42_0
                                                                let w_65_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_64_0 w_47_0
                                                                let w_66_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_38_0 w_58_0
                                                                let w_67_0 : mxxMat1 := MxxRuntime.matrixSub w_65_0 w_66_0
                                                                let w_68_0 : mxxMat1 := (0 : mxxMat1)
                                                                let w_69_0 : mxxMat1 := MxxRuntime.matrixAdd w_67_0 w_68_0
                                                                ∃ (w_70_0 : mxxMat3),
                                                                  ∃ (w_71_0 : mxxMat3),
                                                                    let w_72_0 : mxxMat2 := MxxRuntime.matrixMul w_37_0 w_71_0
                                                                    let w_73_0 : mxxMat2 := MxxRuntime.matrixAdd w_61_0 w_72_0
                                                                    ∃ (w_74_0 : mxxMat1),
                                                                      ∃ (w_75_0 : mxxMat1),
                                                                        ∃ (w_76_0 : mxxMat1),
                                                                          parallel_generatedRoot_79.constraints_0 backend hashModel w_0_0 w_1_0 w_5_0 w_13_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_79.constraints_1 backend hashModel i_0 w_21_0 w_26_0 w_30_0 w_6_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_27_0 w_28_0 w_31_0 w_32_0 w_34_0 w_35_0 w_36_0 w_37_0 ∧
    parallel_generatedRoot_79.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 ∧
    parallel_generatedRoot_79.constraints_3 w_22_0 w_52_0 w_57_0 w_58_0 w_69_0 w_73_0 w_74_0 w_75_0 w_76_0 outputs

abbrev parallel_parallel_generatedRoot_80_4.constraints_0 (w_0_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_5_0 : Fin 16 → mxxMat1) (w_7_0 : Fin 16 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 16 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, ()))))

def parallel_parallel_generatedRoot_80_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat1 := inputs.1
  let w_3_0 : Fin 16 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 16 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 16 → mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          parallel_parallel_generatedRoot_80_4.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 outputs

abbrev sequential_parallel_generatedRoot_80_17.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_parallel_generatedRoot_80_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.1
  let w_6_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 2 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_parallel_generatedRoot_80_17.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_80.Witness where
  w_4_0 : Fin 2 → mxxMat1
  w_4_1 : Fin 2 → mxxMat1
  w_4_2 : Fin 2 → mxxMat1
  w_4_3 : Fin 2 → mxxMat1
  w_5_0 : mxxMat1
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_9_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat1
  w_17_0 : mxxMat1
  w_17_1 : mxxMat1
  w_17_2 : mxxMat1
  w_17_3 : mxxMat1

abbrev parallel_generatedRoot_80.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat1) (w_1_0 : Fin 16 → mxxMat1) (w_2_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_4_0 : Fin 2 → mxxMat1) (w_4_1 : Fin 2 → mxxMat1) (w_4_2 : Fin 2 → mxxMat1) (w_4_3 : Fin 2 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_17_1 : mxxMat1) (w_17_2 : mxxMat1) (w_17_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_80_4 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_4_0 (0) w_5_0 ∧
  MxxRuntime.familyGetStatic w_4_0 (1) w_6_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (1) w_9_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (0) w_11_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (1) w_12_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (0) w_14_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (1) w_15_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_80_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_4_0, (w_4_1, (w_4_2, (w_4_3, ())))))))) next) (Int.toNat (0)) (w_7_0, (w_10_0, (w_13_0, (w_16_0, ())))) (w_17_0, (w_17_1, (w_17_2, (w_17_3, ())))) ∧
  outputs = (w_17_0, (w_17_1, (w_17_2, (w_17_3, ()))))

abbrev parallel_generatedRoot_80.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_80.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 16 → mxxMat1 := inputs.1
  let w_1_0 : Fin 16 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 16 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 16 → mxxMat1 := inputs.2.2.2.1
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_4_2 := witness.w_4_2
  let w_4_3 := witness.w_4_3
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_11_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_14_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_17_1 := witness.w_17_1
  let w_17_2 := witness.w_17_2
  let w_17_3 := witness.w_17_3
  parallel_generatedRoot_80.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs

def parallel_generatedRoot_80 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_80.Witness,
    parallel_generatedRoot_80.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_81_10.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_39_0 1 2 0 25 w_66_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_62_0 1 2 0 25 w_69_0 ∧
  outputs = (w_41_0, (w_64_0, (w_67_0, (w_70_0, ()))))

abbrev sequential_parallel_generatedRoot_81_10.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 8 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
  MxxRuntime.familyGetDynamic w_48_0 w_12_0 w_49_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_49_0 w_50_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_50_0 w_51_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_53_0 0 1 0 25 w_54_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_54_0 w_55_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_55_0 w_56_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_53_0 1 2 0 25 w_58_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_62_0 0 1 0 25 w_63_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0

abbrev sequential_parallel_generatedRoot_81_10.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_27_0 0 1 0 25 w_28_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_28_0 w_29_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_29_0 w_30_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_32_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_33_0 ∧
  MxxRuntime.concatRows w_32_0 w_33_0 w_34_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_27_0 1 2 0 25 w_35_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_35_0 w_36_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_36_0 w_37_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_39_0 0 1 0 25 w_40_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8

abbrev sequential_parallel_generatedRoot_81_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 8 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_parallel_generatedRoot_81_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (3), (5), (7), (9), (11), (13), (15)])
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_5_0 : mxxMat2),
        ∃ (w_7_0 : mxxMat1),
          ∃ (w_9_0 : mxxMat1),
            ∃ (w_10_0 : mxxMat2),
              let w_12_0 : Int := (Int.ofNat i_1)
              ∃ (w_13_0 : mxxMat1),
                ∃ (w_14_0 : mxxMat3),
                  ∃ (w_15_0 : mxxMat3),
                    let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_15_0
                    ∃ (w_18_0 : mxxMat1),
                      ∃ (w_20_0 : mxxMat1),
                        ∃ (w_21_0 : mxxMat2),
                          ∃ (w_23_0 : mxxMat1),
                            ∃ (w_24_0 : mxxMat3),
                              ∃ (w_25_0 : mxxMat3),
                                let w_26_0 : mxxMat2 := MxxRuntime.matrixMul w_21_0 w_25_0
                                let w_27_0 : mxxMat2 := MxxRuntime.matrixAdd w_16_0 w_26_0
                                ∃ (w_28_0 : mxxMat1),
                                  ∃ (w_29_0 : mxxMat3),
                                    ∃ (w_30_0 : mxxMat3),
                                      let w_31_0 : mxxMat2 := MxxRuntime.matrixMul w_5_0 w_30_0
                                      ∃ (w_32_0 : mxxMat1),
                                        ∃ (w_33_0 : mxxMat1),
                                          ∃ (w_34_0 : mxxMat2),
                                            ∃ (w_35_0 : mxxMat1),
                                              ∃ (w_36_0 : mxxMat3),
                                                ∃ (w_37_0 : mxxMat3),
                                                  let w_38_0 : mxxMat2 := MxxRuntime.matrixMul w_34_0 w_37_0
                                                  let w_39_0 : mxxMat2 := MxxRuntime.matrixAdd w_31_0 w_38_0
                                                  ∃ (w_40_0 : mxxMat1),
                                                    let w_41_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_40_0
                                                    ∃ (w_44_0 : mxxMat1),
                                                      ∃ (w_45_0 : mxxMat3),
                                                        ∃ (w_46_0 : mxxMat3),
                                                          let w_47_0 : mxxMat2 := MxxRuntime.matrixMul w_10_0 w_46_0
                                                          ∃ (w_49_0 : mxxMat1),
                                                            ∃ (w_50_0 : mxxMat3),
                                                              ∃ (w_51_0 : mxxMat3),
                                                                let w_52_0 : mxxMat2 := MxxRuntime.matrixMul w_21_0 w_51_0
                                                                let w_53_0 : mxxMat2 := MxxRuntime.matrixAdd w_47_0 w_52_0
                                                                ∃ (w_54_0 : mxxMat1),
                                                                  ∃ (w_55_0 : mxxMat3),
                                                                    ∃ (w_56_0 : mxxMat3),
                                                                      let w_57_0 : mxxMat2 := MxxRuntime.matrixMul w_5_0 w_56_0
                                                                      ∃ (w_58_0 : mxxMat1),
                                                                        ∃ (w_59_0 : mxxMat3),
                                                                          ∃ (w_60_0 : mxxMat3),
                                                                            let w_61_0 : mxxMat2 := MxxRuntime.matrixMul w_34_0 w_60_0
                                                                            let w_62_0 : mxxMat2 := MxxRuntime.matrixAdd w_57_0 w_61_0
                                                                            ∃ (w_63_0 : mxxMat1),
                                                                              let w_64_0 : mxxMat1 := MxxRuntime.matrixAdd w_42_0 w_63_0
                                                                              ∃ (w_66_0 : mxxMat1),
                                                                                let w_67_0 : mxxMat1 := MxxRuntime.matrixAdd w_65_0 w_66_0
                                                                                ∃ (w_69_0 : mxxMat1),
                                                                                  let w_70_0 : mxxMat1 := MxxRuntime.matrixAdd w_68_0 w_69_0
                                                                                  sequential_parallel_generatedRoot_81_10.constraints_0 backend hashModel i_1 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_parallel_generatedRoot_81_10.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_parallel_generatedRoot_81_10.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_parallel_generatedRoot_81_10.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_81.Witness where
  w_10_0 : mxxMat1
  w_10_1 : mxxMat1
  w_10_2 : mxxMat1
  w_10_3 : mxxMat1

abbrev parallel_generatedRoot_81.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_8_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_0_0 : mxxMat1) (w_10_0 : mxxMat1) (w_10_1 : mxxMat1) (w_10_2 : mxxMat1) (w_10_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_81_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, (w_6_0, (w_7_0, (w_8_0, (w_9_0, ()))))))))))))) next) (Int.toNat (8)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, ())))) (w_10_0, (w_10_1, (w_10_2, (w_10_3, ())))) ∧
  outputs = (w_10_0, (w_10_1, (w_10_2, (w_10_3, ()))))

abbrev parallel_generatedRoot_81.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_81.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : ByteArray := inputs.1
  let w_2_0 : mxxMat1 := inputs.2.1
  let w_3_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : Fin 8 → mxxMat1 := inputs.2.2.2.1
  let w_5_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_7_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_8_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_9_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
  let w_10_0 := witness.w_10_0
  let w_10_1 := witness.w_10_1
  let w_10_2 := witness.w_10_2
  let w_10_3 := witness.w_10_3
  parallel_generatedRoot_81.constraints_0 backend hashModel params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs

def parallel_generatedRoot_81 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_81.Witness,
    parallel_generatedRoot_81.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_82.constraints_0 (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  outputs = (w_2_0, (w_5_0, (w_8_0, (w_11_0, ()))))

def parallel_generatedRoot_82 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_3_0 : mxxMat1 := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_2_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_0_0 w_1_0
  let w_4_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  let w_7_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_8_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_6_0 w_7_0
  let w_10_0 : Int := (Int.fmod ((Int.ofNat i_0)) (8))
  let w_11_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_9_0 w_10_0
  parallel_generatedRoot_82.constraints_0 w_2_0 w_5_0 w_8_0 w_11_0 outputs

abbrev sequential_generatedRoot_95.constraints_0 (w_1_0 : Fin 8 → mxxMat1) (w_6_0 : Fin 8 → mxxMat1) (w_10_0 : Fin 8 → mxxMat1) (w_14_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_generatedRoot_95 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.1
  let w_6_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 8 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_generatedRoot_95.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Fin 4 → mxxMat1
  w_3_1 : Fin 4 → mxxMat1
  w_3_2 : Fin 4 → mxxMat1
  w_3_3 : Fin 4 → mxxMat1
  w_4_0 : Fin 8 → mxxMat1
  w_4_1 : Fin 8 → mxxMat1
  w_4_2 : Fin 8 → mxxMat1
  w_4_3 : Fin 8 → mxxMat1
  w_7_0 : mxxMat1
  w_10_0 : mxxMat1
  w_13_0 : mxxMat2
  w_14_0 : mxxMat1
  w_15_0 : mxxMat3
  w_17_0 : mxxMat1
  w_20_0 : mxxMat1
  w_23_0 : mxxMat2
  w_24_0 : mxxMat1
  w_25_0 : mxxMat3
  w_28_0 : mxxMat1
  w_29_0 : mxxMat1
  w_30_0 : mxxMat2
  w_33_0 : Fin 1 → mxxMat1
  w_33_1 : Fin 1 → mxxMat1
  w_33_2 : Fin 1 → mxxMat1
  w_33_3 : Fin 1 → mxxMat1
  w_34_0 : mxxMat1
  w_35_0 : mxxMat3
  w_36_0 : mxxMat3
  w_38_0 : mxxMat1
  w_39_0 : mxxMat3
  w_41_0 : mxxMat1
  w_42_0 : mxxMat3
  w_45_0 : mxxMat1
  w_46_0 : mxxMat1
  w_47_0 : mxxMat2
  w_48_0 : mxxMat1
  w_49_0 : mxxMat3
  w_50_0 : mxxMat3
  w_53_0 : mxxMat1
  w_54_0 : mxxMat1
  w_55_0 : mxxMat1
  w_56_0 : mxxMat3
  w_57_0 : mxxMat3
  w_59_0 : mxxMat1
  w_60_0 : mxxMat3
  w_61_0 : mxxMat3
  w_64_0 : mxxMat1
  w_65_0 : mxxMat1
  w_66_0 : mxxMat1
  w_66_1 : mxxMat1
  w_66_2 : mxxMat1
  w_66_3 : mxxMat1
  w_69_0 : mxxMat1
  w_70_0 : mxxMat1
  w_71_0 : mxxMat1
  w_72_0 : mxxMat1
  w_75_0 : Fin 16 → mxxMat1
  w_75_1 : Fin 16 → mxxMat1
  w_75_2 : Fin 16 → mxxMat1
  w_75_3 : Fin 16 → mxxMat1
  w_75_4 : Fin 16 → mxxMat1
  w_75_5 : Fin 16 → mxxMat1
  w_75_6 : Fin 16 → mxxMat1
  w_75_7 : Fin 16 → mxxMat1
  w_76_0 : Fin 8 → mxxMat1
  w_76_1 : Fin 8 → mxxMat1
  w_76_2 : Fin 8 → mxxMat1
  w_76_3 : Fin 8 → mxxMat1
  w_79_0 : Fin 16 → mxxMat1
  w_79_1 : Fin 16 → mxxMat1
  w_79_2 : Fin 16 → mxxMat1
  w_79_3 : Fin 16 → mxxMat1
  w_79_4 : Fin 16 → mxxMat1
  w_79_5 : Fin 16 → mxxMat1
  w_79_6 : Fin 16 → mxxMat1
  w_79_7 : Fin 16 → mxxMat1
  w_80_0 : Fin 8 → mxxMat1
  w_80_1 : Fin 8 → mxxMat1
  w_80_2 : Fin 8 → mxxMat1
  w_80_3 : Fin 8 → mxxMat1
  w_81_0 : Fin 8 → mxxMat1
  w_81_1 : Fin 8 → mxxMat1
  w_81_2 : Fin 8 → mxxMat1
  w_81_3 : Fin 8 → mxxMat1
  w_82_0 : Fin 8 → mxxMat1
  w_82_1 : Fin 8 → mxxMat1
  w_82_2 : Fin 8 → mxxMat1
  w_82_3 : Fin 8 → mxxMat1
  w_83_0 : mxxMat1
  w_84_0 : mxxMat1
  w_86_0 : mxxMat1
  w_87_0 : mxxMat1
  w_89_0 : mxxMat1
  w_90_0 : mxxMat1
  w_92_0 : mxxMat1
  w_93_0 : mxxMat1
  w_95_0 : mxxMat1
  w_95_1 : mxxMat1
  w_95_2 : mxxMat1
  w_95_3 : mxxMat1

abbrev generatedRoot.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_6_0 : ByteArray) (w_67_0 : Fin 16 → Int) (w_68_0 : Fin 16 → Int) (w_73_0 : Fin 16 → Int) (w_74_0 : Fin 16 → Int) (w_77_0 : Fin 16 → Int) (w_78_0 : Fin 16 → Int) (w_2_0 : mxxMat0) (w_3_0 : Fin 4 → mxxMat1) (w_3_1 : Fin 4 → mxxMat1) (w_3_2 : Fin 4 → mxxMat1) (w_3_3 : Fin 4 → mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_4_1 : Fin 8 → mxxMat1) (w_4_2 : Fin 8 → mxxMat1) (w_4_3 : Fin 8 → mxxMat1) (w_33_2 : Fin 1 → mxxMat1) (w_33_3 : Fin 1 → mxxMat1) (w_66_0 : mxxMat1) (w_66_1 : mxxMat1) (w_66_2 : mxxMat1) (w_66_3 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_72_0 : mxxMat1) (w_75_0 : Fin 16 → mxxMat1) (w_75_1 : Fin 16 → mxxMat1) (w_75_2 : Fin 16 → mxxMat1) (w_75_3 : Fin 16 → mxxMat1) (w_75_4 : Fin 16 → mxxMat1) (w_75_5 : Fin 16 → mxxMat1) (w_75_6 : Fin 16 → mxxMat1) (w_75_7 : Fin 16 → mxxMat1) (w_76_0 : Fin 8 → mxxMat1) (w_76_1 : Fin 8 → mxxMat1) (w_76_2 : Fin 8 → mxxMat1) (w_76_3 : Fin 8 → mxxMat1) (w_79_0 : Fin 16 → mxxMat1) (w_79_1 : Fin 16 → mxxMat1) (w_79_2 : Fin 16 → mxxMat1) (w_79_3 : Fin 16 → mxxMat1) (w_79_4 : Fin 16 → mxxMat1) (w_79_5 : Fin 16 → mxxMat1) (w_79_6 : Fin 16 → mxxMat1) (w_79_7 : Fin 16 → mxxMat1) (w_80_0 : Fin 8 → mxxMat1) (w_80_1 : Fin 8 → mxxMat1) (w_80_2 : Fin 8 → mxxMat1) (w_80_3 : Fin 8 → mxxMat1) (w_81_0 : Fin 8 → mxxMat1) (w_81_1 : Fin 8 → mxxMat1) (w_81_2 : Fin 8 → mxxMat1) (w_81_3 : Fin 8 → mxxMat1) (w_82_0 : Fin 8 → mxxMat1) (w_82_1 : Fin 8 → mxxMat1) (w_82_2 : Fin 8 → mxxMat1) (w_82_3 : Fin 8 → mxxMat1) (w_83_0 : mxxMat1) (w_84_0 : mxxMat1) (w_85_0 : mxxMat1) (w_86_0 : mxxMat1) (w_87_0 : mxxMat1) (w_88_0 : mxxMat1) (w_89_0 : mxxMat1) (w_90_0 : mxxMat1) (w_91_0 : mxxMat1) (w_92_0 : mxxMat1) (w_93_0 : mxxMat1) (w_94_0 : mxxMat1) (w_95_0 : mxxMat1) (w_95_1 : mxxMat1) (w_95_2 : mxxMat1) (w_95_3 : mxxMat1) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) : Prop :=
  (16) = 16 ∧
  (∀ i : Fin 16, parallel_generatedRoot_75 backend hashModel params i (w_6_0, ((w_68_0 i), (w_69_0, (w_70_0, (w_0_0, (w_71_0, (w_72_0, (w_1_0, ((w_73_0 i), ((w_74_0 i), (w_2_0, ()))))))))))) ((w_75_0 i), ((w_75_1 i), ((w_75_2 i), ((w_75_3 i), ((w_75_4 i), ((w_75_5 i), ((w_75_6 i), ((w_75_7 i), ()))))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_76 backend hashModel params i (w_75_0, (w_75_1, (w_75_2, (w_75_3, ())))) ((w_76_0 i), ((w_76_1 i), ((w_76_2 i), ((w_76_3 i), ()))))) ∧
  (16) = 16 ∧
  (∀ i : Fin 16, parallel_generatedRoot_79 backend hashModel params i (w_6_0, ((w_67_0 i), (w_76_0, (w_76_2, (w_0_0, (w_76_1, (w_76_3, (w_1_0, ((w_77_0 i), ((w_78_0 i), (w_2_0, ()))))))))))) ((w_79_0 i), ((w_79_1 i), ((w_79_2 i), ((w_79_3 i), ((w_79_4 i), ((w_79_5 i), ((w_79_6 i), ((w_79_7 i), ()))))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_80 backend hashModel params i (w_79_0, (w_79_1, (w_79_2, (w_79_3, ())))) ((w_80_0 i), ((w_80_1 i), ((w_80_2 i), ((w_80_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_81 backend hashModel params i (w_6_0, ((w_80_0 i), ((w_80_2 i), (w_4_0, ((w_80_1 i), ((w_80_3 i), (w_4_2, (w_4_1, (w_4_3, ()))))))))) ((w_81_0 i), ((w_81_1 i), ((w_81_2 i), ((w_81_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_82 backend hashModel params i ((w_81_0 i), ((w_81_1 i), ((w_81_2 i), ((w_81_3 i), ())))) ((w_82_0 i), ((w_82_1 i), ((w_82_2 i), ((w_82_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_82_0 (0) w_83_0 ∧
  MxxRuntime.familyGetStatic w_82_0 (1) w_84_0 ∧
  MxxRuntime.familyGetStatic w_82_1 (0) w_86_0 ∧
  MxxRuntime.familyGetStatic w_82_1 (1) w_87_0 ∧
  MxxRuntime.familyGetStatic w_82_2 (0) w_89_0 ∧
  MxxRuntime.familyGetStatic w_82_2 (1) w_90_0 ∧
  MxxRuntime.familyGetStatic w_82_3 (0) w_92_0 ∧
  MxxRuntime.familyGetStatic w_82_3 (1) w_93_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_95 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_82_0, (w_82_1, (w_82_2, (w_82_3, ())))))))) next) (Int.toNat (6)) (w_85_0, (w_88_0, (w_91_0, (w_94_0, ())))) (w_95_0, (w_95_1, (w_95_2, (w_95_3, ())))) ∧
  outputs = (w_3_0, (w_3_1, (w_3_2, (w_3_3, (w_4_0, (w_4_1, (w_4_2, (w_4_3, (w_66_0, (w_66_1, (w_66_2, (w_66_3, (w_95_0, (w_95_1, (w_95_2, (w_95_3, (w_75_6, (w_75_7, (w_79_6, (w_79_7, (w_33_2, (w_33_3, ()))))))))))))))))))))))

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_6_0 : ByteArray) (w_3_0 : Fin 4 → mxxMat1) (w_3_1 : Fin 4 → mxxMat1) (w_3_2 : Fin 4 → mxxMat1) (w_3_3 : Fin 4 → mxxMat1) (w_5_0 : mxxMat1) (w_33_1 : Fin 1 → mxxMat1) (w_33_3 : Fin 1 → mxxMat1) (w_52_0 : mxxMat2) (w_53_0 : mxxMat1) (w_54_0 : mxxMat1) (w_55_0 : mxxMat1) (w_56_0 : mxxMat3) (w_57_0 : mxxMat3) (w_59_0 : mxxMat1) (w_60_0 : mxxMat3) (w_61_0 : mxxMat3) (w_63_0 : mxxMat2) (w_64_0 : mxxMat1) (w_65_0 : mxxMat1) (w_66_0 : mxxMat1) (w_66_1 : mxxMat1) (w_66_2 : mxxMat1) (w_66_3 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (w_72_0 : mxxMat1) : Prop :=
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 1 2 0 25 w_54_0 ∧
  MxxRuntime.familyGetStatic w_33_1 (0) w_55_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_55_0 w_56_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_56_0 w_57_0 ∧
  MxxRuntime.familyGetStatic w_33_3 (0) w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_59_0 w_60_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_60_0 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_63_0 0 1 0 25 w_64_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_63_0 1 2 0 25 w_65_0 ∧
  0 ≤ (4) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_66 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_6_0, (w_53_0, (w_54_0, (w_3_0, (w_64_0, (w_65_0, (w_3_2, (w_3_1, (w_3_3, ()))))))))))))) next) (Int.toNat (4)) (w_5_0, (w_5_0, (w_5_0, (w_5_0, ())))) (w_66_0, (w_66_1, (w_66_2, (w_66_3, ())))) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_6_0) w_69_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_6_0) w_70_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_6_0) w_71_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_6_0) w_72_0

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_6_0 : ByteArray) (w_33_2 : Fin 1 → mxxMat1) (w_34_0 : mxxMat1) (w_35_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat1) (w_39_0 : mxxMat3) (w_41_0 : mxxMat1) (w_42_0 : mxxMat3) (w_44_0 : mxxMat2) (w_45_0 : mxxMat1) (w_46_0 : mxxMat1) (w_47_0 : mxxMat2) (w_48_0 : mxxMat1) (w_49_0 : mxxMat3) (w_50_0 : mxxMat3) (w_52_0 : mxxMat2) (w_53_0 : mxxMat1) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_34_0 w_35_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_35_0 w_36_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_6_0) w_38_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_38_0 w_39_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_6_0) w_41_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_41_0 w_42_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 0 25 w_45_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_44_0 1 2 0 25 w_46_0 ∧
  MxxRuntime.concatRows w_45_0 w_46_0 w_47_0 ∧
  MxxRuntime.familyGetStatic w_33_2 (0) w_48_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_48_0 w_49_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_49_0 w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 25 w_53_0 ∧
  0 ≤ 1

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_6_0 : ByteArray) (w_31_0 : Fin 1 → Int) (w_32_0 : Fin 1 → Int) (w_2_0 : mxxMat0) (w_3_0 : Fin 4 → mxxMat1) (w_3_1 : Fin 4 → mxxMat1) (w_3_2 : Fin 4 → mxxMat1) (w_3_3 : Fin 4 → mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_4_1 : Fin 8 → mxxMat1) (w_4_2 : Fin 8 → mxxMat1) (w_4_3 : Fin 8 → mxxMat1) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat2) (w_14_0 : mxxMat1) (w_15_0 : mxxMat3) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : mxxMat1) (w_23_0 : mxxMat2) (w_24_0 : mxxMat1) (w_25_0 : mxxMat3) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat1) (w_30_0 : mxxMat2) (w_33_0 : Fin 1 → mxxMat1) (w_33_1 : Fin 1 → mxxMat1) (w_33_2 : Fin 1 → mxxMat1) (w_33_3 : Fin 1 → mxxMat1) (w_34_0 : mxxMat1) : Prop :=
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_3 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_3_0 i), ((w_3_1 i), ((w_3_2 i), ((w_3_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_4 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_6_0) w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_6_0) w_10_0 ∧
  MxxRuntime.concatRows w_9_0 w_12_0 w_13_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_6_0) w_14_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_14_0 w_15_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_6_0) w_17_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_6_0) w_20_0 ∧
  MxxRuntime.concatRows w_19_0 w_22_0 w_23_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_6_0) w_24_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_27_0 0 1 0 25 w_28_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_27_0 1 2 0 25 w_29_0 ∧
  MxxRuntime.concatRows w_28_0 w_29_0 w_30_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_33 backend hashModel params i (w_0_0, (w_1_0, ((w_31_0 i), ((w_32_0 i), (w_2_0, ()))))) ((w_33_0 i), ((w_33_1 i), ((w_33_2 i), ((w_33_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_33_0 (0) w_34_0

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × mxxMat0 × ByteArray × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_6_0 : ByteArray := inputs.2.2.1
  let w_31_0 : Fin 1 → Int := inputs.2.2.2.1
  let w_32_0 : Fin 1 → Int := inputs.2.2.2.2.1
  let w_67_0 : Fin 16 → Int := inputs.2.2.2.2.2.1
  let w_68_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.1
  let w_73_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.1
  let w_74_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_77_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_78_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : mxxMat0 := (1 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_3_1 := witness.w_3_1
  let w_3_2 := witness.w_3_2
  let w_3_3 := witness.w_3_3
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_4_2 := witness.w_4_2
  let w_4_3 := witness.w_4_3
  let w_5_0 : mxxMat1 := (0 : mxxMat1)
  let w_7_0 := witness.w_7_0
  let w_8_0 : Int := 2
  let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : Int := 2
  let w_12_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_13_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : Int := 2
  let w_19_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : Int := 2
  let w_22_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_20_0 w_21_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat2 := MxxRuntime.matrixMul w_23_0 w_25_0
  let w_27_0 : mxxMat2 := MxxRuntime.matrixAdd w_16_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_33_0 := witness.w_33_0
  let w_33_1 := witness.w_33_1
  let w_33_2 := witness.w_33_2
  let w_33_3 := witness.w_33_3
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat2 := MxxRuntime.matrixMul w_30_0 w_36_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat2 := MxxRuntime.matrixMul w_13_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 := witness.w_42_0
  let w_43_0 : mxxMat2 := MxxRuntime.matrixMul w_23_0 w_42_0
  let w_44_0 : mxxMat2 := MxxRuntime.matrixAdd w_40_0 w_43_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 := witness.w_50_0
  let w_51_0 : mxxMat2 := MxxRuntime.matrixMul w_47_0 w_50_0
  let w_52_0 : mxxMat2 := MxxRuntime.matrixAdd w_37_0 w_51_0
  let w_53_0 := witness.w_53_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMul w_30_0 w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat2 := MxxRuntime.matrixMul w_47_0 w_61_0
  let w_63_0 : mxxMat2 := MxxRuntime.matrixAdd w_58_0 w_62_0
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_66_0 := witness.w_66_0
  let w_66_1 := witness.w_66_1
  let w_66_2 := witness.w_66_2
  let w_66_3 := witness.w_66_3
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 := witness.w_72_0
  let w_75_0 := witness.w_75_0
  let w_75_1 := witness.w_75_1
  let w_75_2 := witness.w_75_2
  let w_75_3 := witness.w_75_3
  let w_75_4 := witness.w_75_4
  let w_75_5 := witness.w_75_5
  let w_75_6 := witness.w_75_6
  let w_75_7 := witness.w_75_7
  let w_76_0 := witness.w_76_0
  let w_76_1 := witness.w_76_1
  let w_76_2 := witness.w_76_2
  let w_76_3 := witness.w_76_3
  let w_79_0 := witness.w_79_0
  let w_79_1 := witness.w_79_1
  let w_79_2 := witness.w_79_2
  let w_79_3 := witness.w_79_3
  let w_79_4 := witness.w_79_4
  let w_79_5 := witness.w_79_5
  let w_79_6 := witness.w_79_6
  let w_79_7 := witness.w_79_7
  let w_80_0 := witness.w_80_0
  let w_80_1 := witness.w_80_1
  let w_80_2 := witness.w_80_2
  let w_80_3 := witness.w_80_3
  let w_81_0 := witness.w_81_0
  let w_81_1 := witness.w_81_1
  let w_81_2 := witness.w_81_2
  let w_81_3 := witness.w_81_3
  let w_82_0 := witness.w_82_0
  let w_82_1 := witness.w_82_1
  let w_82_2 := witness.w_82_2
  let w_82_3 := witness.w_82_3
  let w_83_0 := witness.w_83_0
  let w_84_0 := witness.w_84_0
  let w_85_0 : mxxMat1 := MxxRuntime.matrixAdd w_83_0 w_84_0
  let w_86_0 := witness.w_86_0
  let w_87_0 := witness.w_87_0
  let w_88_0 : mxxMat1 := MxxRuntime.matrixAdd w_86_0 w_87_0
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 : mxxMat1 := MxxRuntime.matrixAdd w_89_0 w_90_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 : mxxMat1 := MxxRuntime.matrixAdd w_92_0 w_93_0
  let w_95_0 := witness.w_95_0
  let w_95_1 := witness.w_95_1
  let w_95_2 := witness.w_95_2
  let w_95_3 := witness.w_95_3
  generatedRoot.constraints_0 backend hashModel params w_0_0 w_1_0 w_6_0 w_31_0 w_32_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_33_0 w_33_1 w_33_2 w_33_3 w_34_0 ∧
    generatedRoot.constraints_1 backend hashModel w_6_0 w_33_2 w_34_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 ∧
    generatedRoot.constraints_2 backend hashModel params w_6_0 w_3_0 w_3_1 w_3_2 w_3_3 w_5_0 w_33_1 w_33_3 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_64_0 w_65_0 w_66_0 w_66_1 w_66_2 w_66_3 w_69_0 w_70_0 w_71_0 w_72_0 ∧
    generatedRoot.constraints_3 backend hashModel params w_0_0 w_1_0 w_6_0 w_67_0 w_68_0 w_73_0 w_74_0 w_77_0 w_78_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_33_2 w_33_3 w_66_0 w_66_1 w_66_2 w_66_3 w_69_0 w_70_0 w_71_0 w_72_0 w_75_0 w_75_1 w_75_2 w_75_3 w_75_4 w_75_5 w_75_6 w_75_7 w_76_0 w_76_1 w_76_2 w_76_3 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_79_6 w_79_7 w_80_0 w_80_1 w_80_2 w_80_3 w_81_0 w_81_1 w_81_2 w_81_3 w_82_0 w_82_1 w_82_2 w_82_3 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_95_1 w_95_2 w_95_3 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × mxxMat0 × ByteArray × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × Unit) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end W_Stage_preprocess_1_1
