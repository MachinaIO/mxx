import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_preprocess_0_0

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993
local notation "mxxNat1" => 1356938545745524308187737599098762797340022013559604731140326430414605958745
local notation "mxxNat2" => 1583094970036445026219027198948556596896692349152872186330380835483706951869
local notation "mxxNat3" => 29270270146341990167195247313302370525046749248022274291861926128744555509284
local notation "mxxNat4" => 60863387405588526063623438538454545359885565288667817947515354790518700822262

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 2 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 25 25
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 1 112500
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 8 1 4500
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix modulus_0 8 1 22500
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix modulus_0 8 1 900

abbrev parallel_parallel_generatedRoot_3_12.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 4 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_3_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (0), (1), (0)])
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

abbrev parallel_parallel_generatedRoot_5_12.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_5_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (1), (3), (3), (5), (5), (7), (7)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_5_12.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_5_13.constraints_0 (i_0 : Nat) (w_7_0 : Fin 8 → Int) (w_2_0 : Int) (w_5_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : Int) (w_12_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_5_0 w_6_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_2_0 w_8_0 ∧
  outputs = w_12_0

def sequential_parallel_generatedRoot_5_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × Int × (Fin 8 → Int) × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_3_0 : Int := inputs.2.2.1
  let w_7_0 : Fin 8 → Int := inputs.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_1_0 w_4_0
  ∃ (w_6_0 : mxxMat0),
    ∃ (w_8_0 : Int),
      let w_9_0 : Int := 1
      let w_10_0 : Int := (w_8_0 * w_9_0)
      let w_11_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_6_0 w_10_0
      let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_11_0
      sequential_parallel_generatedRoot_5_13.constraints_0 i_0 w_7_0 w_2_0 w_5_0 w_6_0 w_8_0 w_12_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_5.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_12_0 : Fin 8 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1

abbrev parallel_generatedRoot_5.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : Int) (w_12_0 : Fin 8 → Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_22_0 : mxxMat1) (w_31_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 49, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 49, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_3_0 w_4_0 ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_parallel_generatedRoot_5_12 backend hashModel params i_0 i () (w_12_0 i)) ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_5_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (8)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  MxxRuntime.centeredRebaseRuns w_17_0 w_18_0 ∧
  outputs = (w_1_0, (w_2_0, (w_22_0, (w_31_0, ()))))

abbrev parallel_generatedRoot_5.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_5.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat0)
  let w_9_0 : mxxMat0 := (0 : mxxMat0)
  let w_10_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
  let w_11_0 : Int := -1
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
  parallel_generatedRoot_5.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_5.Witness,
    parallel_generatedRoot_5.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_6_12.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_6_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1), (1), (3), (3), (5), (5), (7), (7)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_6_12.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_6_13.constraints_0 (i_0 : Nat) (w_7_0 : Fin 8 → Int) (w_2_0 : Int) (w_5_0 : mxxMat0) (w_6_0 : mxxMat0) (w_8_0 : Int) (w_12_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_5_0 w_6_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_2_0 w_8_0 ∧
  outputs = w_12_0

def sequential_parallel_generatedRoot_6_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × Int × (Fin 8 → Int) × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_3_0 : Int := inputs.2.2.1
  let w_7_0 : Fin 8 → Int := inputs.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 * w_3_0)
  let w_5_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_1_0 w_4_0
  ∃ (w_6_0 : mxxMat0),
    ∃ (w_8_0 : Int),
      let w_9_0 : Int := 1
      let w_10_0 : Int := (w_8_0 * w_9_0)
      let w_11_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_6_0 w_10_0
      let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_11_0
      sequential_parallel_generatedRoot_6_13.constraints_0 i_0 w_7_0 w_2_0 w_5_0 w_6_0 w_8_0 w_12_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_6.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_12_0 : Fin 8 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1

abbrev parallel_generatedRoot_6.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : Int) (w_12_0 : Fin 8 → Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_22_0 : mxxMat1) (w_31_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 50, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 50, 41, 47, 50, 47, 51]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_3_0 w_4_0 ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_parallel_generatedRoot_6_12 backend hashModel params i_0 i () (w_12_0 i)) ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_6_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (8)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  MxxRuntime.centeredRebaseRuns w_17_0 w_18_0 ∧
  outputs = (w_1_0, (w_2_0, (w_22_0, (w_31_0, ()))))

abbrev parallel_generatedRoot_6.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_6.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat0)
  let w_9_0 : mxxMat0 := (0 : mxxMat0)
  let w_10_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
  let w_11_0 : Int := -1
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
  parallel_generatedRoot_6.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs

def parallel_generatedRoot_6 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_6.Witness,
    parallel_generatedRoot_6.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_7_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_7_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_7_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_7_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 1 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_7_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 1 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 1 → Int := inputs.2.1
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
      sequential_parallel_generatedRoot_7_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_7.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 1 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_7.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 1 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 48, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 48, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_3_0 w_4_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_parallel_generatedRoot_7_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_7_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (1)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_7.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_7.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
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
  parallel_generatedRoot_7.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_7 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_7.Witness,
    parallel_generatedRoot_7.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_8_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_8_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(0), (0), (0), (0), (0), (1), (0), (1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_8_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_8_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 8 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_8_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 8 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
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
      sequential_parallel_generatedRoot_8_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_8.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 8 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_8.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 8 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 50, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 49, 50, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_3_0 w_4_0 ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_parallel_generatedRoot_8_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_8_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (8)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_8.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_8.Witness) : Prop :=
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
  parallel_generatedRoot_8.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_8.Witness,
    parallel_generatedRoot_8.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_9_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 8 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_9_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(0), (1), (0), (1), (0), (1), (0), (1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_9_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_9_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 8 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 8 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_9_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 8 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
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
      sequential_parallel_generatedRoot_9_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_9.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 8 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_9.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 8 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 51, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 51, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_3_0 w_4_0 ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_parallel_generatedRoot_9_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_9_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (8)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_9.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_9.Witness) : Prop :=
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
  parallel_generatedRoot_9.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_9 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_9.Witness,
    parallel_generatedRoot_9.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_10_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_10_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_10_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_10_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 1 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_10_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 1 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 1 → Int := inputs.2.1
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
      sequential_parallel_generatedRoot_10_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_10.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 1 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 1 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 52, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 52, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_3_0 w_4_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_parallel_generatedRoot_10_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_10_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (1)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_10.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_10.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
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
  parallel_generatedRoot_10.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_10.Witness,
    parallel_generatedRoot_10.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_11_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_11_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_11_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_11_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 1 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_11_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 1 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 1 → Int := inputs.2.1
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
      sequential_parallel_generatedRoot_11_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_11.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 1 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 1 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 53, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 53, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_3_0 w_4_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_parallel_generatedRoot_11_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_11_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (1)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_11.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_11.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
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
  parallel_generatedRoot_11.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_11.Witness,
    parallel_generatedRoot_11.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_12_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_12_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_12_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_12_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 1 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_12_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 1 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 1 → Int := inputs.2.1
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
      sequential_parallel_generatedRoot_12_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_12.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 1 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_12.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 1 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 54, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 54, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_3_0 w_4_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_parallel_generatedRoot_12_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_12_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (1)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_12.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_12.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
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
  parallel_generatedRoot_12.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_12 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_12.Witness,
    parallel_generatedRoot_12.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_13_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_13_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_13_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_13_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 1 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_13_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 1 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 1 → Int := inputs.2.1
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
      sequential_parallel_generatedRoot_13_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_13.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 1 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_13.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 1 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 55, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 55, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_3_0 w_4_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_parallel_generatedRoot_13_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_13_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (1)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_13.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_13.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
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
  parallel_generatedRoot_13.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_13.Witness,
    parallel_generatedRoot_13.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_14_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_14_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_14_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_14_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 1 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_14_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 1 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 1 → Int := inputs.2.1
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
      sequential_parallel_generatedRoot_14_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_14.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 1 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_14.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 1 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 56, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 56, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_3_0 w_4_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_parallel_generatedRoot_14_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_14_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (1)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_14.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_14.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
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
  parallel_generatedRoot_14.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_14.Witness,
    parallel_generatedRoot_14.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_parallel_generatedRoot_15_10.constraints_0 (i_1 : Nat) (w_2_0 : Int) (outputs : Int) : Prop :=
  0 ≤ ((Int.ofNat i_1)) ∧
  ((Int.ofNat i_1)) < 1 ∧
  outputs = w_2_0

def parallel_parallel_generatedRoot_15_10 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (MxxIR.selectInt ((Int.ofNat i_1)) [(1)])
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_parallel_generatedRoot_15_10.constraints_0 i_1 w_2_0 outputs

abbrev sequential_parallel_generatedRoot_15_13.constraints_0 (i_0 : Nat) (w_1_0 : Fin 1 → Int) (w_2_0 : Int) (w_3_0 : Int) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (outputs : mxxMat0) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_8_0 w_9_0 ∧
  outputs = w_11_0

def sequential_parallel_generatedRoot_15_13 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × (Fin 1 → Int) × mxxMat0 × Int × Unit) (outputs : mxxMat0) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : Fin 1 → Int := inputs.2.1
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
      sequential_parallel_generatedRoot_15_13.constraints_0 i_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_15.Witness where
  w_1_0 : mxxMat1
  w_2_0 : mxxMat1
  w_4_0 : mxxMat0
  w_10_0 : Fin 1 → Int
  w_13_0 : mxxMat0
  w_17_0 : mxxMat1

abbrev parallel_generatedRoot_15.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : ByteArray) (w_3_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_10_0 : Fin 1 → Int) (w_11_0 : mxxMat0) (w_12_0 : Int) (w_13_0 : mxxMat0) (w_17_0 : mxxMat1) (w_21_0 : mxxMat1) (w_30_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 57, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 98, 47, 76, 117, 116, 73, 100, 40, 57, 41, 47, 51, 47, 52]) ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_3_0 w_4_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_parallel_generatedRoot_15_10 backend hashModel params i_0 i () (w_10_0 i)) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0) => sequential_parallel_generatedRoot_15_13 backend hashModel params i_0 i (current, (w_10_0, (w_11_0, (w_12_0, ())))) next) (Int.toNat (1)) w_9_0 w_13_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  outputs = (w_1_0, (w_2_0, (w_21_0, (w_30_0, ()))))

abbrev parallel_generatedRoot_15.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_15.Witness) : Prop :=
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
  let w_8_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
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
  parallel_generatedRoot_15.constraints_0 backend hashModel params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs

def parallel_generatedRoot_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_15.Witness,
    parallel_generatedRoot_15.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_36.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_17_0 : mxxMat1) (w_25_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_4_0 ∧
  MxxRuntime.centeredRebaseRuns w_4_0 w_5_0 ∧
  outputs = (w_1_0, (w_2_0, (w_17_0, (w_25_0, ()))))

def parallel_generatedRoot_36 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          parallel_generatedRoot_36.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_4_0 w_5_0 w_17_0 w_25_0 outputs

abbrev sequential_generatedRoot_69.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_69.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 4 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_69.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 4 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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

abbrev sequential_generatedRoot_69.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 4 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 4 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
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

def sequential_generatedRoot_69 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                                  sequential_generatedRoot_69.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_69.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_69.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_69.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev parallel_generatedRoot_79.constraints_2 (backend : MxxRuntime.BackendContext) (w_19_0 : mxxMat1) (w_47_0 : mxxMat1) (w_51_0 : mxxMat2) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat3) (w_55_0 : mxxMat3) (w_64_0 : mxxMat1) (w_65_0 : mxxMat3) (w_66_0 : mxxMat3) (w_68_0 : mxxMat2) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_79.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_18_0 : ByteArray) (w_23_0 : mxxMat1) (w_26_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_24_0 : mxxMat3) (w_27_0 : mxxMat3) (w_29_0 : mxxMat2) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_32_0 : mxxMat2) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat3) (w_49_0 : mxxMat3) (w_51_0 : mxxMat2) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat3) : Prop :=
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

abbrev parallel_generatedRoot_79.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_1_0 : Int) (w_5_0 : mxxMat1) (w_11_0 : mxxMat1) (w_18_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_6_0 : mxxMat3) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : mxxMat3) (w_14_0 : mxxMat2) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat2) (w_19_0 : mxxMat1) : Prop :=
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

def parallel_generatedRoot_79 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Int × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                  parallel_generatedRoot_79.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_5_0 w_11_0 w_18_0 w_2_0 w_3_0 w_4_0 w_6_0 w_8_0 w_9_0 w_10_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_19_0 ∧
    parallel_generatedRoot_79.constraints_1 backend hashModel i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_54_0 ∧
    parallel_generatedRoot_79.constraints_2 backend w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs

abbrev parallel_parallel_generatedRoot_80_4.constraints_0 (w_0_0 : Fin 72000 → mxxMat1) (w_3_0 : Fin 72000 → mxxMat1) (w_5_0 : Fin 72000 → mxxMat1) (w_7_0 : Fin 72000 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, ()))))

def parallel_parallel_generatedRoot_80_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat1 := inputs.1
  let w_3_0 : Fin 72000 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 72000 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.1
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

abbrev parallel_generatedRoot_80.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 72000 → mxxMat1) (w_1_0 : Fin 72000 → mxxMat1) (w_2_0 : Fin 72000 → mxxMat1) (w_3_0 : Fin 72000 → mxxMat1) (w_4_0 : Fin 2 → mxxMat1) (w_4_1 : Fin 2 → mxxMat1) (w_4_2 : Fin 2 → mxxMat1) (w_4_3 : Fin 2 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_17_1 : mxxMat1) (w_17_2 : mxxMat1) (w_17_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_80.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_80.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat1 := inputs.1
  let w_1_0 : Fin 72000 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 72000 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.1
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

def parallel_generatedRoot_80 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_80.Witness,
    parallel_generatedRoot_80.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_83.constraints_3 (w_22_0 : mxxMat1) (w_52_0 : mxxMat1) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_69_0 : mxxMat1) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) (w_76_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_73_0 1 2 0 25 w_76_0 ∧
  outputs = (w_57_0, (w_74_0, (w_75_0, (w_76_0, (w_22_0, (w_58_0, (w_52_0, (w_69_0, ()))))))))

abbrev parallel_generatedRoot_83.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_21_0 : ByteArray) (w_39_0 : mxxMat1) (w_40_0 : mxxMat1) (w_52_0 : mxxMat1) (w_53_0 : mxxMat3) (w_54_0 : mxxMat3) (w_56_0 : mxxMat2) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_69_0 : mxxMat1) (w_70_0 : mxxMat3) (w_71_0 : mxxMat3) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) : Prop :=
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

abbrev parallel_generatedRoot_83.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_21_0 : ByteArray) (w_26_0 : Fin 36000 → mxxMat1) (w_30_0 : Fin 36000 → mxxMat1) (w_6_0 : Int) (w_17_0 : mxxMat2) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat2) (w_22_0 : mxxMat1) (w_23_0 : mxxMat3) (w_24_0 : mxxMat3) (w_27_0 : mxxMat1) (w_28_0 : mxxMat3) (w_31_0 : mxxMat1) (w_32_0 : mxxMat3) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat2) : Prop :=
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
  w_6_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_26_0 w_6_0 w_27_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_27_0 w_28_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 36000 ∧
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

abbrev parallel_generatedRoot_83.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_0_0 : ByteArray) (w_1_0 : Int) (w_5_0 : Fin 36000 → mxxMat1) (w_13_0 : Fin 36000 → mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_6_0 : Int) (w_7_0 : mxxMat1) (w_8_0 : mxxMat3) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat2) (w_14_0 : mxxMat1) (w_15_0 : mxxMat3) (w_17_0 : mxxMat2) (w_18_0 : mxxMat1) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_3_0 ∧
  MxxRuntime.concatRows w_2_0 w_3_0 w_4_0 ∧
  (2) ≠ 0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 36000 ∧
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
  w_6_0 < 36000 ∧
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

def parallel_generatedRoot_83 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Int × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × ByteArray × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_1_0 : Int := inputs.2.1
  let w_5_0 : Fin 36000 → mxxMat1 := inputs.2.2.1
  let w_13_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.1
  let w_21_0 : ByteArray := inputs.2.2.2.2.1
  let w_26_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.1
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
                                                                          parallel_generatedRoot_83.constraints_0 backend hashModel w_0_0 w_1_0 w_5_0 w_13_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_83.constraints_1 backend hashModel i_0 w_21_0 w_26_0 w_30_0 w_6_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_27_0 w_28_0 w_31_0 w_32_0 w_34_0 w_35_0 w_36_0 w_37_0 ∧
    parallel_generatedRoot_83.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 ∧
    parallel_generatedRoot_83.constraints_3 w_22_0 w_52_0 w_57_0 w_58_0 w_69_0 w_73_0 w_74_0 w_75_0 w_76_0 outputs

abbrev parallel_parallel_generatedRoot_84_4.constraints_0 (w_0_0 : Fin 72000 → mxxMat1) (w_3_0 : Fin 72000 → mxxMat1) (w_5_0 : Fin 72000 → mxxMat1) (w_7_0 : Fin 72000 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 72000 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, ()))))

def parallel_parallel_generatedRoot_84_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat1 := inputs.1
  let w_3_0 : Fin 72000 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 72000 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          parallel_parallel_generatedRoot_84_4.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 outputs

abbrev sequential_parallel_generatedRoot_84_17.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_84_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          sequential_parallel_generatedRoot_84_17.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_84.Witness where
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

abbrev parallel_generatedRoot_84.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 72000 → mxxMat1) (w_1_0 : Fin 72000 → mxxMat1) (w_2_0 : Fin 72000 → mxxMat1) (w_3_0 : Fin 72000 → mxxMat1) (w_4_0 : Fin 2 → mxxMat1) (w_4_1 : Fin 2 → mxxMat1) (w_4_2 : Fin 2 → mxxMat1) (w_4_3 : Fin 2 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_17_1 : mxxMat1) (w_17_2 : mxxMat1) (w_17_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_84_4 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_4_0 (0) w_5_0 ∧
  MxxRuntime.familyGetStatic w_4_0 (1) w_6_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (1) w_9_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (0) w_11_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (1) w_12_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (0) w_14_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (1) w_15_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_84_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_4_0, (w_4_1, (w_4_2, (w_4_3, ())))))))) next) (Int.toNat (0)) (w_7_0, (w_10_0, (w_13_0, (w_16_0, ())))) (w_17_0, (w_17_1, (w_17_2, (w_17_3, ())))) ∧
  outputs = (w_17_0, (w_17_1, (w_17_2, (w_17_3, ()))))

abbrev parallel_generatedRoot_84.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_84.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 72000 → mxxMat1 := inputs.1
  let w_1_0 : Fin 72000 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 72000 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 72000 → mxxMat1 := inputs.2.2.2.1
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
  parallel_generatedRoot_84.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs

def parallel_generatedRoot_84 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_84.Witness,
    parallel_generatedRoot_84.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_85_10.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_85_10.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 8 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_parallel_generatedRoot_85_10.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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

abbrev sequential_parallel_generatedRoot_85_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 8 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
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

def sequential_parallel_generatedRoot_85_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                                  sequential_parallel_generatedRoot_85_10.constraints_0 backend hashModel i_1 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_parallel_generatedRoot_85_10.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_parallel_generatedRoot_85_10.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_parallel_generatedRoot_85_10.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_85.Witness where
  w_10_0 : mxxMat1
  w_10_1 : mxxMat1
  w_10_2 : mxxMat1
  w_10_3 : mxxMat1

abbrev parallel_generatedRoot_85.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_8_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_0_0 : mxxMat1) (w_10_0 : mxxMat1) (w_10_1 : mxxMat1) (w_10_2 : mxxMat1) (w_10_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_85_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, (w_6_0, (w_7_0, (w_8_0, (w_9_0, ()))))))))))))) next) (Int.toNat (8)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, ())))) (w_10_0, (w_10_1, (w_10_2, (w_10_3, ())))) ∧
  outputs = (w_10_0, (w_10_1, (w_10_2, (w_10_3, ()))))

abbrev parallel_generatedRoot_85.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_85.Witness) : Prop :=
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
  parallel_generatedRoot_85.constraints_0 backend hashModel params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs

def parallel_generatedRoot_85 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_85.Witness,
    parallel_generatedRoot_85.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_86.constraints_0 (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  outputs = (w_2_0, (w_5_0, (w_8_0, (w_11_0, ()))))

def parallel_generatedRoot_86 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
  parallel_generatedRoot_86.constraints_0 w_2_0 w_5_0 w_8_0 w_11_0 outputs

abbrev sequential_generatedRoot_99.constraints_0 (w_1_0 : Fin 36000 → mxxMat1) (w_6_0 : Fin 36000 → mxxMat1) (w_10_0 : Fin 36000 → mxxMat1) (w_14_0 : Fin 36000 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_generatedRoot_99 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.1
  let w_6_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_generatedRoot_99.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

abbrev sequential_sequential_generatedRoot_107_20.constraints_0 (w_1_0 : Fin 36000 → mxxMat1) (w_6_0 : Fin 36000 → mxxMat1) (w_10_0 : Fin 36000 → mxxMat1) (w_14_0 : Fin 36000 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_sequential_generatedRoot_107_20 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.1
  let w_6_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := ((10 + (8 * (Int.ofNat i_0))) + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_sequential_generatedRoot_107_20.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_107.Witness where
  w_4_0 : mxxMat1
  w_6_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_20_0 : mxxMat1
  w_20_1 : mxxMat1
  w_20_2 : mxxMat1
  w_20_3 : mxxMat1
  w_21_0 : mxxMat4
  w_25_0 : mxxMat4
  w_29_0 : mxxMat4
  w_33_0 : mxxMat4

abbrev sequential_generatedRoot_107.constraints_1 (i_0 : Nat) (w_20_2 : mxxMat1) (w_20_3 : mxxMat1) (w_22_0 : mxxMat4) (w_26_0 : mxxMat4) (w_28_0 : mxxMat5) (w_29_0 : mxxMat4) (w_30_0 : mxxMat4) (w_32_0 : mxxMat5) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  MxxRuntime.tensorRuns w_28_0 w_20_2 w_29_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  MxxRuntime.tensorRuns w_32_0 w_20_3 w_33_0 ∧
  outputs = (w_22_0, (w_26_0, (w_30_0, (w_34_0, ()))))

abbrev sequential_generatedRoot_107.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_2_0 : Fin 36000 → mxxMat1) (w_8_0 : Fin 36000 → mxxMat1) (w_12_0 : Fin 36000 → mxxMat1) (w_16_0 : Fin 36000 → mxxMat1) (w_1_0 : mxxMat5) (w_3_0 : Int) (w_4_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_20_1 : mxxMat1) (w_20_2 : mxxMat1) (w_20_3 : mxxMat1) (w_21_0 : mxxMat4) (w_24_0 : mxxMat5) (w_25_0 : mxxMat4) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_3_0 w_9_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_5_0 w_10_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_3_0 w_13_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_5_0 w_14_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_3_0 w_17_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 36000 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_5_0 w_18_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_sequential_generatedRoot_107_20 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_2_0, (w_8_0, (w_12_0, (w_16_0, ())))))))) next) (Int.toNat (6)) (w_7_0, (w_11_0, (w_15_0, (w_19_0, ())))) (w_20_0, (w_20_1, (w_20_2, (w_20_3, ())))) ∧
  MxxRuntime.tensorRuns w_1_0 w_20_0 w_21_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 4500 ∧
  MxxRuntime.tensorRuns w_24_0 w_20_1 w_25_0

abbrev sequential_generatedRoot_107.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit) (outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) (witness : sequential_generatedRoot_107.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat4 := inputs.1
  let w_23_0 : mxxMat4 := inputs.2.1
  let w_27_0 : mxxMat4 := inputs.2.2.1
  let w_31_0 : mxxMat4 := inputs.2.2.2.1
  let w_2_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.1
  let w_8_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_12_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_16_0 : Fin 36000 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat5 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)
  let w_3_0 : Int := (8 + (8 * (Int.ofNat i_0)))
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := (9 + (8 * (Int.ofNat i_0)))
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_6_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_10_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_20_1 := witness.w_20_1
  let w_20_2 := witness.w_20_2
  let w_20_3 := witness.w_20_3
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat4 := MxxRuntime.matrixAdd w_0_0 w_21_0
  let w_24_0 : mxxMat5 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat4 := MxxRuntime.matrixAdd w_23_0 w_25_0
  let w_28_0 : mxxMat5 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat4 := MxxRuntime.matrixAdd w_27_0 w_29_0
  let w_32_0 : mxxMat5 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat5)
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat4 := MxxRuntime.matrixAdd w_31_0 w_33_0
  sequential_generatedRoot_107.constraints_0 backend hashModel params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_107.constraints_1 i_0 w_20_2 w_20_3 w_22_0 w_26_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 outputs

def sequential_generatedRoot_107 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × (Fin 36000 → mxxMat1) × Unit) (outputs : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_107.Witness,
    sequential_generatedRoot_107.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_116.constraints_2 (backend : MxxRuntime.BackendContext) (w_19_0 : mxxMat1) (w_47_0 : mxxMat1) (w_51_0 : mxxMat2) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat3) (w_55_0 : mxxMat3) (w_64_0 : mxxMat1) (w_65_0 : mxxMat3) (w_66_0 : mxxMat3) (w_68_0 : mxxMat2) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_116.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_18_0 : ByteArray) (w_23_0 : mxxMat1) (w_26_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_24_0 : mxxMat3) (w_27_0 : mxxMat3) (w_29_0 : mxxMat2) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_32_0 : mxxMat2) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat3) (w_49_0 : mxxMat3) (w_51_0 : mxxMat2) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat3) : Prop :=
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_18_0) w_53_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_53_0 w_54_0

abbrev parallel_generatedRoot_116.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_1_0 : Int) (w_5_0 : mxxMat1) (w_11_0 : mxxMat1) (w_18_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_6_0 : mxxMat3) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : mxxMat3) (w_14_0 : mxxMat2) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat2) (w_19_0 : mxxMat1) : Prop :=
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 51, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_18_0) w_19_0

def parallel_generatedRoot_116 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Int × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                  parallel_generatedRoot_116.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_5_0 w_11_0 w_18_0 w_2_0 w_3_0 w_4_0 w_6_0 w_8_0 w_9_0 w_10_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_19_0 ∧
    parallel_generatedRoot_116.constraints_1 backend hashModel i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_54_0 ∧
    parallel_generatedRoot_116.constraints_2 backend w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs

abbrev parallel_parallel_generatedRoot_117_4.constraints_0 (w_0_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_5_0 : Fin 16 → mxxMat1) (w_7_0 : Fin 16 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_117_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          parallel_parallel_generatedRoot_117_4.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 outputs

abbrev sequential_parallel_generatedRoot_117_17.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_117_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          sequential_parallel_generatedRoot_117_17.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_117.Witness where
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

abbrev parallel_generatedRoot_117.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat1) (w_1_0 : Fin 16 → mxxMat1) (w_2_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_4_0 : Fin 2 → mxxMat1) (w_4_1 : Fin 2 → mxxMat1) (w_4_2 : Fin 2 → mxxMat1) (w_4_3 : Fin 2 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_17_1 : mxxMat1) (w_17_2 : mxxMat1) (w_17_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_117_4 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_4_0 (0) w_5_0 ∧
  MxxRuntime.familyGetStatic w_4_0 (1) w_6_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (1) w_9_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (0) w_11_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (1) w_12_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (0) w_14_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (1) w_15_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_117_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_4_0, (w_4_1, (w_4_2, (w_4_3, ())))))))) next) (Int.toNat (0)) (w_7_0, (w_10_0, (w_13_0, (w_16_0, ())))) (w_17_0, (w_17_1, (w_17_2, (w_17_3, ())))) ∧
  outputs = (w_17_0, (w_17_1, (w_17_2, (w_17_3, ()))))

abbrev parallel_generatedRoot_117.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_117.Witness) : Prop :=
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
  parallel_generatedRoot_117.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs

def parallel_generatedRoot_117 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_117.Witness,
    parallel_generatedRoot_117.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_120.constraints_3 (w_22_0 : mxxMat1) (w_52_0 : mxxMat1) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_69_0 : mxxMat1) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) (w_76_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_73_0 1 2 0 25 w_76_0 ∧
  outputs = (w_57_0, (w_74_0, (w_75_0, (w_76_0, (w_22_0, (w_58_0, (w_52_0, (w_69_0, ()))))))))

abbrev parallel_generatedRoot_120.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_21_0 : ByteArray) (w_39_0 : mxxMat1) (w_40_0 : mxxMat1) (w_52_0 : mxxMat1) (w_53_0 : mxxMat3) (w_54_0 : mxxMat3) (w_56_0 : mxxMat2) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_69_0 : mxxMat1) (w_70_0 : mxxMat3) (w_71_0 : mxxMat3) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) : Prop :=
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 52, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_21_0) w_58_0 ∧
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

abbrev parallel_generatedRoot_120.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_21_0 : ByteArray) (w_26_0 : Fin 8 → mxxMat1) (w_30_0 : Fin 8 → mxxMat1) (w_6_0 : Int) (w_17_0 : mxxMat2) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat2) (w_22_0 : mxxMat1) (w_23_0 : mxxMat3) (w_24_0 : mxxMat3) (w_27_0 : mxxMat1) (w_28_0 : mxxMat3) (w_31_0 : mxxMat1) (w_32_0 : mxxMat3) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat2) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_17_0 1 2 0 25 w_19_0 ∧
  MxxRuntime.concatRows w_18_0 w_19_0 w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 52, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_21_0) w_22_0 ∧
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

abbrev parallel_generatedRoot_120.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_0_0 : ByteArray) (w_1_0 : Int) (w_5_0 : Fin 8 → mxxMat1) (w_13_0 : Fin 8 → mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_6_0 : Int) (w_7_0 : mxxMat1) (w_8_0 : mxxMat3) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat2) (w_14_0 : mxxMat1) (w_15_0 : mxxMat3) (w_17_0 : mxxMat2) (w_18_0 : mxxMat1) : Prop :=
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

def parallel_generatedRoot_120 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Int × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × ByteArray × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                          parallel_generatedRoot_120.constraints_0 backend hashModel w_0_0 w_1_0 w_5_0 w_13_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_120.constraints_1 backend hashModel i_0 w_21_0 w_26_0 w_30_0 w_6_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_27_0 w_28_0 w_31_0 w_32_0 w_34_0 w_35_0 w_36_0 w_37_0 ∧
    parallel_generatedRoot_120.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 ∧
    parallel_generatedRoot_120.constraints_3 w_22_0 w_52_0 w_57_0 w_58_0 w_69_0 w_73_0 w_74_0 w_75_0 w_76_0 outputs

abbrev parallel_parallel_generatedRoot_121_4.constraints_0 (w_0_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_5_0 : Fin 16 → mxxMat1) (w_7_0 : Fin 16 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def parallel_parallel_generatedRoot_121_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          parallel_parallel_generatedRoot_121_4.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 outputs

abbrev sequential_parallel_generatedRoot_121_17.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_121_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          sequential_parallel_generatedRoot_121_17.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_121.Witness where
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

abbrev parallel_generatedRoot_121.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 16 → mxxMat1) (w_1_0 : Fin 16 → mxxMat1) (w_2_0 : Fin 16 → mxxMat1) (w_3_0 : Fin 16 → mxxMat1) (w_4_0 : Fin 2 → mxxMat1) (w_4_1 : Fin 2 → mxxMat1) (w_4_2 : Fin 2 → mxxMat1) (w_4_3 : Fin 2 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_17_1 : mxxMat1) (w_17_2 : mxxMat1) (w_17_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_121_4 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_4_0 (0) w_5_0 ∧
  MxxRuntime.familyGetStatic w_4_0 (1) w_6_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (1) w_9_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (0) w_11_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (1) w_12_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (0) w_14_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (1) w_15_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_121_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_4_0, (w_4_1, (w_4_2, (w_4_3, ())))))))) next) (Int.toNat (0)) (w_7_0, (w_10_0, (w_13_0, (w_16_0, ())))) (w_17_0, (w_17_1, (w_17_2, (w_17_3, ())))) ∧
  outputs = (w_17_0, (w_17_1, (w_17_2, (w_17_3, ()))))

abbrev parallel_generatedRoot_121.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_121.Witness) : Prop :=
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
  parallel_generatedRoot_121.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs

def parallel_generatedRoot_121 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_121.Witness,
    parallel_generatedRoot_121.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_122_10.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_122_10.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 8 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_parallel_generatedRoot_122_10.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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

abbrev sequential_parallel_generatedRoot_122_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 8 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
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

def sequential_parallel_generatedRoot_122_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                                  sequential_parallel_generatedRoot_122_10.constraints_0 backend hashModel i_1 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_parallel_generatedRoot_122_10.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_parallel_generatedRoot_122_10.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_parallel_generatedRoot_122_10.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_122.Witness where
  w_10_0 : mxxMat1
  w_10_1 : mxxMat1
  w_10_2 : mxxMat1
  w_10_3 : mxxMat1

abbrev parallel_generatedRoot_122.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_8_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_0_0 : mxxMat1) (w_10_0 : mxxMat1) (w_10_1 : mxxMat1) (w_10_2 : mxxMat1) (w_10_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_122_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, (w_6_0, (w_7_0, (w_8_0, (w_9_0, ()))))))))))))) next) (Int.toNat (8)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, ())))) (w_10_0, (w_10_1, (w_10_2, (w_10_3, ())))) ∧
  outputs = (w_10_0, (w_10_1, (w_10_2, (w_10_3, ()))))

abbrev parallel_generatedRoot_122.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_122.Witness) : Prop :=
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
  parallel_generatedRoot_122.constraints_0 backend hashModel params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs

def parallel_generatedRoot_122 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_122.Witness,
    parallel_generatedRoot_122.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_123_10.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_123_10.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 8 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_parallel_generatedRoot_123_10.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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

abbrev sequential_parallel_generatedRoot_123_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 8 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
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

def sequential_parallel_generatedRoot_123_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                                  sequential_parallel_generatedRoot_123_10.constraints_0 backend hashModel i_1 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_parallel_generatedRoot_123_10.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_parallel_generatedRoot_123_10.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_parallel_generatedRoot_123_10.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_123.Witness where
  w_10_0 : mxxMat1
  w_10_1 : mxxMat1
  w_10_2 : mxxMat1
  w_10_3 : mxxMat1

abbrev parallel_generatedRoot_123.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_8_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_0_0 : mxxMat1) (w_10_0 : mxxMat1) (w_10_1 : mxxMat1) (w_10_2 : mxxMat1) (w_10_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_123_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, (w_6_0, (w_7_0, (w_8_0, (w_9_0, ()))))))))))))) next) (Int.toNat (8)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, ())))) (w_10_0, (w_10_1, (w_10_2, (w_10_3, ())))) ∧
  outputs = (w_10_0, (w_10_1, (w_10_2, (w_10_3, ()))))

abbrev parallel_generatedRoot_123.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_123.Witness) : Prop :=
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
  parallel_generatedRoot_123.constraints_0 backend hashModel params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs

def parallel_generatedRoot_123 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_123.Witness,
    parallel_generatedRoot_123.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_124.constraints_0 (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  outputs = (w_2_0, (w_5_0, (w_8_0, (w_11_0, ()))))

def parallel_generatedRoot_124 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_3_0 : mxxMat1 := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  let w_2_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_0_0 w_1_0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
  let w_7_0 : Int := (Int.ofNat i_0)
  let w_8_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_6_0 w_7_0
  let w_10_0 : Int := (Int.ofNat i_0)
  let w_11_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_9_0 w_10_0
  parallel_generatedRoot_124.constraints_0 w_2_0 w_5_0 w_8_0 w_11_0 outputs

abbrev sequential_generatedRoot_137.constraints_0 (w_1_0 : Fin 8 → mxxMat1) (w_6_0 : Fin 8 → mxxMat1) (w_10_0 : Fin 8 → mxxMat1) (w_14_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_generatedRoot_137 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          sequential_generatedRoot_137.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_139.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_139.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 1 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_139.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1

abbrev sequential_generatedRoot_139.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_generatedRoot_139 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3)])
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
                                                                                  sequential_generatedRoot_139.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_139.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_139.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_139.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev sequential_generatedRoot_141.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_141.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 1 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_141.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1

abbrev sequential_generatedRoot_141.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_generatedRoot_141 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5)])
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
                                                                                  sequential_generatedRoot_141.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_141.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_141.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_141.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev sequential_generatedRoot_143.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_143.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 1 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_143.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1

abbrev sequential_generatedRoot_143.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_generatedRoot_143 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(7)])
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
                                                                                  sequential_generatedRoot_143.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_143.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_143.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_143.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev sequential_generatedRoot_145.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_145.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 1 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_145.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1

abbrev sequential_generatedRoot_145.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_generatedRoot_145 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(9)])
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
                                                                                  sequential_generatedRoot_145.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_145.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_145.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_145.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev sequential_generatedRoot_147.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_147.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 1 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_147.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1

abbrev sequential_generatedRoot_147.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_generatedRoot_147 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(11)])
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
                                                                                  sequential_generatedRoot_147.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_147.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_147.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_147.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev sequential_generatedRoot_149.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_149.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 1 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_149.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1

abbrev sequential_generatedRoot_149.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_generatedRoot_149 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(13)])
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
                                                                                  sequential_generatedRoot_149.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_149.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_149.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_149.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev sequential_generatedRoot_151.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_generatedRoot_151.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 1 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_generatedRoot_151.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_43_0 w_12_0 w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_45_0 w_46_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1

abbrev sequential_generatedRoot_151.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_4_0 ∧
  MxxRuntime.concatRows w_3_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatRows w_7_0 w_9_0 w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_13_0 w_14_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_17_0 w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_19_0 w_20_0 ∧
  MxxRuntime.concatRows w_18_0 w_20_0 w_21_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_22_0 w_12_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_23_0 w_24_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2

def sequential_generatedRoot_151 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_42_0 : mxxMat1 := inputs.2.1
  let w_65_0 : mxxMat1 := inputs.2.2.1
  let w_68_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_8_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_11_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_17_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_22_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_43_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_48_0 : Fin 1 → mxxMat1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15)])
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
                                                                                  sequential_generatedRoot_151.constraints_0 backend hashModel i_0 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_151.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_generatedRoot_151.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_generatedRoot_151.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

abbrev parallel_generatedRoot_161.constraints_2 (backend : MxxRuntime.BackendContext) (w_19_0 : mxxMat1) (w_47_0 : mxxMat1) (w_51_0 : mxxMat2) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat3) (w_55_0 : mxxMat3) (w_64_0 : mxxMat1) (w_65_0 : mxxMat3) (w_66_0 : mxxMat3) (w_68_0 : mxxMat2) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (w_71_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev parallel_generatedRoot_161.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_18_0 : ByteArray) (w_23_0 : mxxMat1) (w_26_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_24_0 : mxxMat3) (w_27_0 : mxxMat3) (w_29_0 : mxxMat2) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_32_0 : mxxMat2) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat3) (w_49_0 : mxxMat3) (w_51_0 : mxxMat2) (w_52_0 : mxxMat1) (w_53_0 : mxxMat1) (w_54_0 : mxxMat3) : Prop :=
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 53, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_18_0) w_53_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_53_0 w_54_0

abbrev parallel_generatedRoot_161.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_1_0 : Int) (w_5_0 : mxxMat1) (w_11_0 : mxxMat1) (w_18_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_6_0 : mxxMat3) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : mxxMat3) (w_14_0 : mxxMat2) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat2) (w_19_0 : mxxMat1) : Prop :=
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 53, 41, 47, 48, 47, 49]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_18_0) w_19_0

def parallel_generatedRoot_161 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Int × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                  parallel_generatedRoot_161.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_5_0 w_11_0 w_18_0 w_2_0 w_3_0 w_4_0 w_6_0 w_8_0 w_9_0 w_10_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_19_0 ∧
    parallel_generatedRoot_161.constraints_1 backend hashModel i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_54_0 ∧
    parallel_generatedRoot_161.constraints_2 backend w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs

abbrev parallel_parallel_generatedRoot_162_4.constraints_0 (w_0_0 : Fin 14400 → mxxMat1) (w_3_0 : Fin 14400 → mxxMat1) (w_5_0 : Fin 14400 → mxxMat1) (w_7_0 : Fin 14400 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, ()))))

def parallel_parallel_generatedRoot_162_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat1 := inputs.1
  let w_3_0 : Fin 14400 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 14400 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          parallel_parallel_generatedRoot_162_4.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 outputs

abbrev sequential_parallel_generatedRoot_162_17.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_162_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          sequential_parallel_generatedRoot_162_17.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_162.Witness where
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

abbrev parallel_generatedRoot_162.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 14400 → mxxMat1) (w_1_0 : Fin 14400 → mxxMat1) (w_2_0 : Fin 14400 → mxxMat1) (w_3_0 : Fin 14400 → mxxMat1) (w_4_0 : Fin 2 → mxxMat1) (w_4_1 : Fin 2 → mxxMat1) (w_4_2 : Fin 2 → mxxMat1) (w_4_3 : Fin 2 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_17_1 : mxxMat1) (w_17_2 : mxxMat1) (w_17_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_162_4 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_4_0 (0) w_5_0 ∧
  MxxRuntime.familyGetStatic w_4_0 (1) w_6_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (1) w_9_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (0) w_11_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (1) w_12_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (0) w_14_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (1) w_15_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_162_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_4_0, (w_4_1, (w_4_2, (w_4_3, ())))))))) next) (Int.toNat (0)) (w_7_0, (w_10_0, (w_13_0, (w_16_0, ())))) (w_17_0, (w_17_1, (w_17_2, (w_17_3, ())))) ∧
  outputs = (w_17_0, (w_17_1, (w_17_2, (w_17_3, ()))))

abbrev parallel_generatedRoot_162.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_162.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat1 := inputs.1
  let w_1_0 : Fin 14400 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 14400 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.1
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
  parallel_generatedRoot_162.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs

def parallel_generatedRoot_162 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_162.Witness,
    parallel_generatedRoot_162.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_165.constraints_3 (w_22_0 : mxxMat1) (w_52_0 : mxxMat1) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_69_0 : mxxMat1) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) (w_76_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_73_0 1 2 0 25 w_76_0 ∧
  outputs = (w_57_0, (w_74_0, (w_75_0, (w_76_0, (w_22_0, (w_58_0, (w_52_0, (w_69_0, ()))))))))

abbrev parallel_generatedRoot_165.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_21_0 : ByteArray) (w_39_0 : mxxMat1) (w_40_0 : mxxMat1) (w_52_0 : mxxMat1) (w_53_0 : mxxMat3) (w_54_0 : mxxMat3) (w_56_0 : mxxMat2) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_69_0 : mxxMat1) (w_70_0 : mxxMat3) (w_71_0 : mxxMat3) (w_73_0 : mxxMat2) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) : Prop :=
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
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 54, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_21_0) w_58_0 ∧
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

abbrev parallel_generatedRoot_165.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_21_0 : ByteArray) (w_26_0 : Fin 7200 → mxxMat1) (w_30_0 : Fin 7200 → mxxMat1) (w_6_0 : Int) (w_17_0 : mxxMat2) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat2) (w_22_0 : mxxMat1) (w_23_0 : mxxMat3) (w_24_0 : mxxMat3) (w_27_0 : mxxMat1) (w_28_0 : mxxMat3) (w_31_0 : mxxMat1) (w_32_0 : mxxMat3) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat2) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_17_0 1 2 0 25 w_19_0 ∧
  MxxRuntime.concatRows w_18_0 w_19_0 w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 54, 41, 47, 49, 47, 50]) ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (mxxNat3), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (mxxNat4), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_21_0) w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_22_0 w_23_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_23_0 w_24_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_26_0 w_6_0 w_27_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_27_0 w_28_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 7200 ∧
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

abbrev parallel_generatedRoot_165.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_0_0 : ByteArray) (w_1_0 : Int) (w_5_0 : Fin 7200 → mxxMat1) (w_13_0 : Fin 7200 → mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_6_0 : Int) (w_7_0 : mxxMat1) (w_8_0 : mxxMat3) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat2) (w_14_0 : mxxMat1) (w_15_0 : mxxMat3) (w_17_0 : mxxMat2) (w_18_0 : mxxMat1) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 2 ^ 64 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le w_1_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_3_0 ∧
  MxxRuntime.concatRows w_2_0 w_3_0 w_4_0 ∧
  (2) ≠ 0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 7200 ∧
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
  w_6_0 < 7200 ∧
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

def parallel_generatedRoot_165 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × Int × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × ByteArray × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × mxxMat0 × Int × Int × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_1_0 : Int := inputs.2.1
  let w_5_0 : Fin 7200 → mxxMat1 := inputs.2.2.1
  let w_13_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.1
  let w_21_0 : ByteArray := inputs.2.2.2.2.1
  let w_26_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_30_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.1
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
                                                                          parallel_generatedRoot_165.constraints_0 backend hashModel w_0_0 w_1_0 w_5_0 w_13_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_17_0 w_18_0 ∧
    parallel_generatedRoot_165.constraints_1 backend hashModel i_0 w_21_0 w_26_0 w_30_0 w_6_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_27_0 w_28_0 w_31_0 w_32_0 w_34_0 w_35_0 w_36_0 w_37_0 ∧
    parallel_generatedRoot_165.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 ∧
    parallel_generatedRoot_165.constraints_3 w_22_0 w_52_0 w_57_0 w_58_0 w_69_0 w_73_0 w_74_0 w_75_0 w_76_0 outputs

abbrev parallel_parallel_generatedRoot_166_4.constraints_0 (w_0_0 : Fin 14400 → mxxMat1) (w_3_0 : Fin 14400 → mxxMat1) (w_5_0 : Fin 14400 → mxxMat1) (w_7_0 : Fin 14400 → mxxMat1) (w_1_0 : Int) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_0_0 w_1_0 w_2_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_3_0 w_1_0 w_4_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_1_0 w_6_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 14400 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_1_0 w_8_0 ∧
  outputs = (w_2_0, (w_4_0, (w_6_0, (w_8_0, ()))))

def parallel_parallel_generatedRoot_166_4 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat1 := inputs.1
  let w_3_0 : Fin 14400 → mxxMat1 := inputs.2.1
  let w_5_0 : Fin 14400 → mxxMat1 := inputs.2.2.1
  let w_7_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Int := ((2 * (Int.ofNat i_0)) + (Int.ofNat i_1))
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_4_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat1),
        ∃ (w_8_0 : mxxMat1),
          parallel_parallel_generatedRoot_166_4.constraints_0 w_0_0 w_3_0 w_5_0 w_7_0 w_1_0 w_2_0 w_4_0 w_6_0 w_8_0 outputs

abbrev sequential_parallel_generatedRoot_166_17.constraints_0 (w_1_0 : Fin 2 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_10_0 : Fin 2 → mxxMat1) (w_14_0 : Fin 2 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

def sequential_parallel_generatedRoot_166_17 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
          sequential_parallel_generatedRoot_166_17.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_166.Witness where
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

abbrev parallel_generatedRoot_166.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_0_0 : Fin 14400 → mxxMat1) (w_1_0 : Fin 14400 → mxxMat1) (w_2_0 : Fin 14400 → mxxMat1) (w_3_0 : Fin 14400 → mxxMat1) (w_4_0 : Fin 2 → mxxMat1) (w_4_1 : Fin 2 → mxxMat1) (w_4_2 : Fin 2 → mxxMat1) (w_4_3 : Fin 2 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (w_17_1 : mxxMat1) (w_17_2 : mxxMat1) (w_17_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_parallel_generatedRoot_166_4 backend hashModel params i_0 i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_4_0 (0) w_5_0 ∧
  MxxRuntime.familyGetStatic w_4_0 (1) w_6_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (0) w_8_0 ∧
  MxxRuntime.familyGetStatic w_4_1 (1) w_9_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (0) w_11_0 ∧
  MxxRuntime.familyGetStatic w_4_2 (1) w_12_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (0) w_14_0 ∧
  MxxRuntime.familyGetStatic w_4_3 (1) w_15_0 ∧
  0 ≤ (0) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_166_17 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_4_0, (w_4_1, (w_4_2, (w_4_3, ())))))))) next) (Int.toNat (0)) (w_7_0, (w_10_0, (w_13_0, (w_16_0, ())))) (w_17_0, (w_17_1, (w_17_2, (w_17_3, ())))) ∧
  outputs = (w_17_0, (w_17_1, (w_17_2, (w_17_3, ()))))

abbrev parallel_generatedRoot_166.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_166.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 14400 → mxxMat1 := inputs.1
  let w_1_0 : Fin 14400 → mxxMat1 := inputs.2.1
  let w_2_0 : Fin 14400 → mxxMat1 := inputs.2.2.1
  let w_3_0 : Fin 14400 → mxxMat1 := inputs.2.2.2.1
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
  parallel_generatedRoot_166.constraints_0 backend hashModel params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs

def parallel_generatedRoot_166 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_166.Witness,
    parallel_generatedRoot_166.body backend hashModel params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_167_10.constraints_3 (w_39_0 : mxxMat2) (w_41_0 : mxxMat1) (w_62_0 : mxxMat2) (w_64_0 : mxxMat1) (w_66_0 : mxxMat1) (w_67_0 : mxxMat1) (w_69_0 : mxxMat1) (w_70_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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

abbrev sequential_parallel_generatedRoot_167_10.constraints_2 (backend : MxxRuntime.BackendContext) (w_48_0 : Fin 8 → mxxMat1) (w_12_0 : Int) (w_49_0 : mxxMat1) (w_50_0 : mxxMat3) (w_51_0 : mxxMat3) (w_53_0 : mxxMat2) (w_54_0 : mxxMat1) (w_55_0 : mxxMat3) (w_56_0 : mxxMat3) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat2) (w_63_0 : mxxMat1) : Prop :=
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

abbrev sequential_parallel_generatedRoot_167_10.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : ByteArray) (w_43_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_12_0 : Int) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat2) (w_35_0 : mxxMat1) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_39_0 : mxxMat2) (w_40_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) (w_46_0 : mxxMat3) : Prop :=
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

abbrev sequential_parallel_generatedRoot_167_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_1 : Nat) (w_1_0 : ByteArray) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : Fin 8 → mxxMat1) (w_17_0 : mxxMat1) (w_19_0 : mxxMat1) (w_22_0 : Fin 8 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat2) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat2) (w_12_0 : Int) (w_13_0 : mxxMat1) (w_14_0 : mxxMat3) (w_15_0 : mxxMat3) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat2) (w_23_0 : mxxMat1) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) : Prop :=
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

def sequential_parallel_generatedRoot_167_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
                                                                                  sequential_parallel_generatedRoot_167_10.constraints_0 backend hashModel i_1 w_1_0 w_6_0 w_8_0 w_11_0 w_17_0 w_19_0 w_22_0 w_2_0 w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 ∧
    sequential_parallel_generatedRoot_167_10.constraints_1 backend hashModel w_1_0 w_43_0 w_2_0 w_12_0 w_27_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_39_0 w_40_0 w_44_0 w_45_0 w_46_0 ∧
    sequential_parallel_generatedRoot_167_10.constraints_2 backend w_48_0 w_12_0 w_49_0 w_50_0 w_51_0 w_53_0 w_54_0 w_55_0 w_56_0 w_58_0 w_59_0 w_60_0 w_62_0 w_63_0 ∧
    sequential_parallel_generatedRoot_167_10.constraints_3 w_39_0 w_41_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_70_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_167.Witness where
  w_10_0 : mxxMat1
  w_10_1 : mxxMat1
  w_10_2 : mxxMat1
  w_10_3 : mxxMat1

abbrev parallel_generatedRoot_167.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : Fin 8 → mxxMat1) (w_8_0 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_0_0 : mxxMat1) (w_10_0 : mxxMat1) (w_10_1 : mxxMat1) (w_10_2 : mxxMat1) (w_10_3 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ (8) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_167_10 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_1_0, (w_2_0, (w_3_0, (w_4_0, (w_5_0, (w_6_0, (w_7_0, (w_8_0, (w_9_0, ()))))))))))))) next) (Int.toNat (8)) (w_0_0, (w_0_0, (w_0_0, (w_0_0, ())))) (w_10_0, (w_10_1, (w_10_2, (w_10_3, ())))) ∧
  outputs = (w_10_0, (w_10_1, (w_10_2, (w_10_3, ()))))

abbrev parallel_generatedRoot_167.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (witness : parallel_generatedRoot_167.Witness) : Prop :=
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
  parallel_generatedRoot_167.constraints_0 backend hashModel params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs

def parallel_generatedRoot_167 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_167.Witness,
    parallel_generatedRoot_167.body backend hashModel params i_0 inputs outputs witness

abbrev parallel_generatedRoot_168.constraints_0 (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  (8) ≠ 0 ∧
  outputs = (w_2_0, (w_5_0, (w_8_0, (w_11_0, ()))))

def parallel_generatedRoot_168 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
  parallel_generatedRoot_168.constraints_0 w_2_0 w_5_0 w_8_0 w_11_0 outputs

abbrev sequential_generatedRoot_181.constraints_0 (w_1_0 : Fin 7200 → mxxMat1) (w_6_0 : Fin 7200 → mxxMat1) (w_10_0 : Fin 7200 → mxxMat1) (w_14_0 : Fin 7200 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_generatedRoot_181 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.1
  let w_6_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (2 + (Int.ofNat i_0))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_generatedRoot_181.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

abbrev sequential_sequential_generatedRoot_189_20.constraints_0 (w_1_0 : Fin 7200 → mxxMat1) (w_6_0 : Fin 7200 → mxxMat1) (w_10_0 : Fin 7200 → mxxMat1) (w_14_0 : Fin 7200 → mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_6_0 w_2_0 w_7_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_10_0 w_2_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_2_0 w_15_0 ∧
  outputs = (w_4_0, (w_8_0, (w_12_0, (w_16_0, ()))))

def sequential_sequential_generatedRoot_189_20 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_5_0 : mxxMat1 := inputs.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  let w_1_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.1
  let w_6_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_10_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_14_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Int := ((10 + (8 * (Int.ofNat i_0))) + (Int.ofNat i_1))
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_3_0
    ∃ (w_7_0 : mxxMat1),
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_5_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
        ∃ (w_15_0 : mxxMat1),
          let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_15_0
          sequential_sequential_generatedRoot_189_20.constraints_0 w_1_0 w_6_0 w_10_0 w_14_0 w_2_0 w_3_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_189.Witness where
  w_4_0 : mxxMat1
  w_6_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_20_0 : mxxMat1
  w_20_1 : mxxMat1
  w_20_2 : mxxMat1
  w_20_3 : mxxMat1
  w_21_0 : mxxMat6
  w_25_0 : mxxMat6
  w_29_0 : mxxMat6
  w_33_0 : mxxMat6

abbrev sequential_generatedRoot_189.constraints_1 (i_0 : Nat) (w_20_2 : mxxMat1) (w_20_3 : mxxMat1) (w_22_0 : mxxMat6) (w_26_0 : mxxMat6) (w_28_0 : mxxMat7) (w_29_0 : mxxMat6) (w_30_0 : mxxMat6) (w_32_0 : mxxMat7) (w_33_0 : mxxMat6) (w_34_0 : mxxMat6) (outputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × Unit) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  MxxRuntime.tensorRuns w_28_0 w_20_2 w_29_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  MxxRuntime.tensorRuns w_32_0 w_20_3 w_33_0 ∧
  outputs = (w_22_0, (w_26_0, (w_30_0, (w_34_0, ()))))

abbrev sequential_generatedRoot_189.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (w_2_0 : Fin 7200 → mxxMat1) (w_8_0 : Fin 7200 → mxxMat1) (w_12_0 : Fin 7200 → mxxMat1) (w_16_0 : Fin 7200 → mxxMat1) (w_1_0 : mxxMat7) (w_3_0 : Int) (w_4_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (w_20_1 : mxxMat1) (w_20_2 : mxxMat1) (w_20_3 : mxxMat1) (w_21_0 : mxxMat6) (w_24_0 : mxxMat7) (w_25_0 : mxxMat6) : Prop :=
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_3_0 w_9_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_5_0 w_10_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_3_0 w_13_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_5_0 w_14_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_3_0 w_17_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 7200 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_5_0 w_18_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_sequential_generatedRoot_189_20 backend hashModel params i_0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_2_0, (w_8_0, (w_12_0, (w_16_0, ())))))))) next) (Int.toNat (6)) (w_7_0, (w_11_0, (w_15_0, (w_19_0, ())))) (w_20_0, (w_20_1, (w_20_2, (w_20_3, ())))) ∧
  MxxRuntime.tensorRuns w_1_0 w_20_0 w_21_0 ∧
  0 ≤ ((1 + (Int.ofNat i_0))) ∧
  ((1 + (Int.ofNat i_0))) < 900 ∧
  MxxRuntime.tensorRuns w_24_0 w_20_1 w_25_0

abbrev sequential_generatedRoot_189.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit) (outputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × Unit) (witness : sequential_generatedRoot_189.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat6 := inputs.1
  let w_23_0 : mxxMat6 := inputs.2.1
  let w_27_0 : mxxMat6 := inputs.2.2.1
  let w_31_0 : mxxMat6 := inputs.2.2.2.1
  let w_2_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.1
  let w_8_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_12_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_16_0 : Fin 7200 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_3_0 : Int := (8 + (8 * (Int.ofNat i_0)))
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := (9 + (8 * (Int.ofNat i_0)))
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_6_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_10_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_20_1 := witness.w_20_1
  let w_20_2 := witness.w_20_2
  let w_20_3 := witness.w_20_3
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat6 := MxxRuntime.matrixAdd w_0_0 w_21_0
  let w_24_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat6 := MxxRuntime.matrixAdd w_23_0 w_25_0
  let w_28_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_29_0 := witness.w_29_0
  let w_30_0 : mxxMat6 := MxxRuntime.matrixAdd w_27_0 w_29_0
  let w_32_0 : mxxMat7 := (MxxRuntime.unitRow ((1 + (Int.ofNat i_0))) : mxxMat7)
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat6 := MxxRuntime.matrixAdd w_31_0 w_33_0
  sequential_generatedRoot_189.constraints_0 backend hashModel params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0 ∧
    sequential_generatedRoot_189.constraints_1 i_0 w_20_2 w_20_3 w_22_0 w_26_0 w_28_0 w_29_0 w_30_0 w_32_0 w_33_0 w_34_0 outputs

def sequential_generatedRoot_189 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × (Fin 7200 → mxxMat1) × Unit) (outputs : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_189.Witness,
    sequential_generatedRoot_189.body backend hashModel params i_0 inputs outputs witness

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
  w_5_0 : Fin 8 → mxxMat1
  w_5_1 : Fin 8 → mxxMat1
  w_5_2 : Fin 8 → mxxMat1
  w_5_3 : Fin 8 → mxxMat1
  w_6_0 : Fin 8 → mxxMat1
  w_6_1 : Fin 8 → mxxMat1
  w_6_2 : Fin 8 → mxxMat1
  w_6_3 : Fin 8 → mxxMat1
  w_7_0 : Fin 1 → mxxMat1
  w_7_1 : Fin 1 → mxxMat1
  w_7_2 : Fin 1 → mxxMat1
  w_7_3 : Fin 1 → mxxMat1
  w_8_0 : Fin 8 → mxxMat1
  w_8_1 : Fin 8 → mxxMat1
  w_8_2 : Fin 8 → mxxMat1
  w_8_3 : Fin 8 → mxxMat1
  w_9_0 : Fin 8 → mxxMat1
  w_9_1 : Fin 8 → mxxMat1
  w_9_2 : Fin 8 → mxxMat1
  w_9_3 : Fin 8 → mxxMat1
  w_10_0 : Fin 1 → mxxMat1
  w_10_1 : Fin 1 → mxxMat1
  w_10_2 : Fin 1 → mxxMat1
  w_10_3 : Fin 1 → mxxMat1
  w_11_0 : Fin 1 → mxxMat1
  w_11_1 : Fin 1 → mxxMat1
  w_11_2 : Fin 1 → mxxMat1
  w_11_3 : Fin 1 → mxxMat1
  w_12_0 : Fin 1 → mxxMat1
  w_12_1 : Fin 1 → mxxMat1
  w_12_2 : Fin 1 → mxxMat1
  w_12_3 : Fin 1 → mxxMat1
  w_13_0 : Fin 1 → mxxMat1
  w_13_1 : Fin 1 → mxxMat1
  w_13_2 : Fin 1 → mxxMat1
  w_13_3 : Fin 1 → mxxMat1
  w_14_0 : Fin 1 → mxxMat1
  w_14_1 : Fin 1 → mxxMat1
  w_14_2 : Fin 1 → mxxMat1
  w_14_3 : Fin 1 → mxxMat1
  w_15_0 : Fin 1 → mxxMat1
  w_15_1 : Fin 1 → mxxMat1
  w_15_2 : Fin 1 → mxxMat1
  w_15_3 : Fin 1 → mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat1
  w_20_0 : mxxMat2
  w_21_0 : mxxMat1
  w_22_0 : mxxMat3
  w_24_0 : mxxMat1
  w_25_0 : mxxMat1
  w_26_0 : mxxMat2
  w_27_0 : mxxMat1
  w_28_0 : mxxMat3
  w_31_0 : mxxMat1
  w_32_0 : mxxMat1
  w_33_0 : mxxMat2
  w_36_0 : Fin 1 → mxxMat1
  w_36_1 : Fin 1 → mxxMat1
  w_36_2 : Fin 1 → mxxMat1
  w_36_3 : Fin 1 → mxxMat1
  w_37_0 : mxxMat1
  w_38_0 : mxxMat3
  w_39_0 : mxxMat3
  w_41_0 : mxxMat1
  w_42_0 : mxxMat3
  w_44_0 : mxxMat1
  w_45_0 : mxxMat3
  w_48_0 : mxxMat1
  w_49_0 : mxxMat1
  w_50_0 : mxxMat2
  w_51_0 : mxxMat1
  w_52_0 : mxxMat3
  w_53_0 : mxxMat3
  w_56_0 : mxxMat1
  w_57_0 : mxxMat1
  w_58_0 : mxxMat1
  w_59_0 : mxxMat3
  w_60_0 : mxxMat3
  w_62_0 : mxxMat1
  w_63_0 : mxxMat3
  w_64_0 : mxxMat3
  w_67_0 : mxxMat1
  w_68_0 : mxxMat1
  w_69_0 : mxxMat1
  w_69_1 : mxxMat1
  w_69_2 : mxxMat1
  w_69_3 : mxxMat1
  w_73_0 : mxxMat1
  w_74_0 : mxxMat1
  w_75_0 : mxxMat1
  w_76_0 : mxxMat1
  w_79_0 : Fin 72000 → mxxMat1
  w_79_1 : Fin 72000 → mxxMat1
  w_79_2 : Fin 72000 → mxxMat1
  w_79_3 : Fin 72000 → mxxMat1
  w_79_4 : Fin 72000 → mxxMat1
  w_79_5 : Fin 72000 → mxxMat1
  w_79_6 : Fin 72000 → mxxMat1
  w_79_7 : Fin 72000 → mxxMat1
  w_80_0 : Fin 36000 → mxxMat1
  w_80_1 : Fin 36000 → mxxMat1
  w_80_2 : Fin 36000 → mxxMat1
  w_80_3 : Fin 36000 → mxxMat1
  w_83_0 : Fin 72000 → mxxMat1
  w_83_1 : Fin 72000 → mxxMat1
  w_83_2 : Fin 72000 → mxxMat1
  w_83_3 : Fin 72000 → mxxMat1
  w_83_4 : Fin 72000 → mxxMat1
  w_83_5 : Fin 72000 → mxxMat1
  w_83_6 : Fin 72000 → mxxMat1
  w_83_7 : Fin 72000 → mxxMat1
  w_84_0 : Fin 36000 → mxxMat1
  w_84_1 : Fin 36000 → mxxMat1
  w_84_2 : Fin 36000 → mxxMat1
  w_84_3 : Fin 36000 → mxxMat1
  w_85_0 : Fin 36000 → mxxMat1
  w_85_1 : Fin 36000 → mxxMat1
  w_85_2 : Fin 36000 → mxxMat1
  w_85_3 : Fin 36000 → mxxMat1
  w_86_0 : Fin 36000 → mxxMat1
  w_86_1 : Fin 36000 → mxxMat1
  w_86_2 : Fin 36000 → mxxMat1
  w_86_3 : Fin 36000 → mxxMat1
  w_87_0 : mxxMat1
  w_88_0 : mxxMat1
  w_90_0 : mxxMat1
  w_91_0 : mxxMat1
  w_93_0 : mxxMat1
  w_94_0 : mxxMat1
  w_96_0 : mxxMat1
  w_97_0 : mxxMat1
  w_99_0 : mxxMat1
  w_99_1 : mxxMat1
  w_99_2 : mxxMat1
  w_99_3 : mxxMat1
  w_100_0 : mxxMat4
  w_102_0 : mxxMat4
  w_104_0 : mxxMat4
  w_106_0 : mxxMat4
  w_107_0 : mxxMat4
  w_107_1 : mxxMat4
  w_107_2 : mxxMat4
  w_107_3 : mxxMat4
  w_110_0 : mxxMat1
  w_111_0 : mxxMat1
  w_112_0 : mxxMat1
  w_113_0 : mxxMat1
  w_116_0 : Fin 16 → mxxMat1
  w_116_1 : Fin 16 → mxxMat1
  w_116_2 : Fin 16 → mxxMat1
  w_116_3 : Fin 16 → mxxMat1
  w_116_4 : Fin 16 → mxxMat1
  w_116_5 : Fin 16 → mxxMat1
  w_116_6 : Fin 16 → mxxMat1
  w_116_7 : Fin 16 → mxxMat1
  w_117_0 : Fin 8 → mxxMat1
  w_117_1 : Fin 8 → mxxMat1
  w_117_2 : Fin 8 → mxxMat1
  w_117_3 : Fin 8 → mxxMat1
  w_120_0 : Fin 16 → mxxMat1
  w_120_1 : Fin 16 → mxxMat1
  w_120_2 : Fin 16 → mxxMat1
  w_120_3 : Fin 16 → mxxMat1
  w_120_4 : Fin 16 → mxxMat1
  w_120_5 : Fin 16 → mxxMat1
  w_120_6 : Fin 16 → mxxMat1
  w_120_7 : Fin 16 → mxxMat1
  w_121_0 : Fin 8 → mxxMat1
  w_121_1 : Fin 8 → mxxMat1
  w_121_2 : Fin 8 → mxxMat1
  w_121_3 : Fin 8 → mxxMat1
  w_122_0 : Fin 8 → mxxMat1
  w_122_1 : Fin 8 → mxxMat1
  w_122_2 : Fin 8 → mxxMat1
  w_122_3 : Fin 8 → mxxMat1
  w_123_0 : Fin 8 → mxxMat1
  w_123_1 : Fin 8 → mxxMat1
  w_123_2 : Fin 8 → mxxMat1
  w_123_3 : Fin 8 → mxxMat1
  w_124_0 : Fin 8 → mxxMat1
  w_124_1 : Fin 8 → mxxMat1
  w_124_2 : Fin 8 → mxxMat1
  w_124_3 : Fin 8 → mxxMat1
  w_125_0 : mxxMat1
  w_126_0 : mxxMat1
  w_128_0 : mxxMat1
  w_129_0 : mxxMat1
  w_131_0 : mxxMat1
  w_132_0 : mxxMat1
  w_134_0 : mxxMat1
  w_135_0 : mxxMat1
  w_137_0 : mxxMat1
  w_137_1 : mxxMat1
  w_137_2 : mxxMat1
  w_137_3 : mxxMat1
  w_139_0 : mxxMat1
  w_139_1 : mxxMat1
  w_139_2 : mxxMat1
  w_139_3 : mxxMat1
  w_141_0 : mxxMat1
  w_141_1 : mxxMat1
  w_141_2 : mxxMat1
  w_141_3 : mxxMat1
  w_143_0 : mxxMat1
  w_143_1 : mxxMat1
  w_143_2 : mxxMat1
  w_143_3 : mxxMat1
  w_145_0 : mxxMat1
  w_145_1 : mxxMat1
  w_145_2 : mxxMat1
  w_145_3 : mxxMat1
  w_147_0 : mxxMat1
  w_147_1 : mxxMat1
  w_147_2 : mxxMat1
  w_147_3 : mxxMat1
  w_149_0 : mxxMat1
  w_149_1 : mxxMat1
  w_149_2 : mxxMat1
  w_149_3 : mxxMat1
  w_151_0 : mxxMat1
  w_151_1 : mxxMat1
  w_151_2 : mxxMat1
  w_151_3 : mxxMat1
  w_155_0 : mxxMat1
  w_156_0 : mxxMat1
  w_157_0 : mxxMat1
  w_158_0 : mxxMat1
  w_161_0 : Fin 14400 → mxxMat1
  w_161_1 : Fin 14400 → mxxMat1
  w_161_2 : Fin 14400 → mxxMat1
  w_161_3 : Fin 14400 → mxxMat1
  w_161_4 : Fin 14400 → mxxMat1
  w_161_5 : Fin 14400 → mxxMat1
  w_161_6 : Fin 14400 → mxxMat1
  w_161_7 : Fin 14400 → mxxMat1
  w_162_0 : Fin 7200 → mxxMat1
  w_162_1 : Fin 7200 → mxxMat1
  w_162_2 : Fin 7200 → mxxMat1
  w_162_3 : Fin 7200 → mxxMat1
  w_165_0 : Fin 14400 → mxxMat1
  w_165_1 : Fin 14400 → mxxMat1
  w_165_2 : Fin 14400 → mxxMat1
  w_165_3 : Fin 14400 → mxxMat1
  w_165_4 : Fin 14400 → mxxMat1
  w_165_5 : Fin 14400 → mxxMat1
  w_165_6 : Fin 14400 → mxxMat1
  w_165_7 : Fin 14400 → mxxMat1
  w_166_0 : Fin 7200 → mxxMat1
  w_166_1 : Fin 7200 → mxxMat1
  w_166_2 : Fin 7200 → mxxMat1
  w_166_3 : Fin 7200 → mxxMat1
  w_167_0 : Fin 7200 → mxxMat1
  w_167_1 : Fin 7200 → mxxMat1
  w_167_2 : Fin 7200 → mxxMat1
  w_167_3 : Fin 7200 → mxxMat1
  w_168_0 : Fin 7200 → mxxMat1
  w_168_1 : Fin 7200 → mxxMat1
  w_168_2 : Fin 7200 → mxxMat1
  w_168_3 : Fin 7200 → mxxMat1
  w_169_0 : mxxMat1
  w_170_0 : mxxMat1
  w_172_0 : mxxMat1
  w_173_0 : mxxMat1
  w_175_0 : mxxMat1
  w_176_0 : mxxMat1
  w_178_0 : mxxMat1
  w_179_0 : mxxMat1
  w_181_0 : mxxMat1
  w_181_1 : mxxMat1
  w_181_2 : mxxMat1
  w_181_3 : mxxMat1
  w_182_0 : mxxMat6
  w_184_0 : mxxMat6
  w_186_0 : mxxMat6
  w_188_0 : mxxMat6
  w_189_0 : mxxMat6
  w_189_1 : mxxMat6
  w_189_2 : mxxMat6
  w_189_3 : mxxMat6
  w_190_0 : mxxMat1
  w_191_0 : mxxMat1
  w_192_0 : mxxMat2
  w_193_0 : mxxMat1
  w_194_0 : mxxMat3
  w_196_0 : mxxMat1
  w_197_0 : mxxMat1
  w_198_0 : mxxMat2
  w_199_0 : mxxMat1
  w_200_0 : mxxMat3
  w_203_0 : mxxMat1
  w_204_0 : mxxMat1
  w_205_0 : mxxMat3
  w_207_0 : mxxMat1
  w_208_0 : mxxMat3
  w_211_0 : mxxMat1
  w_212_0 : mxxMat1
  w_213_0 : mxxMat1
  w_214_0 : mxxMat1
  w_215_0 : mxxMat1
  w_216_0 : mxxMat2
  w_217_0 : mxxMat1
  w_218_0 : mxxMat3
  w_220_0 : mxxMat1
  w_221_0 : mxxMat1
  w_222_0 : mxxMat2
  w_223_0 : mxxMat1
  w_224_0 : mxxMat3
  w_227_0 : mxxMat1
  w_228_0 : mxxMat1
  w_229_0 : mxxMat2
  w_230_0 : mxxMat1
  w_231_0 : mxxMat1
  w_232_0 : mxxMat2
  w_233_0 : mxxMat1
  w_234_0 : mxxMat3
  w_236_0 : mxxMat1
  w_237_0 : mxxMat1
  w_238_0 : mxxMat2
  w_239_0 : mxxMat1
  w_240_0 : mxxMat3
  w_243_0 : mxxMat1
  w_244_0 : mxxMat3
  w_246_0 : mxxMat1
  w_247_0 : mxxMat3
  w_249_0 : mxxMat1
  w_250_0 : mxxMat3
  w_253_0 : mxxMat1
  w_254_0 : mxxMat1
  w_255_0 : mxxMat2
  w_256_0 : mxxMat1
  w_257_0 : mxxMat3
  w_260_0 : mxxMat1
  w_261_0 : mxxMat1
  w_262_0 : mxxMat3
  w_264_0 : mxxMat1
  w_265_0 : mxxMat3
  w_268_0 : mxxMat1
  w_269_0 : mxxMat3
  w_271_0 : mxxMat1
  w_272_0 : mxxMat3
  w_275_0 : mxxMat1
  w_276_0 : mxxMat1
  w_277_0 : mxxMat1
  w_278_0 : mxxMat1
  w_279_0 : mxxMat1
  w_280_0 : mxxMat2
  w_281_0 : mxxMat1
  w_282_0 : mxxMat3
  w_284_0 : mxxMat1
  w_285_0 : mxxMat1
  w_286_0 : mxxMat2
  w_287_0 : mxxMat1
  w_288_0 : mxxMat3
  w_291_0 : mxxMat1
  w_292_0 : mxxMat1
  w_293_0 : mxxMat2
  w_294_0 : mxxMat1
  w_295_0 : mxxMat3
  w_297_0 : mxxMat1
  w_298_0 : mxxMat3
  w_300_0 : mxxMat1
  w_301_0 : mxxMat3
  w_304_0 : mxxMat1
  w_305_0 : mxxMat1
  w_306_0 : mxxMat2
  w_307_0 : mxxMat1
  w_308_0 : mxxMat3
  w_311_0 : mxxMat1
  w_312_0 : mxxMat1
  w_313_0 : mxxMat3
  w_315_0 : mxxMat1
  w_316_0 : mxxMat3
  w_319_0 : mxxMat1
  w_320_0 : mxxMat1
  w_321_0 : mxxMat1
  w_322_0 : mxxMat1
  w_323_0 : mxxMat1
  w_324_0 : mxxMat2
  w_325_0 : mxxMat1
  w_326_0 : mxxMat3
  w_328_0 : mxxMat1
  w_329_0 : mxxMat1
  w_330_0 : mxxMat2
  w_331_0 : mxxMat1
  w_332_0 : mxxMat3
  w_335_0 : mxxMat1
  w_336_0 : mxxMat1
  w_337_0 : mxxMat3
  w_339_0 : mxxMat1
  w_340_0 : mxxMat3
  w_343_0 : mxxMat1
  w_344_0 : mxxMat1
  w_345_0 : mxxMat1
  w_346_0 : mxxMat1
  w_347_0 : mxxMat1
  w_348_0 : mxxMat2
  w_349_0 : mxxMat1
  w_350_0 : mxxMat3
  w_352_0 : mxxMat1
  w_353_0 : mxxMat1
  w_354_0 : mxxMat2
  w_355_0 : mxxMat1
  w_356_0 : mxxMat3
  w_359_0 : mxxMat1
  w_360_0 : mxxMat1
  w_361_0 : mxxMat3
  w_363_0 : mxxMat1
  w_364_0 : mxxMat3
  w_367_0 : mxxMat1
  w_368_0 : mxxMat1
  w_369_0 : mxxMat1

abbrev generatedRoot.constraints_17 (w_3_0 : Fin 4 → mxxMat1) (w_3_1 : Fin 4 → mxxMat1) (w_3_2 : Fin 4 → mxxMat1) (w_3_3 : Fin 4 → mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_4_1 : Fin 8 → mxxMat1) (w_4_2 : Fin 8 → mxxMat1) (w_4_3 : Fin 8 → mxxMat1) (w_5_0 : Fin 8 → mxxMat1) (w_5_1 : Fin 8 → mxxMat1) (w_5_2 : Fin 8 → mxxMat1) (w_5_3 : Fin 8 → mxxMat1) (w_6_0 : Fin 8 → mxxMat1) (w_6_1 : Fin 8 → mxxMat1) (w_6_2 : Fin 8 → mxxMat1) (w_6_3 : Fin 8 → mxxMat1) (w_7_0 : Fin 1 → mxxMat1) (w_7_1 : Fin 1 → mxxMat1) (w_7_2 : Fin 1 → mxxMat1) (w_7_3 : Fin 1 → mxxMat1) (w_8_0 : Fin 8 → mxxMat1) (w_8_1 : Fin 8 → mxxMat1) (w_8_2 : Fin 8 → mxxMat1) (w_8_3 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_9_1 : Fin 8 → mxxMat1) (w_9_2 : Fin 8 → mxxMat1) (w_9_3 : Fin 8 → mxxMat1) (w_10_0 : Fin 1 → mxxMat1) (w_10_1 : Fin 1 → mxxMat1) (w_10_2 : Fin 1 → mxxMat1) (w_10_3 : Fin 1 → mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_11_1 : Fin 1 → mxxMat1) (w_11_2 : Fin 1 → mxxMat1) (w_11_3 : Fin 1 → mxxMat1) (w_12_0 : Fin 1 → mxxMat1) (w_12_1 : Fin 1 → mxxMat1) (w_12_2 : Fin 1 → mxxMat1) (w_12_3 : Fin 1 → mxxMat1) (w_13_0 : Fin 1 → mxxMat1) (w_13_1 : Fin 1 → mxxMat1) (w_13_2 : Fin 1 → mxxMat1) (w_13_3 : Fin 1 → mxxMat1) (w_14_0 : Fin 1 → mxxMat1) (w_14_1 : Fin 1 → mxxMat1) (w_14_2 : Fin 1 → mxxMat1) (w_14_3 : Fin 1 → mxxMat1) (w_15_0 : Fin 1 → mxxMat1) (w_15_1 : Fin 1 → mxxMat1) (w_15_2 : Fin 1 → mxxMat1) (w_15_3 : Fin 1 → mxxMat1) (w_36_2 : Fin 1 → mxxMat1) (w_36_3 : Fin 1 → mxxMat1) (w_69_0 : mxxMat1) (w_69_1 : mxxMat1) (w_69_2 : mxxMat1) (w_69_3 : mxxMat1) (w_79_6 : Fin 72000 → mxxMat1) (w_79_7 : Fin 72000 → mxxMat1) (w_83_6 : Fin 72000 → mxxMat1) (w_83_7 : Fin 72000 → mxxMat1) (w_107_0 : mxxMat4) (w_107_1 : mxxMat4) (w_107_2 : mxxMat4) (w_107_3 : mxxMat4) (w_116_6 : Fin 16 → mxxMat1) (w_116_7 : Fin 16 → mxxMat1) (w_120_6 : Fin 16 → mxxMat1) (w_120_7 : Fin 16 → mxxMat1) (w_137_0 : mxxMat1) (w_137_1 : mxxMat1) (w_137_2 : mxxMat1) (w_137_3 : mxxMat1) (w_139_0 : mxxMat1) (w_139_1 : mxxMat1) (w_139_2 : mxxMat1) (w_139_3 : mxxMat1) (w_141_0 : mxxMat1) (w_141_1 : mxxMat1) (w_141_2 : mxxMat1) (w_141_3 : mxxMat1) (w_143_0 : mxxMat1) (w_143_1 : mxxMat1) (w_143_2 : mxxMat1) (w_143_3 : mxxMat1) (w_145_0 : mxxMat1) (w_145_1 : mxxMat1) (w_145_2 : mxxMat1) (w_145_3 : mxxMat1) (w_147_0 : mxxMat1) (w_147_1 : mxxMat1) (w_147_2 : mxxMat1) (w_147_3 : mxxMat1) (w_149_0 : mxxMat1) (w_149_1 : mxxMat1) (w_149_2 : mxxMat1) (w_149_3 : mxxMat1) (w_151_0 : mxxMat1) (w_151_1 : mxxMat1) (w_151_2 : mxxMat1) (w_151_3 : mxxMat1) (w_161_6 : Fin 14400 → mxxMat1) (w_161_7 : Fin 14400 → mxxMat1) (w_165_6 : Fin 14400 → mxxMat1) (w_165_7 : Fin 14400 → mxxMat1) (w_189_0 : mxxMat6) (w_189_1 : mxxMat6) (w_189_2 : mxxMat6) (w_189_3 : mxxMat6) (w_203_0 : mxxMat1) (w_211_0 : mxxMat1) (w_212_0 : mxxMat1) (w_213_0 : mxxMat1) (w_260_0 : mxxMat1) (w_275_0 : mxxMat1) (w_276_0 : mxxMat1) (w_277_0 : mxxMat1) (w_311_0 : mxxMat1) (w_319_0 : mxxMat1) (w_320_0 : mxxMat1) (w_321_0 : mxxMat1) (w_335_0 : mxxMat1) (w_343_0 : mxxMat1) (w_344_0 : mxxMat1) (w_345_0 : mxxMat1) (w_358_0 : mxxMat2) (w_359_0 : mxxMat1) (w_366_0 : mxxMat2) (w_367_0 : mxxMat1) (w_368_0 : mxxMat1) (w_369_0 : mxxMat1) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) : Prop :=
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_366_0 0 1 0 25 w_367_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_358_0 1 2 0 25 w_368_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_366_0 1 2 0 25 w_369_0 ∧
  outputs = (w_3_0, (w_3_1, (w_3_2, (w_3_3, (w_4_0, (w_4_1, (w_4_2, (w_4_3, (w_5_0, (w_5_1, (w_5_2, (w_5_3, (w_6_0, (w_6_1, (w_6_2, (w_6_3, (w_7_0, (w_7_1, (w_7_2, (w_7_3, (w_8_0, (w_8_1, (w_8_2, (w_8_3, (w_9_0, (w_9_1, (w_9_2, (w_9_3, (w_10_0, (w_10_1, (w_10_2, (w_10_3, (w_11_0, (w_11_1, (w_11_2, (w_11_3, (w_12_0, (w_12_1, (w_12_2, (w_12_3, (w_13_0, (w_13_1, (w_13_2, (w_13_3, (w_14_0, (w_14_1, (w_14_2, (w_14_3, (w_15_0, (w_15_1, (w_15_2, (w_15_3, (w_69_0, (w_69_1, (w_69_2, (w_69_3, (w_107_0, (w_107_1, (w_107_2, (w_107_3, (w_137_0, (w_137_1, (w_137_2, (w_137_3, (w_139_0, (w_139_1, (w_139_2, (w_139_3, (w_141_0, (w_141_1, (w_141_2, (w_141_3, (w_143_0, (w_143_1, (w_143_2, (w_143_3, (w_145_0, (w_145_1, (w_145_2, (w_145_3, (w_147_0, (w_147_1, (w_147_2, (w_147_3, (w_149_0, (w_149_1, (w_149_2, (w_149_3, (w_151_0, (w_151_1, (w_151_2, (w_151_3, (w_189_0, (w_189_1, (w_189_2, (w_189_3, (w_203_0, (w_211_0, (w_212_0, (w_213_0, (w_260_0, (w_275_0, (w_276_0, (w_277_0, (w_311_0, (w_319_0, (w_320_0, (w_321_0, (w_335_0, (w_343_0, (w_344_0, (w_345_0, (w_359_0, (w_367_0, (w_368_0, (w_369_0, (w_79_6, (w_79_7, (w_83_6, (w_83_7, (w_36_2, (w_36_3, (w_116_6, (w_116_7, (w_120_6, (w_120_7, (w_161_6, (w_161_7, (w_165_6, (w_165_7, ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

abbrev generatedRoot.constraints_16 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_342_0 : mxxMat2) (w_345_0 : mxxMat1) (w_346_0 : mxxMat1) (w_347_0 : mxxMat1) (w_348_0 : mxxMat2) (w_349_0 : mxxMat1) (w_350_0 : mxxMat3) (w_352_0 : mxxMat1) (w_353_0 : mxxMat1) (w_354_0 : mxxMat2) (w_355_0 : mxxMat1) (w_356_0 : mxxMat3) (w_358_0 : mxxMat2) (w_359_0 : mxxMat1) (w_360_0 : mxxMat1) (w_361_0 : mxxMat3) (w_363_0 : mxxMat1) (w_364_0 : mxxMat3) : Prop :=
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_342_0 1 2 0 25 w_345_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_346_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_347_0 ∧
  MxxRuntime.concatRows w_346_0 w_347_0 w_348_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_349_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_349_0 w_350_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_352_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_353_0 ∧
  MxxRuntime.concatRows w_352_0 w_353_0 w_354_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_355_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_355_0 w_356_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_358_0 0 1 0 25 w_359_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_360_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_360_0 w_361_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_363_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_363_0 w_364_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25

abbrev generatedRoot.constraints_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_325_0 : mxxMat1) (w_326_0 : mxxMat3) (w_328_0 : mxxMat1) (w_329_0 : mxxMat1) (w_330_0 : mxxMat2) (w_331_0 : mxxMat1) (w_332_0 : mxxMat3) (w_334_0 : mxxMat2) (w_335_0 : mxxMat1) (w_336_0 : mxxMat1) (w_337_0 : mxxMat3) (w_339_0 : mxxMat1) (w_340_0 : mxxMat3) (w_342_0 : mxxMat2) (w_343_0 : mxxMat1) (w_344_0 : mxxMat1) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_325_0 w_326_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_328_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_329_0 ∧
  MxxRuntime.concatRows w_328_0 w_329_0 w_330_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_331_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_331_0 w_332_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_334_0 0 1 0 25 w_335_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_336_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_336_0 w_337_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_339_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_339_0 w_340_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_342_0 0 1 0 25 w_343_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_334_0 1 2 0 25 w_344_0 ∧
  0 ≤ 1

abbrev generatedRoot.constraints_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_310_0 : mxxMat2) (w_311_0 : mxxMat1) (w_312_0 : mxxMat1) (w_313_0 : mxxMat3) (w_315_0 : mxxMat1) (w_316_0 : mxxMat3) (w_318_0 : mxxMat2) (w_319_0 : mxxMat1) (w_320_0 : mxxMat1) (w_321_0 : mxxMat1) (w_322_0 : mxxMat1) (w_323_0 : mxxMat1) (w_324_0 : mxxMat2) (w_325_0 : mxxMat1) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_310_0 0 1 0 25 w_311_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_312_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_312_0 w_313_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_315_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_315_0 w_316_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_318_0 0 1 0 25 w_319_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_310_0 1 2 0 25 w_320_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_318_0 1 2 0 25 w_321_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_322_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_323_0 ∧
  MxxRuntime.concatRows w_322_0 w_323_0 w_324_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_325_0

abbrev generatedRoot.constraints_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_290_0 : mxxMat2) (w_291_0 : mxxMat1) (w_292_0 : mxxMat1) (w_293_0 : mxxMat2) (w_294_0 : mxxMat1) (w_295_0 : mxxMat3) (w_297_0 : mxxMat1) (w_298_0 : mxxMat3) (w_300_0 : mxxMat1) (w_301_0 : mxxMat3) (w_303_0 : mxxMat2) (w_304_0 : mxxMat1) (w_305_0 : mxxMat1) (w_306_0 : mxxMat2) (w_307_0 : mxxMat1) (w_308_0 : mxxMat3) : Prop :=
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_290_0 1 2 0 25 w_292_0 ∧
  MxxRuntime.concatRows w_291_0 w_292_0 w_293_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_294_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_294_0 w_295_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_297_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_297_0 w_298_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_300_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_300_0 w_301_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_303_0 0 1 0 25 w_304_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_303_0 1 2 0 25 w_305_0 ∧
  MxxRuntime.concatRows w_304_0 w_305_0 w_306_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_307_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_307_0 w_308_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_12 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_259_0 : mxxMat2) (w_274_0 : mxxMat2) (w_276_0 : mxxMat1) (w_277_0 : mxxMat1) (w_278_0 : mxxMat1) (w_279_0 : mxxMat1) (w_280_0 : mxxMat2) (w_281_0 : mxxMat1) (w_282_0 : mxxMat3) (w_284_0 : mxxMat1) (w_285_0 : mxxMat1) (w_286_0 : mxxMat2) (w_287_0 : mxxMat1) (w_288_0 : mxxMat3) (w_290_0 : mxxMat2) (w_291_0 : mxxMat1) : Prop :=
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_259_0 1 2 0 25 w_276_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_274_0 1 2 0 25 w_277_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_278_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_279_0 ∧
  MxxRuntime.concatRows w_278_0 w_279_0 w_280_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_281_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_281_0 w_282_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_284_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_285_0 ∧
  MxxRuntime.concatRows w_284_0 w_285_0 w_286_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_287_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_287_0 w_288_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_290_0 0 1 0 25 w_291_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2

abbrev generatedRoot.constraints_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_259_0 : mxxMat2) (w_260_0 : mxxMat1) (w_261_0 : mxxMat1) (w_262_0 : mxxMat3) (w_264_0 : mxxMat1) (w_265_0 : mxxMat3) (w_267_0 : mxxMat2) (w_268_0 : mxxMat1) (w_269_0 : mxxMat3) (w_271_0 : mxxMat1) (w_272_0 : mxxMat3) (w_274_0 : mxxMat2) (w_275_0 : mxxMat1) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_259_0 0 1 0 25 w_260_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_261_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_261_0 w_262_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_264_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_264_0 w_265_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_267_0 0 1 0 25 w_268_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_268_0 w_269_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_267_0 1 2 0 25 w_271_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_271_0 w_272_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_274_0 0 1 0 25 w_275_0 ∧
  0 ≤ 1 ∧
  1 < 2

abbrev generatedRoot.constraints_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_242_0 : mxxMat2) (w_243_0 : mxxMat1) (w_244_0 : mxxMat3) (w_246_0 : mxxMat1) (w_247_0 : mxxMat3) (w_249_0 : mxxMat1) (w_250_0 : mxxMat3) (w_252_0 : mxxMat2) (w_253_0 : mxxMat1) (w_254_0 : mxxMat1) (w_255_0 : mxxMat2) (w_256_0 : mxxMat1) (w_257_0 : mxxMat3) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_243_0 w_244_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_246_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_246_0 w_247_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_249_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_249_0 w_250_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_252_0 0 1 0 25 w_253_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_252_0 1 2 0 25 w_254_0 ∧
  MxxRuntime.concatRows w_253_0 w_254_0 w_255_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_242_0 1 2 0 25 w_256_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_256_0 w_257_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_9 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_226_0 : mxxMat2) (w_227_0 : mxxMat1) (w_228_0 : mxxMat1) (w_229_0 : mxxMat2) (w_230_0 : mxxMat1) (w_231_0 : mxxMat1) (w_232_0 : mxxMat2) (w_233_0 : mxxMat1) (w_234_0 : mxxMat3) (w_236_0 : mxxMat1) (w_237_0 : mxxMat1) (w_238_0 : mxxMat2) (w_239_0 : mxxMat1) (w_240_0 : mxxMat3) (w_242_0 : mxxMat2) (w_243_0 : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_226_0 0 1 0 25 w_227_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_226_0 1 2 0 25 w_228_0 ∧
  MxxRuntime.concatRows w_227_0 w_228_0 w_229_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_230_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_231_0 ∧
  MxxRuntime.concatRows w_230_0 w_231_0 w_232_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_233_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_233_0 w_234_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_236_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_237_0 ∧
  MxxRuntime.concatRows w_236_0 w_237_0 w_238_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_239_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_239_0 w_240_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_242_0 0 1 0 25 w_243_0

abbrev generatedRoot.constraints_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_17_0 : ByteArray) (w_202_0 : mxxMat2) (w_207_0 : mxxMat1) (w_208_0 : mxxMat3) (w_210_0 : mxxMat2) (w_211_0 : mxxMat1) (w_212_0 : mxxMat1) (w_213_0 : mxxMat1) (w_214_0 : mxxMat1) (w_215_0 : mxxMat1) (w_216_0 : mxxMat2) (w_217_0 : mxxMat1) (w_218_0 : mxxMat3) (w_220_0 : mxxMat1) (w_221_0 : mxxMat1) (w_222_0 : mxxMat2) (w_223_0 : mxxMat1) (w_224_0 : mxxMat3) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_207_0 w_208_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_210_0 0 1 0 25 w_211_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_202_0 1 2 0 25 w_212_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_210_0 1 2 0 25 w_213_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_214_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_215_0 ∧
  MxxRuntime.concatRows w_214_0 w_215_0 w_216_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_217_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_217_0 w_218_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_220_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_221_0 ∧
  MxxRuntime.concatRows w_220_0 w_221_0 w_222_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_223_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_223_0 w_224_0

abbrev generatedRoot.constraints_7 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_17_0 : ByteArray) (w_152_0 : mxxMat7) (w_168_0 : Fin 7200 → mxxMat1) (w_168_1 : Fin 7200 → mxxMat1) (w_168_2 : Fin 7200 → mxxMat1) (w_168_3 : Fin 7200 → mxxMat1) (w_181_0 : mxxMat1) (w_181_1 : mxxMat1) (w_181_2 : mxxMat1) (w_181_3 : mxxMat1) (w_182_0 : mxxMat6) (w_183_0 : mxxMat7) (w_184_0 : mxxMat6) (w_185_0 : mxxMat7) (w_186_0 : mxxMat6) (w_187_0 : mxxMat7) (w_188_0 : mxxMat6) (w_189_0 : mxxMat6) (w_189_1 : mxxMat6) (w_189_2 : mxxMat6) (w_189_3 : mxxMat6) (w_190_0 : mxxMat1) (w_191_0 : mxxMat1) (w_192_0 : mxxMat2) (w_193_0 : mxxMat1) (w_194_0 : mxxMat3) (w_196_0 : mxxMat1) (w_197_0 : mxxMat1) (w_198_0 : mxxMat2) (w_199_0 : mxxMat1) (w_200_0 : mxxMat3) (w_202_0 : mxxMat2) (w_203_0 : mxxMat1) (w_204_0 : mxxMat1) (w_205_0 : mxxMat3) (w_207_0 : mxxMat1) : Prop :=
  MxxRuntime.tensorRuns w_152_0 w_181_0 w_182_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_183_0 w_181_1 w_184_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_185_0 w_181_2 w_186_0 ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.tensorRuns w_187_0 w_181_3 w_188_0 ∧
  0 ≤ (899) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × Unit) => sequential_generatedRoot_189 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_168_0, (w_168_1, (w_168_2, (w_168_3, ())))))))) next) (Int.toNat (899)) (w_182_0, (w_184_0, (w_186_0, (w_188_0, ())))) (w_189_0, (w_189_1, (w_189_2, (w_189_3, ())))) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_190_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_191_0 ∧
  MxxRuntime.concatRows w_190_0 w_191_0 w_192_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_193_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_193_0 w_194_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_196_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_197_0 ∧
  MxxRuntime.concatRows w_196_0 w_197_0 w_198_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_199_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_199_0 w_200_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_202_0 0 1 0 25 w_203_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_204_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_204_0 w_205_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_207_0

abbrev generatedRoot.constraints_6 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_17_0 : ByteArray) (w_153_0 : Fin 14400 → Int) (w_154_0 : Fin 14400 → Int) (w_159_0 : Fin 14400 → Int) (w_160_0 : Fin 14400 → Int) (w_163_0 : Fin 14400 → Int) (w_164_0 : Fin 14400 → Int) (w_2_0 : mxxMat0) (w_5_0 : Fin 8 → mxxMat1) (w_5_1 : Fin 8 → mxxMat1) (w_5_2 : Fin 8 → mxxMat1) (w_5_3 : Fin 8 → mxxMat1) (w_7_0 : Fin 1 → mxxMat1) (w_7_1 : Fin 1 → mxxMat1) (w_7_2 : Fin 1 → mxxMat1) (w_7_3 : Fin 1 → mxxMat1) (w_15_0 : Fin 1 → mxxMat1) (w_15_1 : Fin 1 → mxxMat1) (w_15_2 : Fin 1 → mxxMat1) (w_15_3 : Fin 1 → mxxMat1) (w_137_0 : mxxMat1) (w_137_1 : mxxMat1) (w_137_2 : mxxMat1) (w_137_3 : mxxMat1) (w_148_0 : mxxMat1) (w_149_0 : mxxMat1) (w_149_1 : mxxMat1) (w_149_2 : mxxMat1) (w_149_3 : mxxMat1) (w_150_0 : mxxMat1) (w_151_0 : mxxMat1) (w_151_1 : mxxMat1) (w_151_2 : mxxMat1) (w_151_3 : mxxMat1) (w_155_0 : mxxMat1) (w_156_0 : mxxMat1) (w_157_0 : mxxMat1) (w_158_0 : mxxMat1) (w_161_0 : Fin 14400 → mxxMat1) (w_161_1 : Fin 14400 → mxxMat1) (w_161_2 : Fin 14400 → mxxMat1) (w_161_3 : Fin 14400 → mxxMat1) (w_161_4 : Fin 14400 → mxxMat1) (w_161_5 : Fin 14400 → mxxMat1) (w_161_6 : Fin 14400 → mxxMat1) (w_161_7 : Fin 14400 → mxxMat1) (w_162_0 : Fin 7200 → mxxMat1) (w_162_1 : Fin 7200 → mxxMat1) (w_162_2 : Fin 7200 → mxxMat1) (w_162_3 : Fin 7200 → mxxMat1) (w_165_0 : Fin 14400 → mxxMat1) (w_165_1 : Fin 14400 → mxxMat1) (w_165_2 : Fin 14400 → mxxMat1) (w_165_3 : Fin 14400 → mxxMat1) (w_165_4 : Fin 14400 → mxxMat1) (w_165_5 : Fin 14400 → mxxMat1) (w_165_6 : Fin 14400 → mxxMat1) (w_165_7 : Fin 14400 → mxxMat1) (w_166_0 : Fin 7200 → mxxMat1) (w_166_1 : Fin 7200 → mxxMat1) (w_166_2 : Fin 7200 → mxxMat1) (w_166_3 : Fin 7200 → mxxMat1) (w_167_0 : Fin 7200 → mxxMat1) (w_167_1 : Fin 7200 → mxxMat1) (w_167_2 : Fin 7200 → mxxMat1) (w_167_3 : Fin 7200 → mxxMat1) (w_168_0 : Fin 7200 → mxxMat1) (w_168_1 : Fin 7200 → mxxMat1) (w_168_2 : Fin 7200 → mxxMat1) (w_168_3 : Fin 7200 → mxxMat1) (w_169_0 : mxxMat1) (w_170_0 : mxxMat1) (w_171_0 : mxxMat1) (w_172_0 : mxxMat1) (w_173_0 : mxxMat1) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_176_0 : mxxMat1) (w_177_0 : mxxMat1) (w_178_0 : mxxMat1) (w_179_0 : mxxMat1) (w_180_0 : mxxMat1) (w_181_0 : mxxMat1) (w_181_1 : mxxMat1) (w_181_2 : mxxMat1) (w_181_3 : mxxMat1) : Prop :=
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_149 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_17_0, (w_137_0, (w_137_2, (w_15_0, (w_137_1, (w_137_3, (w_15_2, (w_15_1, (w_15_3, ()))))))))))))) next) (Int.toNat (1)) (w_148_0, (w_148_0, (w_148_0, (w_148_0, ())))) (w_149_0, (w_149_1, (w_149_2, (w_149_3, ())))) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_151 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_17_0, (w_137_0, (w_137_2, (w_7_0, (w_137_1, (w_137_3, (w_7_2, (w_7_1, (w_7_3, ()))))))))))))) next) (Int.toNat (1)) (w_150_0, (w_150_0, (w_150_0, (w_150_0, ())))) (w_151_0, (w_151_1, (w_151_2, (w_151_3, ())))) ∧
  0 ≤ (0) ∧
  (0) < 900 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_155_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_156_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_157_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_158_0 ∧
  (14400) = 14400 ∧
  (∀ i : Fin 14400, parallel_generatedRoot_161 backend hashModel params i (w_17_0, ((w_154_0 i), (w_155_0, (w_156_0, (w_0_0, (w_157_0, (w_158_0, (w_1_0, ((w_159_0 i), ((w_160_0 i), (w_2_0, ()))))))))))) ((w_161_0 i), ((w_161_1 i), ((w_161_2 i), ((w_161_3 i), ((w_161_4 i), ((w_161_5 i), ((w_161_6 i), ((w_161_7 i), ()))))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_162 backend hashModel params i (w_161_0, (w_161_1, (w_161_2, (w_161_3, ())))) ((w_162_0 i), ((w_162_1 i), ((w_162_2 i), ((w_162_3 i), ()))))) ∧
  (14400) = 14400 ∧
  (∀ i : Fin 14400, parallel_generatedRoot_165 backend hashModel params i (w_17_0, ((w_153_0 i), (w_162_0, (w_162_2, (w_0_0, (w_162_1, (w_162_3, (w_1_0, ((w_163_0 i), ((w_164_0 i), (w_2_0, ()))))))))))) ((w_165_0 i), ((w_165_1 i), ((w_165_2 i), ((w_165_3 i), ((w_165_4 i), ((w_165_5 i), ((w_165_6 i), ((w_165_7 i), ()))))))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_166 backend hashModel params i (w_165_0, (w_165_1, (w_165_2, (w_165_3, ())))) ((w_166_0 i), ((w_166_1 i), ((w_166_2 i), ((w_166_3 i), ()))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_167 backend hashModel params i (w_17_0, ((w_166_0 i), ((w_166_2 i), (w_5_0, ((w_166_1 i), ((w_166_3 i), (w_5_2, (w_5_1, (w_5_3, ()))))))))) ((w_167_0 i), ((w_167_1 i), ((w_167_2 i), ((w_167_3 i), ()))))) ∧
  (7200) = 7200 ∧
  (∀ i : Fin 7200, parallel_generatedRoot_168 backend hashModel params i ((w_167_0 i), ((w_167_1 i), ((w_167_2 i), ((w_167_3 i), ())))) ((w_168_0 i), ((w_168_1 i), ((w_168_2 i), ((w_168_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_168_0 (0) w_169_0 ∧
  MxxRuntime.familyGetStatic w_168_0 (1) w_170_0 ∧
  MxxRuntime.familyGetStatic w_168_1 (0) w_172_0 ∧
  MxxRuntime.familyGetStatic w_168_1 (1) w_173_0 ∧
  MxxRuntime.familyGetStatic w_168_2 (0) w_175_0 ∧
  MxxRuntime.familyGetStatic w_168_2 (1) w_176_0 ∧
  MxxRuntime.familyGetStatic w_168_3 (0) w_178_0 ∧
  MxxRuntime.familyGetStatic w_168_3 (1) w_179_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_181 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_168_0, (w_168_1, (w_168_2, (w_168_3, ())))))))) next) (Int.toNat (6)) (w_171_0, (w_174_0, (w_177_0, (w_180_0, ())))) (w_181_0, (w_181_1, (w_181_2, (w_181_3, ()))))

abbrev generatedRoot.constraints_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_17_0 : ByteArray) (w_108_0 : Fin 16 → Int) (w_118_0 : Fin 16 → Int) (w_119_0 : Fin 16 → Int) (w_2_0 : mxxMat0) (w_6_0 : Fin 8 → mxxMat1) (w_6_1 : Fin 8 → mxxMat1) (w_6_2 : Fin 8 → mxxMat1) (w_6_3 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_9_1 : Fin 8 → mxxMat1) (w_9_2 : Fin 8 → mxxMat1) (w_9_3 : Fin 8 → mxxMat1) (w_10_0 : Fin 1 → mxxMat1) (w_10_1 : Fin 1 → mxxMat1) (w_10_2 : Fin 1 → mxxMat1) (w_10_3 : Fin 1 → mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_11_1 : Fin 1 → mxxMat1) (w_11_2 : Fin 1 → mxxMat1) (w_11_3 : Fin 1 → mxxMat1) (w_12_0 : Fin 1 → mxxMat1) (w_12_1 : Fin 1 → mxxMat1) (w_12_2 : Fin 1 → mxxMat1) (w_12_3 : Fin 1 → mxxMat1) (w_13_0 : Fin 1 → mxxMat1) (w_13_1 : Fin 1 → mxxMat1) (w_13_2 : Fin 1 → mxxMat1) (w_13_3 : Fin 1 → mxxMat1) (w_14_0 : Fin 1 → mxxMat1) (w_14_1 : Fin 1 → mxxMat1) (w_14_2 : Fin 1 → mxxMat1) (w_14_3 : Fin 1 → mxxMat1) (w_116_0 : Fin 16 → mxxMat1) (w_116_1 : Fin 16 → mxxMat1) (w_116_2 : Fin 16 → mxxMat1) (w_116_3 : Fin 16 → mxxMat1) (w_117_0 : Fin 8 → mxxMat1) (w_117_1 : Fin 8 → mxxMat1) (w_117_2 : Fin 8 → mxxMat1) (w_117_3 : Fin 8 → mxxMat1) (w_120_0 : Fin 16 → mxxMat1) (w_120_1 : Fin 16 → mxxMat1) (w_120_2 : Fin 16 → mxxMat1) (w_120_3 : Fin 16 → mxxMat1) (w_120_4 : Fin 16 → mxxMat1) (w_120_5 : Fin 16 → mxxMat1) (w_120_6 : Fin 16 → mxxMat1) (w_120_7 : Fin 16 → mxxMat1) (w_121_0 : Fin 8 → mxxMat1) (w_121_1 : Fin 8 → mxxMat1) (w_121_2 : Fin 8 → mxxMat1) (w_121_3 : Fin 8 → mxxMat1) (w_122_0 : Fin 8 → mxxMat1) (w_122_1 : Fin 8 → mxxMat1) (w_122_2 : Fin 8 → mxxMat1) (w_122_3 : Fin 8 → mxxMat1) (w_123_0 : Fin 8 → mxxMat1) (w_123_1 : Fin 8 → mxxMat1) (w_123_2 : Fin 8 → mxxMat1) (w_123_3 : Fin 8 → mxxMat1) (w_124_0 : Fin 8 → mxxMat1) (w_124_1 : Fin 8 → mxxMat1) (w_124_2 : Fin 8 → mxxMat1) (w_124_3 : Fin 8 → mxxMat1) (w_125_0 : mxxMat1) (w_126_0 : mxxMat1) (w_127_0 : mxxMat1) (w_128_0 : mxxMat1) (w_129_0 : mxxMat1) (w_130_0 : mxxMat1) (w_131_0 : mxxMat1) (w_132_0 : mxxMat1) (w_133_0 : mxxMat1) (w_134_0 : mxxMat1) (w_135_0 : mxxMat1) (w_136_0 : mxxMat1) (w_137_0 : mxxMat1) (w_137_1 : mxxMat1) (w_137_2 : mxxMat1) (w_137_3 : mxxMat1) (w_138_0 : mxxMat1) (w_139_0 : mxxMat1) (w_139_1 : mxxMat1) (w_139_2 : mxxMat1) (w_139_3 : mxxMat1) (w_140_0 : mxxMat1) (w_141_0 : mxxMat1) (w_141_1 : mxxMat1) (w_141_2 : mxxMat1) (w_141_3 : mxxMat1) (w_142_0 : mxxMat1) (w_143_0 : mxxMat1) (w_143_1 : mxxMat1) (w_143_2 : mxxMat1) (w_143_3 : mxxMat1) (w_144_0 : mxxMat1) (w_145_0 : mxxMat1) (w_145_1 : mxxMat1) (w_145_2 : mxxMat1) (w_145_3 : mxxMat1) (w_146_0 : mxxMat1) (w_147_0 : mxxMat1) (w_147_1 : mxxMat1) (w_147_2 : mxxMat1) (w_147_3 : mxxMat1) : Prop :=
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_117 backend hashModel params i (w_116_0, (w_116_1, (w_116_2, (w_116_3, ())))) ((w_117_0 i), ((w_117_1 i), ((w_117_2 i), ((w_117_3 i), ()))))) ∧
  (16) = 16 ∧
  (∀ i : Fin 16, parallel_generatedRoot_120 backend hashModel params i (w_17_0, ((w_108_0 i), (w_117_0, (w_117_2, (w_0_0, (w_117_1, (w_117_3, (w_1_0, ((w_118_0 i), ((w_119_0 i), (w_2_0, ()))))))))))) ((w_120_0 i), ((w_120_1 i), ((w_120_2 i), ((w_120_3 i), ((w_120_4 i), ((w_120_5 i), ((w_120_6 i), ((w_120_7 i), ()))))))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_121 backend hashModel params i (w_120_0, (w_120_1, (w_120_2, (w_120_3, ())))) ((w_121_0 i), ((w_121_1 i), ((w_121_2 i), ((w_121_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_122 backend hashModel params i (w_17_0, ((w_121_0 i), ((w_121_2 i), (w_6_0, ((w_121_1 i), ((w_121_3 i), (w_6_2, (w_6_1, (w_6_3, ()))))))))) ((w_122_0 i), ((w_122_1 i), ((w_122_2 i), ((w_122_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_123 backend hashModel params i (w_17_0, ((w_122_0 i), ((w_122_2 i), (w_9_0, ((w_122_1 i), ((w_122_3 i), (w_9_2, (w_9_1, (w_9_3, ()))))))))) ((w_123_0 i), ((w_123_1 i), ((w_123_2 i), ((w_123_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_124 backend hashModel params i ((w_123_0 i), ((w_123_1 i), ((w_123_2 i), ((w_123_3 i), ())))) ((w_124_0 i), ((w_124_1 i), ((w_124_2 i), ((w_124_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_124_0 (0) w_125_0 ∧
  MxxRuntime.familyGetStatic w_124_0 (1) w_126_0 ∧
  MxxRuntime.familyGetStatic w_124_1 (0) w_128_0 ∧
  MxxRuntime.familyGetStatic w_124_1 (1) w_129_0 ∧
  MxxRuntime.familyGetStatic w_124_2 (0) w_131_0 ∧
  MxxRuntime.familyGetStatic w_124_2 (1) w_132_0 ∧
  MxxRuntime.familyGetStatic w_124_3 (0) w_134_0 ∧
  MxxRuntime.familyGetStatic w_124_3 (1) w_135_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_137 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_124_0, (w_124_1, (w_124_2, (w_124_3, ())))))))) next) (Int.toNat (6)) (w_127_0, (w_130_0, (w_133_0, (w_136_0, ())))) (w_137_0, (w_137_1, (w_137_2, (w_137_3, ())))) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_139 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_17_0, (w_137_0, (w_137_2, (w_10_0, (w_137_1, (w_137_3, (w_10_2, (w_10_1, (w_10_3, ()))))))))))))) next) (Int.toNat (1)) (w_138_0, (w_138_0, (w_138_0, (w_138_0, ())))) (w_139_0, (w_139_1, (w_139_2, (w_139_3, ())))) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_141 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_17_0, (w_137_0, (w_137_2, (w_11_0, (w_137_1, (w_137_3, (w_11_2, (w_11_1, (w_11_3, ()))))))))))))) next) (Int.toNat (1)) (w_140_0, (w_140_0, (w_140_0, (w_140_0, ())))) (w_141_0, (w_141_1, (w_141_2, (w_141_3, ())))) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_143 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_17_0, (w_137_0, (w_137_2, (w_12_0, (w_137_1, (w_137_3, (w_12_2, (w_12_1, (w_12_3, ()))))))))))))) next) (Int.toNat (1)) (w_142_0, (w_142_0, (w_142_0, (w_142_0, ())))) (w_143_0, (w_143_1, (w_143_2, (w_143_3, ())))) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_145 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_17_0, (w_137_0, (w_137_2, (w_13_0, (w_137_1, (w_137_3, (w_13_2, (w_13_1, (w_13_3, ()))))))))))))) next) (Int.toNat (1)) (w_144_0, (w_144_0, (w_144_0, (w_144_0, ())))) (w_145_0, (w_145_1, (w_145_2, (w_145_3, ())))) ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_147 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_17_0, (w_137_0, (w_137_2, (w_14_0, (w_137_1, (w_137_3, (w_14_2, (w_14_1, (w_14_3, ()))))))))))))) next) (Int.toNat (1)) (w_146_0, (w_146_0, (w_146_0, (w_146_0, ())))) (w_147_0, (w_147_1, (w_147_2, (w_147_3, ()))))

abbrev generatedRoot.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_17_0 : ByteArray) (w_109_0 : Fin 16 → Int) (w_114_0 : Fin 16 → Int) (w_115_0 : Fin 16 → Int) (w_2_0 : mxxMat0) (w_4_0 : Fin 8 → mxxMat1) (w_4_1 : Fin 8 → mxxMat1) (w_4_2 : Fin 8 → mxxMat1) (w_4_3 : Fin 8 → mxxMat1) (w_70_0 : mxxMat5) (w_84_0 : Fin 36000 → mxxMat1) (w_84_1 : Fin 36000 → mxxMat1) (w_84_2 : Fin 36000 → mxxMat1) (w_84_3 : Fin 36000 → mxxMat1) (w_85_0 : Fin 36000 → mxxMat1) (w_85_1 : Fin 36000 → mxxMat1) (w_85_2 : Fin 36000 → mxxMat1) (w_85_3 : Fin 36000 → mxxMat1) (w_86_0 : Fin 36000 → mxxMat1) (w_86_1 : Fin 36000 → mxxMat1) (w_86_2 : Fin 36000 → mxxMat1) (w_86_3 : Fin 36000 → mxxMat1) (w_87_0 : mxxMat1) (w_88_0 : mxxMat1) (w_89_0 : mxxMat1) (w_90_0 : mxxMat1) (w_91_0 : mxxMat1) (w_92_0 : mxxMat1) (w_93_0 : mxxMat1) (w_94_0 : mxxMat1) (w_95_0 : mxxMat1) (w_96_0 : mxxMat1) (w_97_0 : mxxMat1) (w_98_0 : mxxMat1) (w_99_0 : mxxMat1) (w_99_1 : mxxMat1) (w_99_2 : mxxMat1) (w_99_3 : mxxMat1) (w_100_0 : mxxMat4) (w_101_0 : mxxMat5) (w_102_0 : mxxMat4) (w_103_0 : mxxMat5) (w_104_0 : mxxMat4) (w_105_0 : mxxMat5) (w_106_0 : mxxMat4) (w_107_0 : mxxMat4) (w_107_1 : mxxMat4) (w_107_2 : mxxMat4) (w_107_3 : mxxMat4) (w_110_0 : mxxMat1) (w_111_0 : mxxMat1) (w_112_0 : mxxMat1) (w_113_0 : mxxMat1) (w_116_0 : Fin 16 → mxxMat1) (w_116_1 : Fin 16 → mxxMat1) (w_116_2 : Fin 16 → mxxMat1) (w_116_3 : Fin 16 → mxxMat1) (w_116_4 : Fin 16 → mxxMat1) (w_116_5 : Fin 16 → mxxMat1) (w_116_6 : Fin 16 → mxxMat1) (w_116_7 : Fin 16 → mxxMat1) : Prop :=
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_85 backend hashModel params i (w_17_0, ((w_84_0 i), ((w_84_2 i), (w_4_0, ((w_84_1 i), ((w_84_3 i), (w_4_2, (w_4_1, (w_4_3, ()))))))))) ((w_85_0 i), ((w_85_1 i), ((w_85_2 i), ((w_85_3 i), ()))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_86 backend hashModel params i ((w_85_0 i), ((w_85_1 i), ((w_85_2 i), ((w_85_3 i), ())))) ((w_86_0 i), ((w_86_1 i), ((w_86_2 i), ((w_86_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_86_0 (0) w_87_0 ∧
  MxxRuntime.familyGetStatic w_86_0 (1) w_88_0 ∧
  MxxRuntime.familyGetStatic w_86_1 (0) w_90_0 ∧
  MxxRuntime.familyGetStatic w_86_1 (1) w_91_0 ∧
  MxxRuntime.familyGetStatic w_86_2 (0) w_93_0 ∧
  MxxRuntime.familyGetStatic w_86_2 (1) w_94_0 ∧
  MxxRuntime.familyGetStatic w_86_3 (0) w_96_0 ∧
  MxxRuntime.familyGetStatic w_86_3 (1) w_97_0 ∧
  0 ≤ (6) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_99 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_86_0, (w_86_1, (w_86_2, (w_86_3, ())))))))) next) (Int.toNat (6)) (w_89_0, (w_92_0, (w_95_0, (w_98_0, ())))) (w_99_0, (w_99_1, (w_99_2, (w_99_3, ())))) ∧
  MxxRuntime.tensorRuns w_70_0 w_99_0 w_100_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_101_0 w_99_1 w_102_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_103_0 w_99_2 w_104_0 ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.tensorRuns w_105_0 w_99_3 w_106_0 ∧
  0 ≤ (4499) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × Unit) => sequential_generatedRoot_107 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_86_0, (w_86_1, (w_86_2, (w_86_3, ())))))))) next) (Int.toNat (4499)) (w_100_0, (w_102_0, (w_104_0, (w_106_0, ())))) (w_107_0, (w_107_1, (w_107_2, (w_107_3, ())))) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_110_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_111_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_112_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_113_0 ∧
  (16) = 16 ∧
  (∀ i : Fin 16, parallel_generatedRoot_116 backend hashModel params i (w_17_0, ((w_109_0 i), (w_110_0, (w_111_0, (w_0_0, (w_112_0, (w_113_0, (w_1_0, ((w_114_0 i), ((w_115_0 i), (w_2_0, ()))))))))))) ((w_116_0 i), ((w_116_1 i), ((w_116_2 i), ((w_116_3 i), ((w_116_4 i), ((w_116_5 i), ((w_116_6 i), ((w_116_7 i), ())))))))))

abbrev generatedRoot.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_17_0 : ByteArray) (w_71_0 : Fin 72000 → Int) (w_72_0 : Fin 72000 → Int) (w_77_0 : Fin 72000 → Int) (w_78_0 : Fin 72000 → Int) (w_81_0 : Fin 72000 → Int) (w_82_0 : Fin 72000 → Int) (w_2_0 : mxxMat0) (w_3_0 : Fin 4 → mxxMat1) (w_3_1 : Fin 4 → mxxMat1) (w_3_2 : Fin 4 → mxxMat1) (w_3_3 : Fin 4 → mxxMat1) (w_16_0 : mxxMat1) (w_56_0 : mxxMat1) (w_57_0 : mxxMat1) (w_62_0 : mxxMat1) (w_63_0 : mxxMat3) (w_64_0 : mxxMat3) (w_66_0 : mxxMat2) (w_67_0 : mxxMat1) (w_68_0 : mxxMat1) (w_69_0 : mxxMat1) (w_69_1 : mxxMat1) (w_69_2 : mxxMat1) (w_69_3 : mxxMat1) (w_73_0 : mxxMat1) (w_74_0 : mxxMat1) (w_75_0 : mxxMat1) (w_76_0 : mxxMat1) (w_79_0 : Fin 72000 → mxxMat1) (w_79_1 : Fin 72000 → mxxMat1) (w_79_2 : Fin 72000 → mxxMat1) (w_79_3 : Fin 72000 → mxxMat1) (w_79_4 : Fin 72000 → mxxMat1) (w_79_5 : Fin 72000 → mxxMat1) (w_79_6 : Fin 72000 → mxxMat1) (w_79_7 : Fin 72000 → mxxMat1) (w_80_0 : Fin 36000 → mxxMat1) (w_80_1 : Fin 36000 → mxxMat1) (w_80_2 : Fin 36000 → mxxMat1) (w_80_3 : Fin 36000 → mxxMat1) (w_83_0 : Fin 72000 → mxxMat1) (w_83_1 : Fin 72000 → mxxMat1) (w_83_2 : Fin 72000 → mxxMat1) (w_83_3 : Fin 72000 → mxxMat1) (w_83_4 : Fin 72000 → mxxMat1) (w_83_5 : Fin 72000 → mxxMat1) (w_83_6 : Fin 72000 → mxxMat1) (w_83_7 : Fin 72000 → mxxMat1) (w_84_0 : Fin 36000 → mxxMat1) (w_84_1 : Fin 36000 → mxxMat1) (w_84_2 : Fin 36000 → mxxMat1) (w_84_3 : Fin 36000 → mxxMat1) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_62_0 w_63_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_63_0 w_64_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_66_0 0 1 0 25 w_67_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_66_0 1 2 0 25 w_68_0 ∧
  0 ≤ (4) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_generatedRoot_69 backend hashModel params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_17_0, (w_56_0, (w_57_0, (w_3_0, (w_67_0, (w_68_0, (w_3_2, (w_3_1, (w_3_3, ()))))))))))))) next) (Int.toNat (4)) (w_16_0, (w_16_0, (w_16_0, (w_16_0, ())))) (w_69_0, (w_69_1, (w_69_2, (w_69_3, ())))) ∧
  0 ≤ (0) ∧
  (0) < 4500 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_73_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_74_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_75_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_76_0 ∧
  (72000) = 72000 ∧
  (∀ i : Fin 72000, parallel_generatedRoot_79 backend hashModel params i (w_17_0, ((w_72_0 i), (w_73_0, (w_74_0, (w_0_0, (w_75_0, (w_76_0, (w_1_0, ((w_77_0 i), ((w_78_0 i), (w_2_0, ()))))))))))) ((w_79_0 i), ((w_79_1 i), ((w_79_2 i), ((w_79_3 i), ((w_79_4 i), ((w_79_5 i), ((w_79_6 i), ((w_79_7 i), ()))))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_80 backend hashModel params i (w_79_0, (w_79_1, (w_79_2, (w_79_3, ())))) ((w_80_0 i), ((w_80_1 i), ((w_80_2 i), ((w_80_3 i), ()))))) ∧
  (72000) = 72000 ∧
  (∀ i : Fin 72000, parallel_generatedRoot_83 backend hashModel params i (w_17_0, ((w_71_0 i), (w_80_0, (w_80_2, (w_0_0, (w_80_1, (w_80_3, (w_1_0, ((w_81_0 i), ((w_82_0 i), (w_2_0, ()))))))))))) ((w_83_0 i), ((w_83_1 i), ((w_83_2 i), ((w_83_3 i), ((w_83_4 i), ((w_83_5 i), ((w_83_6 i), ((w_83_7 i), ()))))))))) ∧
  (36000) = 36000 ∧
  (∀ i : Fin 36000, parallel_generatedRoot_84 backend hashModel params i (w_83_0, (w_83_1, (w_83_2, (w_83_3, ())))) ((w_84_0 i), ((w_84_1 i), ((w_84_2 i), ((w_84_3 i), ())))))

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (w_36_1 : Fin 1 → mxxMat1) (w_36_2 : Fin 1 → mxxMat1) (w_36_3 : Fin 1 → mxxMat1) (w_47_0 : mxxMat2) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat2) (w_51_0 : mxxMat1) (w_52_0 : mxxMat3) (w_53_0 : mxxMat3) (w_55_0 : mxxMat2) (w_56_0 : mxxMat1) (w_57_0 : mxxMat1) (w_58_0 : mxxMat1) (w_59_0 : mxxMat3) (w_60_0 : mxxMat3) (w_62_0 : mxxMat1) : Prop :=
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 25 w_48_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_47_0 1 2 0 25 w_49_0 ∧
  MxxRuntime.concatRows w_48_0 w_49_0 w_50_0 ∧
  MxxRuntime.familyGetStatic w_36_2 (0) w_51_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_51_0 w_52_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_52_0 w_53_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_55_0 0 1 0 25 w_56_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_55_0 1 2 0 25 w_57_0 ∧
  MxxRuntime.familyGetStatic w_36_1 (0) w_58_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_58_0 w_59_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_59_0 w_60_0 ∧
  MxxRuntime.familyGetStatic w_36_3 (0) w_62_0

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_17_0 : ByteArray) (w_34_0 : Fin 1 → Int) (w_35_0 : Fin 1 → Int) (w_2_0 : mxxMat0) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat2) (w_27_0 : mxxMat1) (w_28_0 : mxxMat3) (w_30_0 : mxxMat2) (w_31_0 : mxxMat1) (w_32_0 : mxxMat1) (w_33_0 : mxxMat2) (w_36_0 : Fin 1 → mxxMat1) (w_36_1 : Fin 1 → mxxMat1) (w_36_2 : Fin 1 → mxxMat1) (w_36_3 : Fin 1 → mxxMat1) (w_37_0 : mxxMat1) (w_38_0 : mxxMat3) (w_39_0 : mxxMat3) (w_41_0 : mxxMat1) (w_42_0 : mxxMat3) (w_44_0 : mxxMat1) (w_45_0 : mxxMat3) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_25_0 ∧
  MxxRuntime.concatRows w_24_0 w_25_0 w_26_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_27_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_30_0 0 1 0 25 w_31_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 25 ∧
  25 ≤ 25 ∧
  MxxRuntime.sliceMatrix w_30_0 1 2 0 25 w_32_0 ∧
  MxxRuntime.concatRows w_31_0 w_32_0 w_33_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_36 backend hashModel params i (w_0_0, (w_1_0, ((w_34_0 i), ((w_35_0 i), (w_2_0, ()))))) ((w_36_0 i), ((w_36_1 i), ((w_36_2 i), ((w_36_3 i), ()))))) ∧
  MxxRuntime.familyGetStatic w_36_0 (0) w_37_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_37_0 w_38_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 2048 w_38_0 w_39_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_41_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_41_0 w_42_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_44_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_44_0 w_45_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_17_0 : ByteArray) (w_2_0 : mxxMat0) (w_3_0 : Fin 4 → mxxMat1) (w_3_1 : Fin 4 → mxxMat1) (w_3_2 : Fin 4 → mxxMat1) (w_3_3 : Fin 4 → mxxMat1) (w_4_0 : Fin 8 → mxxMat1) (w_4_1 : Fin 8 → mxxMat1) (w_4_2 : Fin 8 → mxxMat1) (w_4_3 : Fin 8 → mxxMat1) (w_5_0 : Fin 8 → mxxMat1) (w_5_1 : Fin 8 → mxxMat1) (w_5_2 : Fin 8 → mxxMat1) (w_5_3 : Fin 8 → mxxMat1) (w_6_0 : Fin 8 → mxxMat1) (w_6_1 : Fin 8 → mxxMat1) (w_6_2 : Fin 8 → mxxMat1) (w_6_3 : Fin 8 → mxxMat1) (w_7_0 : Fin 1 → mxxMat1) (w_7_1 : Fin 1 → mxxMat1) (w_7_2 : Fin 1 → mxxMat1) (w_7_3 : Fin 1 → mxxMat1) (w_8_0 : Fin 8 → mxxMat1) (w_8_1 : Fin 8 → mxxMat1) (w_8_2 : Fin 8 → mxxMat1) (w_8_3 : Fin 8 → mxxMat1) (w_9_0 : Fin 8 → mxxMat1) (w_9_1 : Fin 8 → mxxMat1) (w_9_2 : Fin 8 → mxxMat1) (w_9_3 : Fin 8 → mxxMat1) (w_10_0 : Fin 1 → mxxMat1) (w_10_1 : Fin 1 → mxxMat1) (w_10_2 : Fin 1 → mxxMat1) (w_10_3 : Fin 1 → mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_11_1 : Fin 1 → mxxMat1) (w_11_2 : Fin 1 → mxxMat1) (w_11_3 : Fin 1 → mxxMat1) (w_12_0 : Fin 1 → mxxMat1) (w_12_1 : Fin 1 → mxxMat1) (w_12_2 : Fin 1 → mxxMat1) (w_12_3 : Fin 1 → mxxMat1) (w_13_0 : Fin 1 → mxxMat1) (w_13_1 : Fin 1 → mxxMat1) (w_13_2 : Fin 1 → mxxMat1) (w_13_3 : Fin 1 → mxxMat1) (w_14_0 : Fin 1 → mxxMat1) (w_14_1 : Fin 1 → mxxMat1) (w_14_2 : Fin 1 → mxxMat1) (w_14_3 : Fin 1 → mxxMat1) (w_15_0 : Fin 1 → mxxMat1) (w_15_1 : Fin 1 → mxxMat1) (w_15_2 : Fin 1 → mxxMat1) (w_15_3 : Fin 1 → mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat2) (w_21_0 : mxxMat1) (w_22_0 : mxxMat3) (w_24_0 : mxxMat1) : Prop :=
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_3 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_3_0 i), ((w_3_1 i), ((w_3_2 i), ((w_3_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_4 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_4_0 i), ((w_4_1 i), ((w_4_2 i), ((w_4_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_5 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_5_0 i), ((w_5_1 i), ((w_5_2 i), ((w_5_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_6 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_6_0 i), ((w_6_1 i), ((w_6_2 i), ((w_6_3 i), ()))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_7 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_7_0 i), ((w_7_1 i), ((w_7_2 i), ((w_7_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_8 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_8_0 i), ((w_8_1 i), ((w_8_2 i), ((w_8_3 i), ()))))) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_9 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_9_0 i), ((w_9_1 i), ((w_9_2 i), ((w_9_3 i), ()))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_10 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_10_0 i), ((w_10_1 i), ((w_10_2 i), ((w_10_3 i), ()))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_11 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_11_0 i), ((w_11_1 i), ((w_11_2 i), ((w_11_3 i), ()))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_12 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_12_0 i), ((w_12_1 i), ((w_12_2 i), ((w_12_3 i), ()))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_13 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_13_0 i), ((w_13_1 i), ((w_13_2 i), ((w_13_3 i), ()))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_14 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_14_0 i), ((w_14_1 i), ((w_14_2 i), ((w_14_3 i), ()))))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_15 backend hashModel params i (w_0_0, (w_1_0, (w_2_0, ()))) ((w_15_0 i), ((w_15_1 i), ((w_15_2 i), ((w_15_3 i), ()))))) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_18_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_19_0 ∧
  MxxRuntime.concatRows w_18_0 w_19_0 w_20_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 25 w_21_0 w_22_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_24_0

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × mxxMat0 × ByteArray × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × Unit) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_17_0 : ByteArray := inputs.2.2.1
  let w_34_0 : Fin 1 → Int := inputs.2.2.2.1
  let w_35_0 : Fin 1 → Int := inputs.2.2.2.2.1
  let w_71_0 : Fin 72000 → Int := inputs.2.2.2.2.2.1
  let w_72_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.1
  let w_77_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.1
  let w_78_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.1
  let w_81_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.2.1
  let w_82_0 : Fin 72000 → Int := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_108_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_109_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_114_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_115_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_118_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_119_0 : Fin 16 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_153_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_154_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_159_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_160_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_163_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_164_0 : Fin 14400 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 : mxxMat0 := (1 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_3_1 := witness.w_3_1
  let w_3_2 := witness.w_3_2
  let w_3_3 := witness.w_3_3
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_4_2 := witness.w_4_2
  let w_4_3 := witness.w_4_3
  let w_5_0 := witness.w_5_0
  let w_5_1 := witness.w_5_1
  let w_5_2 := witness.w_5_2
  let w_5_3 := witness.w_5_3
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_6_2 := witness.w_6_2
  let w_6_3 := witness.w_6_3
  let w_7_0 := witness.w_7_0
  let w_7_1 := witness.w_7_1
  let w_7_2 := witness.w_7_2
  let w_7_3 := witness.w_7_3
  let w_8_0 := witness.w_8_0
  let w_8_1 := witness.w_8_1
  let w_8_2 := witness.w_8_2
  let w_8_3 := witness.w_8_3
  let w_9_0 := witness.w_9_0
  let w_9_1 := witness.w_9_1
  let w_9_2 := witness.w_9_2
  let w_9_3 := witness.w_9_3
  let w_10_0 := witness.w_10_0
  let w_10_1 := witness.w_10_1
  let w_10_2 := witness.w_10_2
  let w_10_3 := witness.w_10_3
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_11_2 := witness.w_11_2
  let w_11_3 := witness.w_11_3
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_12_2 := witness.w_12_2
  let w_12_3 := witness.w_12_3
  let w_13_0 := witness.w_13_0
  let w_13_1 := witness.w_13_1
  let w_13_2 := witness.w_13_2
  let w_13_3 := witness.w_13_3
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_14_2 := witness.w_14_2
  let w_14_3 := witness.w_14_3
  let w_15_0 := witness.w_15_0
  let w_15_1 := witness.w_15_1
  let w_15_2 := witness.w_15_2
  let w_15_3 := witness.w_15_3
  let w_16_0 : mxxMat1 := (0 : mxxMat1)
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat2 := MxxRuntime.matrixMul w_20_0 w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat2 := MxxRuntime.matrixMul w_26_0 w_28_0
  let w_30_0 : mxxMat2 := MxxRuntime.matrixAdd w_23_0 w_29_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_36_0 := witness.w_36_0
  let w_36_1 := witness.w_36_1
  let w_36_2 := witness.w_36_2
  let w_36_3 := witness.w_36_3
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat2 := MxxRuntime.matrixMul w_33_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 := witness.w_42_0
  let w_43_0 : mxxMat2 := MxxRuntime.matrixMul w_20_0 w_42_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat2 := MxxRuntime.matrixMul w_26_0 w_45_0
  let w_47_0 : mxxMat2 := MxxRuntime.matrixAdd w_43_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 := witness.w_53_0
  let w_54_0 : mxxMat2 := MxxRuntime.matrixMul w_50_0 w_53_0
  let w_55_0 : mxxMat2 := MxxRuntime.matrixAdd w_40_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat2 := MxxRuntime.matrixMul w_33_0 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat2 := MxxRuntime.matrixMul w_50_0 w_64_0
  let w_66_0 : mxxMat2 := MxxRuntime.matrixAdd w_61_0 w_65_0
  let w_67_0 := witness.w_67_0
  let w_68_0 := witness.w_68_0
  let w_69_0 := witness.w_69_0
  let w_69_1 := witness.w_69_1
  let w_69_2 := witness.w_69_2
  let w_69_3 := witness.w_69_3
  let w_70_0 : mxxMat5 := (MxxRuntime.unitRow (0) : mxxMat5)
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
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
  let w_83_0 := witness.w_83_0
  let w_83_1 := witness.w_83_1
  let w_83_2 := witness.w_83_2
  let w_83_3 := witness.w_83_3
  let w_83_4 := witness.w_83_4
  let w_83_5 := witness.w_83_5
  let w_83_6 := witness.w_83_6
  let w_83_7 := witness.w_83_7
  let w_84_0 := witness.w_84_0
  let w_84_1 := witness.w_84_1
  let w_84_2 := witness.w_84_2
  let w_84_3 := witness.w_84_3
  let w_85_0 := witness.w_85_0
  let w_85_1 := witness.w_85_1
  let w_85_2 := witness.w_85_2
  let w_85_3 := witness.w_85_3
  let w_86_0 := witness.w_86_0
  let w_86_1 := witness.w_86_1
  let w_86_2 := witness.w_86_2
  let w_86_3 := witness.w_86_3
  let w_87_0 := witness.w_87_0
  let w_88_0 := witness.w_88_0
  let w_89_0 : mxxMat1 := MxxRuntime.matrixAdd w_87_0 w_88_0
  let w_90_0 := witness.w_90_0
  let w_91_0 := witness.w_91_0
  let w_92_0 : mxxMat1 := MxxRuntime.matrixAdd w_90_0 w_91_0
  let w_93_0 := witness.w_93_0
  let w_94_0 := witness.w_94_0
  let w_95_0 : mxxMat1 := MxxRuntime.matrixAdd w_93_0 w_94_0
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_98_0 : mxxMat1 := MxxRuntime.matrixAdd w_96_0 w_97_0
  let w_99_0 := witness.w_99_0
  let w_99_1 := witness.w_99_1
  let w_99_2 := witness.w_99_2
  let w_99_3 := witness.w_99_3
  let w_100_0 := witness.w_100_0
  let w_101_0 : mxxMat5 := (MxxRuntime.unitRow (0) : mxxMat5)
  let w_102_0 := witness.w_102_0
  let w_103_0 : mxxMat5 := (MxxRuntime.unitRow (0) : mxxMat5)
  let w_104_0 := witness.w_104_0
  let w_105_0 : mxxMat5 := (MxxRuntime.unitRow (0) : mxxMat5)
  let w_106_0 := witness.w_106_0
  let w_107_0 := witness.w_107_0
  let w_107_1 := witness.w_107_1
  let w_107_2 := witness.w_107_2
  let w_107_3 := witness.w_107_3
  let w_110_0 := witness.w_110_0
  let w_111_0 := witness.w_111_0
  let w_112_0 := witness.w_112_0
  let w_113_0 := witness.w_113_0
  let w_116_0 := witness.w_116_0
  let w_116_1 := witness.w_116_1
  let w_116_2 := witness.w_116_2
  let w_116_3 := witness.w_116_3
  let w_116_4 := witness.w_116_4
  let w_116_5 := witness.w_116_5
  let w_116_6 := witness.w_116_6
  let w_116_7 := witness.w_116_7
  let w_117_0 := witness.w_117_0
  let w_117_1 := witness.w_117_1
  let w_117_2 := witness.w_117_2
  let w_117_3 := witness.w_117_3
  let w_120_0 := witness.w_120_0
  let w_120_1 := witness.w_120_1
  let w_120_2 := witness.w_120_2
  let w_120_3 := witness.w_120_3
  let w_120_4 := witness.w_120_4
  let w_120_5 := witness.w_120_5
  let w_120_6 := witness.w_120_6
  let w_120_7 := witness.w_120_7
  let w_121_0 := witness.w_121_0
  let w_121_1 := witness.w_121_1
  let w_121_2 := witness.w_121_2
  let w_121_3 := witness.w_121_3
  let w_122_0 := witness.w_122_0
  let w_122_1 := witness.w_122_1
  let w_122_2 := witness.w_122_2
  let w_122_3 := witness.w_122_3
  let w_123_0 := witness.w_123_0
  let w_123_1 := witness.w_123_1
  let w_123_2 := witness.w_123_2
  let w_123_3 := witness.w_123_3
  let w_124_0 := witness.w_124_0
  let w_124_1 := witness.w_124_1
  let w_124_2 := witness.w_124_2
  let w_124_3 := witness.w_124_3
  let w_125_0 := witness.w_125_0
  let w_126_0 := witness.w_126_0
  let w_127_0 : mxxMat1 := MxxRuntime.matrixAdd w_125_0 w_126_0
  let w_128_0 := witness.w_128_0
  let w_129_0 := witness.w_129_0
  let w_130_0 : mxxMat1 := MxxRuntime.matrixAdd w_128_0 w_129_0
  let w_131_0 := witness.w_131_0
  let w_132_0 := witness.w_132_0
  let w_133_0 : mxxMat1 := MxxRuntime.matrixAdd w_131_0 w_132_0
  let w_134_0 := witness.w_134_0
  let w_135_0 := witness.w_135_0
  let w_136_0 : mxxMat1 := MxxRuntime.matrixAdd w_134_0 w_135_0
  let w_137_0 := witness.w_137_0
  let w_137_1 := witness.w_137_1
  let w_137_2 := witness.w_137_2
  let w_137_3 := witness.w_137_3
  let w_138_0 : mxxMat1 := (0 : mxxMat1)
  let w_139_0 := witness.w_139_0
  let w_139_1 := witness.w_139_1
  let w_139_2 := witness.w_139_2
  let w_139_3 := witness.w_139_3
  let w_140_0 : mxxMat1 := (0 : mxxMat1)
  let w_141_0 := witness.w_141_0
  let w_141_1 := witness.w_141_1
  let w_141_2 := witness.w_141_2
  let w_141_3 := witness.w_141_3
  let w_142_0 : mxxMat1 := (0 : mxxMat1)
  let w_143_0 := witness.w_143_0
  let w_143_1 := witness.w_143_1
  let w_143_2 := witness.w_143_2
  let w_143_3 := witness.w_143_3
  let w_144_0 : mxxMat1 := (0 : mxxMat1)
  let w_145_0 := witness.w_145_0
  let w_145_1 := witness.w_145_1
  let w_145_2 := witness.w_145_2
  let w_145_3 := witness.w_145_3
  let w_146_0 : mxxMat1 := (0 : mxxMat1)
  let w_147_0 := witness.w_147_0
  let w_147_1 := witness.w_147_1
  let w_147_2 := witness.w_147_2
  let w_147_3 := witness.w_147_3
  let w_148_0 : mxxMat1 := (0 : mxxMat1)
  let w_149_0 := witness.w_149_0
  let w_149_1 := witness.w_149_1
  let w_149_2 := witness.w_149_2
  let w_149_3 := witness.w_149_3
  let w_150_0 : mxxMat1 := (0 : mxxMat1)
  let w_151_0 := witness.w_151_0
  let w_151_1 := witness.w_151_1
  let w_151_2 := witness.w_151_2
  let w_151_3 := witness.w_151_3
  let w_152_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_155_0 := witness.w_155_0
  let w_156_0 := witness.w_156_0
  let w_157_0 := witness.w_157_0
  let w_158_0 := witness.w_158_0
  let w_161_0 := witness.w_161_0
  let w_161_1 := witness.w_161_1
  let w_161_2 := witness.w_161_2
  let w_161_3 := witness.w_161_3
  let w_161_4 := witness.w_161_4
  let w_161_5 := witness.w_161_5
  let w_161_6 := witness.w_161_6
  let w_161_7 := witness.w_161_7
  let w_162_0 := witness.w_162_0
  let w_162_1 := witness.w_162_1
  let w_162_2 := witness.w_162_2
  let w_162_3 := witness.w_162_3
  let w_165_0 := witness.w_165_0
  let w_165_1 := witness.w_165_1
  let w_165_2 := witness.w_165_2
  let w_165_3 := witness.w_165_3
  let w_165_4 := witness.w_165_4
  let w_165_5 := witness.w_165_5
  let w_165_6 := witness.w_165_6
  let w_165_7 := witness.w_165_7
  let w_166_0 := witness.w_166_0
  let w_166_1 := witness.w_166_1
  let w_166_2 := witness.w_166_2
  let w_166_3 := witness.w_166_3
  let w_167_0 := witness.w_167_0
  let w_167_1 := witness.w_167_1
  let w_167_2 := witness.w_167_2
  let w_167_3 := witness.w_167_3
  let w_168_0 := witness.w_168_0
  let w_168_1 := witness.w_168_1
  let w_168_2 := witness.w_168_2
  let w_168_3 := witness.w_168_3
  let w_169_0 := witness.w_169_0
  let w_170_0 := witness.w_170_0
  let w_171_0 : mxxMat1 := MxxRuntime.matrixAdd w_169_0 w_170_0
  let w_172_0 := witness.w_172_0
  let w_173_0 := witness.w_173_0
  let w_174_0 : mxxMat1 := MxxRuntime.matrixAdd w_172_0 w_173_0
  let w_175_0 := witness.w_175_0
  let w_176_0 := witness.w_176_0
  let w_177_0 : mxxMat1 := MxxRuntime.matrixAdd w_175_0 w_176_0
  let w_178_0 := witness.w_178_0
  let w_179_0 := witness.w_179_0
  let w_180_0 : mxxMat1 := MxxRuntime.matrixAdd w_178_0 w_179_0
  let w_181_0 := witness.w_181_0
  let w_181_1 := witness.w_181_1
  let w_181_2 := witness.w_181_2
  let w_181_3 := witness.w_181_3
  let w_182_0 := witness.w_182_0
  let w_183_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_184_0 := witness.w_184_0
  let w_185_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_186_0 := witness.w_186_0
  let w_187_0 : mxxMat7 := (MxxRuntime.unitRow (0) : mxxMat7)
  let w_188_0 := witness.w_188_0
  let w_189_0 := witness.w_189_0
  let w_189_1 := witness.w_189_1
  let w_189_2 := witness.w_189_2
  let w_189_3 := witness.w_189_3
  let w_190_0 := witness.w_190_0
  let w_191_0 := witness.w_191_0
  let w_192_0 := witness.w_192_0
  let w_193_0 := witness.w_193_0
  let w_194_0 := witness.w_194_0
  let w_195_0 : mxxMat2 := MxxRuntime.matrixMul w_192_0 w_194_0
  let w_196_0 := witness.w_196_0
  let w_197_0 := witness.w_197_0
  let w_198_0 := witness.w_198_0
  let w_199_0 := witness.w_199_0
  let w_200_0 := witness.w_200_0
  let w_201_0 : mxxMat2 := MxxRuntime.matrixMul w_198_0 w_200_0
  let w_202_0 : mxxMat2 := MxxRuntime.matrixAdd w_195_0 w_201_0
  let w_203_0 := witness.w_203_0
  let w_204_0 := witness.w_204_0
  let w_205_0 := witness.w_205_0
  let w_206_0 : mxxMat2 := MxxRuntime.matrixMul w_192_0 w_205_0
  let w_207_0 := witness.w_207_0
  let w_208_0 := witness.w_208_0
  let w_209_0 : mxxMat2 := MxxRuntime.matrixMul w_198_0 w_208_0
  let w_210_0 : mxxMat2 := MxxRuntime.matrixAdd w_206_0 w_209_0
  let w_211_0 := witness.w_211_0
  let w_212_0 := witness.w_212_0
  let w_213_0 := witness.w_213_0
  let w_214_0 := witness.w_214_0
  let w_215_0 := witness.w_215_0
  let w_216_0 := witness.w_216_0
  let w_217_0 := witness.w_217_0
  let w_218_0 := witness.w_218_0
  let w_219_0 : mxxMat2 := MxxRuntime.matrixMul w_216_0 w_218_0
  let w_220_0 := witness.w_220_0
  let w_221_0 := witness.w_221_0
  let w_222_0 := witness.w_222_0
  let w_223_0 := witness.w_223_0
  let w_224_0 := witness.w_224_0
  let w_225_0 : mxxMat2 := MxxRuntime.matrixMul w_222_0 w_224_0
  let w_226_0 : mxxMat2 := MxxRuntime.matrixAdd w_219_0 w_225_0
  let w_227_0 := witness.w_227_0
  let w_228_0 := witness.w_228_0
  let w_229_0 := witness.w_229_0
  let w_230_0 := witness.w_230_0
  let w_231_0 := witness.w_231_0
  let w_232_0 := witness.w_232_0
  let w_233_0 := witness.w_233_0
  let w_234_0 := witness.w_234_0
  let w_235_0 : mxxMat2 := MxxRuntime.matrixMul w_232_0 w_234_0
  let w_236_0 := witness.w_236_0
  let w_237_0 := witness.w_237_0
  let w_238_0 := witness.w_238_0
  let w_239_0 := witness.w_239_0
  let w_240_0 := witness.w_240_0
  let w_241_0 : mxxMat2 := MxxRuntime.matrixMul w_238_0 w_240_0
  let w_242_0 : mxxMat2 := MxxRuntime.matrixAdd w_235_0 w_241_0
  let w_243_0 := witness.w_243_0
  let w_244_0 := witness.w_244_0
  let w_245_0 : mxxMat2 := MxxRuntime.matrixMul w_229_0 w_244_0
  let w_246_0 := witness.w_246_0
  let w_247_0 := witness.w_247_0
  let w_248_0 : mxxMat2 := MxxRuntime.matrixMul w_216_0 w_247_0
  let w_249_0 := witness.w_249_0
  let w_250_0 := witness.w_250_0
  let w_251_0 : mxxMat2 := MxxRuntime.matrixMul w_222_0 w_250_0
  let w_252_0 : mxxMat2 := MxxRuntime.matrixAdd w_248_0 w_251_0
  let w_253_0 := witness.w_253_0
  let w_254_0 := witness.w_254_0
  let w_255_0 := witness.w_255_0
  let w_256_0 := witness.w_256_0
  let w_257_0 := witness.w_257_0
  let w_258_0 : mxxMat2 := MxxRuntime.matrixMul w_255_0 w_257_0
  let w_259_0 : mxxMat2 := MxxRuntime.matrixAdd w_245_0 w_258_0
  let w_260_0 := witness.w_260_0
  let w_261_0 := witness.w_261_0
  let w_262_0 := witness.w_262_0
  let w_263_0 : mxxMat2 := MxxRuntime.matrixMul w_232_0 w_262_0
  let w_264_0 := witness.w_264_0
  let w_265_0 := witness.w_265_0
  let w_266_0 : mxxMat2 := MxxRuntime.matrixMul w_238_0 w_265_0
  let w_267_0 : mxxMat2 := MxxRuntime.matrixAdd w_263_0 w_266_0
  let w_268_0 := witness.w_268_0
  let w_269_0 := witness.w_269_0
  let w_270_0 : mxxMat2 := MxxRuntime.matrixMul w_229_0 w_269_0
  let w_271_0 := witness.w_271_0
  let w_272_0 := witness.w_272_0
  let w_273_0 : mxxMat2 := MxxRuntime.matrixMul w_255_0 w_272_0
  let w_274_0 : mxxMat2 := MxxRuntime.matrixAdd w_270_0 w_273_0
  let w_275_0 := witness.w_275_0
  let w_276_0 := witness.w_276_0
  let w_277_0 := witness.w_277_0
  let w_278_0 := witness.w_278_0
  let w_279_0 := witness.w_279_0
  let w_280_0 := witness.w_280_0
  let w_281_0 := witness.w_281_0
  let w_282_0 := witness.w_282_0
  let w_283_0 : mxxMat2 := MxxRuntime.matrixMul w_280_0 w_282_0
  let w_284_0 := witness.w_284_0
  let w_285_0 := witness.w_285_0
  let w_286_0 := witness.w_286_0
  let w_287_0 := witness.w_287_0
  let w_288_0 := witness.w_288_0
  let w_289_0 : mxxMat2 := MxxRuntime.matrixMul w_286_0 w_288_0
  let w_290_0 : mxxMat2 := MxxRuntime.matrixAdd w_283_0 w_289_0
  let w_291_0 := witness.w_291_0
  let w_292_0 := witness.w_292_0
  let w_293_0 := witness.w_293_0
  let w_294_0 := witness.w_294_0
  let w_295_0 := witness.w_295_0
  let w_296_0 : mxxMat2 := MxxRuntime.matrixMul w_293_0 w_295_0
  let w_297_0 := witness.w_297_0
  let w_298_0 := witness.w_298_0
  let w_299_0 : mxxMat2 := MxxRuntime.matrixMul w_280_0 w_298_0
  let w_300_0 := witness.w_300_0
  let w_301_0 := witness.w_301_0
  let w_302_0 : mxxMat2 := MxxRuntime.matrixMul w_286_0 w_301_0
  let w_303_0 : mxxMat2 := MxxRuntime.matrixAdd w_299_0 w_302_0
  let w_304_0 := witness.w_304_0
  let w_305_0 := witness.w_305_0
  let w_306_0 := witness.w_306_0
  let w_307_0 := witness.w_307_0
  let w_308_0 := witness.w_308_0
  let w_309_0 : mxxMat2 := MxxRuntime.matrixMul w_306_0 w_308_0
  let w_310_0 : mxxMat2 := MxxRuntime.matrixAdd w_296_0 w_309_0
  let w_311_0 := witness.w_311_0
  let w_312_0 := witness.w_312_0
  let w_313_0 := witness.w_313_0
  let w_314_0 : mxxMat2 := MxxRuntime.matrixMul w_293_0 w_313_0
  let w_315_0 := witness.w_315_0
  let w_316_0 := witness.w_316_0
  let w_317_0 : mxxMat2 := MxxRuntime.matrixMul w_306_0 w_316_0
  let w_318_0 : mxxMat2 := MxxRuntime.matrixAdd w_314_0 w_317_0
  let w_319_0 := witness.w_319_0
  let w_320_0 := witness.w_320_0
  let w_321_0 := witness.w_321_0
  let w_322_0 := witness.w_322_0
  let w_323_0 := witness.w_323_0
  let w_324_0 := witness.w_324_0
  let w_325_0 := witness.w_325_0
  let w_326_0 := witness.w_326_0
  let w_327_0 : mxxMat2 := MxxRuntime.matrixMul w_324_0 w_326_0
  let w_328_0 := witness.w_328_0
  let w_329_0 := witness.w_329_0
  let w_330_0 := witness.w_330_0
  let w_331_0 := witness.w_331_0
  let w_332_0 := witness.w_332_0
  let w_333_0 : mxxMat2 := MxxRuntime.matrixMul w_330_0 w_332_0
  let w_334_0 : mxxMat2 := MxxRuntime.matrixAdd w_327_0 w_333_0
  let w_335_0 := witness.w_335_0
  let w_336_0 := witness.w_336_0
  let w_337_0 := witness.w_337_0
  let w_338_0 : mxxMat2 := MxxRuntime.matrixMul w_324_0 w_337_0
  let w_339_0 := witness.w_339_0
  let w_340_0 := witness.w_340_0
  let w_341_0 : mxxMat2 := MxxRuntime.matrixMul w_330_0 w_340_0
  let w_342_0 : mxxMat2 := MxxRuntime.matrixAdd w_338_0 w_341_0
  let w_343_0 := witness.w_343_0
  let w_344_0 := witness.w_344_0
  let w_345_0 := witness.w_345_0
  let w_346_0 := witness.w_346_0
  let w_347_0 := witness.w_347_0
  let w_348_0 := witness.w_348_0
  let w_349_0 := witness.w_349_0
  let w_350_0 := witness.w_350_0
  let w_351_0 : mxxMat2 := MxxRuntime.matrixMul w_348_0 w_350_0
  let w_352_0 := witness.w_352_0
  let w_353_0 := witness.w_353_0
  let w_354_0 := witness.w_354_0
  let w_355_0 := witness.w_355_0
  let w_356_0 := witness.w_356_0
  let w_357_0 : mxxMat2 := MxxRuntime.matrixMul w_354_0 w_356_0
  let w_358_0 : mxxMat2 := MxxRuntime.matrixAdd w_351_0 w_357_0
  let w_359_0 := witness.w_359_0
  let w_360_0 := witness.w_360_0
  let w_361_0 := witness.w_361_0
  let w_362_0 : mxxMat2 := MxxRuntime.matrixMul w_348_0 w_361_0
  let w_363_0 := witness.w_363_0
  let w_364_0 := witness.w_364_0
  let w_365_0 : mxxMat2 := MxxRuntime.matrixMul w_354_0 w_364_0
  let w_366_0 : mxxMat2 := MxxRuntime.matrixAdd w_362_0 w_365_0
  let w_367_0 := witness.w_367_0
  let w_368_0 := witness.w_368_0
  let w_369_0 := witness.w_369_0
  generatedRoot.constraints_0 backend hashModel params w_0_0 w_1_0 w_17_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_6_1 w_6_2 w_6_3 w_7_0 w_7_1 w_7_2 w_7_3 w_8_0 w_8_1 w_8_2 w_8_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_15_0 w_15_1 w_15_2 w_15_3 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 ∧
    generatedRoot.constraints_1 backend hashModel params w_0_0 w_1_0 w_17_0 w_34_0 w_35_0 w_2_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_36_0 w_36_1 w_36_2 w_36_3 w_37_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 ∧
    generatedRoot.constraints_2 backend w_36_1 w_36_2 w_36_3 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_62_0 ∧
    generatedRoot.constraints_3 backend hashModel params w_0_0 w_1_0 w_17_0 w_71_0 w_72_0 w_77_0 w_78_0 w_81_0 w_82_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_16_0 w_56_0 w_57_0 w_62_0 w_63_0 w_64_0 w_66_0 w_67_0 w_68_0 w_69_0 w_69_1 w_69_2 w_69_3 w_73_0 w_74_0 w_75_0 w_76_0 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_79_6 w_79_7 w_80_0 w_80_1 w_80_2 w_80_3 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_83_6 w_83_7 w_84_0 w_84_1 w_84_2 w_84_3 ∧
    generatedRoot.constraints_4 backend hashModel params w_0_0 w_1_0 w_17_0 w_109_0 w_114_0 w_115_0 w_2_0 w_4_0 w_4_1 w_4_2 w_4_3 w_70_0 w_84_0 w_84_1 w_84_2 w_84_3 w_85_0 w_85_1 w_85_2 w_85_3 w_86_0 w_86_1 w_86_2 w_86_3 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_99_1 w_99_2 w_99_3 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_107_1 w_107_2 w_107_3 w_110_0 w_111_0 w_112_0 w_113_0 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_116_6 w_116_7 ∧
    generatedRoot.constraints_5 backend hashModel params w_0_0 w_1_0 w_17_0 w_108_0 w_118_0 w_119_0 w_2_0 w_6_0 w_6_1 w_6_2 w_6_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_116_0 w_116_1 w_116_2 w_116_3 w_117_0 w_117_1 w_117_2 w_117_3 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_121_0 w_121_1 w_121_2 w_121_3 w_122_0 w_122_1 w_122_2 w_122_3 w_123_0 w_123_1 w_123_2 w_123_3 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_137_1 w_137_2 w_137_3 w_138_0 w_139_0 w_139_1 w_139_2 w_139_3 w_140_0 w_141_0 w_141_1 w_141_2 w_141_3 w_142_0 w_143_0 w_143_1 w_143_2 w_143_3 w_144_0 w_145_0 w_145_1 w_145_2 w_145_3 w_146_0 w_147_0 w_147_1 w_147_2 w_147_3 ∧
    generatedRoot.constraints_6 backend hashModel params w_0_0 w_1_0 w_17_0 w_153_0 w_154_0 w_159_0 w_160_0 w_163_0 w_164_0 w_2_0 w_5_0 w_5_1 w_5_2 w_5_3 w_7_0 w_7_1 w_7_2 w_7_3 w_15_0 w_15_1 w_15_2 w_15_3 w_137_0 w_137_1 w_137_2 w_137_3 w_148_0 w_149_0 w_149_1 w_149_2 w_149_3 w_150_0 w_151_0 w_151_1 w_151_2 w_151_3 w_155_0 w_156_0 w_157_0 w_158_0 w_161_0 w_161_1 w_161_2 w_161_3 w_161_4 w_161_5 w_161_6 w_161_7 w_162_0 w_162_1 w_162_2 w_162_3 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_165_6 w_165_7 w_166_0 w_166_1 w_166_2 w_166_3 w_167_0 w_167_1 w_167_2 w_167_3 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_181_1 w_181_2 w_181_3 ∧
    generatedRoot.constraints_7 backend hashModel params w_17_0 w_152_0 w_168_0 w_168_1 w_168_2 w_168_3 w_181_0 w_181_1 w_181_2 w_181_3 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_189_1 w_189_2 w_189_3 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_202_0 w_203_0 w_204_0 w_205_0 w_207_0 ∧
    generatedRoot.constraints_8 backend hashModel w_17_0 w_202_0 w_207_0 w_208_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 ∧
    generatedRoot.constraints_9 backend hashModel w_17_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_242_0 w_243_0 ∧
    generatedRoot.constraints_10 backend hashModel w_17_0 w_242_0 w_243_0 w_244_0 w_246_0 w_247_0 w_249_0 w_250_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 ∧
    generatedRoot.constraints_11 backend hashModel w_17_0 w_259_0 w_260_0 w_261_0 w_262_0 w_264_0 w_265_0 w_267_0 w_268_0 w_269_0 w_271_0 w_272_0 w_274_0 w_275_0 ∧
    generatedRoot.constraints_12 backend hashModel w_17_0 w_259_0 w_274_0 w_276_0 w_277_0 w_278_0 w_279_0 w_280_0 w_281_0 w_282_0 w_284_0 w_285_0 w_286_0 w_287_0 w_288_0 w_290_0 w_291_0 ∧
    generatedRoot.constraints_13 backend hashModel w_17_0 w_290_0 w_291_0 w_292_0 w_293_0 w_294_0 w_295_0 w_297_0 w_298_0 w_300_0 w_301_0 w_303_0 w_304_0 w_305_0 w_306_0 w_307_0 w_308_0 ∧
    generatedRoot.constraints_14 backend hashModel w_17_0 w_310_0 w_311_0 w_312_0 w_313_0 w_315_0 w_316_0 w_318_0 w_319_0 w_320_0 w_321_0 w_322_0 w_323_0 w_324_0 w_325_0 ∧
    generatedRoot.constraints_15 backend hashModel w_17_0 w_325_0 w_326_0 w_328_0 w_329_0 w_330_0 w_331_0 w_332_0 w_334_0 w_335_0 w_336_0 w_337_0 w_339_0 w_340_0 w_342_0 w_343_0 w_344_0 ∧
    generatedRoot.constraints_16 backend hashModel w_17_0 w_342_0 w_345_0 w_346_0 w_347_0 w_348_0 w_349_0 w_350_0 w_352_0 w_353_0 w_354_0 w_355_0 w_356_0 w_358_0 w_359_0 w_360_0 w_361_0 w_363_0 w_364_0 ∧
    generatedRoot.constraints_17 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_6_1 w_6_2 w_6_3 w_7_0 w_7_1 w_7_2 w_7_3 w_8_0 w_8_1 w_8_2 w_8_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_15_0 w_15_1 w_15_2 w_15_3 w_36_2 w_36_3 w_69_0 w_69_1 w_69_2 w_69_3 w_79_6 w_79_7 w_83_6 w_83_7 w_107_0 w_107_1 w_107_2 w_107_3 w_116_6 w_116_7 w_120_6 w_120_7 w_137_0 w_137_1 w_137_2 w_137_3 w_139_0 w_139_1 w_139_2 w_139_3 w_141_0 w_141_1 w_141_2 w_141_3 w_143_0 w_143_1 w_143_2 w_143_3 w_145_0 w_145_1 w_145_2 w_145_3 w_147_0 w_147_1 w_147_2 w_147_3 w_149_0 w_149_1 w_149_2 w_149_3 w_151_0 w_151_1 w_151_2 w_151_3 w_161_6 w_161_7 w_165_6 w_165_7 w_189_0 w_189_1 w_189_2 w_189_3 w_203_0 w_211_0 w_212_0 w_213_0 w_260_0 w_275_0 w_276_0 w_277_0 w_311_0 w_319_0 w_320_0 w_321_0 w_335_0 w_343_0 w_344_0 w_345_0 w_358_0 w_359_0 w_366_0 w_367_0 w_368_0 w_369_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × mxxMat0 × ByteArray × (Fin 1 → Int) × (Fin 1 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 72000 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 16 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × (Fin 14400 → Int) × Unit) (outputs : (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat4 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat6 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 72000 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 16 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × (Fin 14400 → mxxMat1) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end Stage_preprocess_0_0
