import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_encrypt_left

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : Fin 630 → Int
  w_4_0 : Int
  w_13_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1
  w_14_0 : Int
  w_22_0 : Int

abbrev generatedRoot.constraints_0 (hashModel : MxxRuntime.HashModel) (w_0_0 : ByteArray) (w_1_0 : Fin 630 → Int) (w_3_0 : Fin 1 → Int) (w_4_0 : Int) (w_13_0 : Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 1) (w_14_0 : Int) (w_19_0 : Int) (w_21_0 : Int) (w_22_0 : Int) (w_24_0 : Int) (w_25_0 : Int) (outputs : (Fin 630 → Int) × Int × Unit) : Prop :=
  MxxRuntime.hashIntFamily (hashModel) (4294967296) ([109, 120, 120, 47, 104, 97, 115, 104, 45, 105, 110, 116, 45, 102, 97, 109, 105, 108, 121, 47, 118, 49, 0, 116, 102, 104, 101, 47, 108, 119, 101, 45, 101, 110, 99, 114, 121, 112, 116, 47, 97, 47, 118, 49]) ([]) (w_0_0) w_1_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_4_0 ∧
  MxxRuntime.gaussianSample ((128/1)) (2048) w_13_0 ∧
  MxxRuntime.extractCoefficient 0 w_13_0 w_14_0 ∧
  0 ≤ w_19_0 ∧
  w_19_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_19_0 [w_21_0, w_14_0] w_22_0 ∧
  w_24_0 ≠ 0 ∧
  outputs = (w_1_0, (w_25_0, ()))

abbrev generatedRoot.body (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × (Fin 630 → Int) × Int × Unit) (outputs : (Fin 630 → Int) × Int × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : ByteArray := inputs.1
  let w_2_0 : Fin 630 → Int := inputs.2.1
  let w_5_0 : Int := inputs.2.2.1
  let w_1_0 := witness.w_1_0
  let w_3_0 : Fin 1 → Int := MxxRuntime.intMatrixVectorProduct false w_1_0 w_2_0
  let w_4_0 := witness.w_4_0
  let w_6_0 : Int := 2
  let w_7_0 : Int := (w_5_0 * w_6_0)
  let w_8_0 : Int := 1
  let w_9_0 : Int := (w_7_0 - w_8_0)
  let w_10_0 : Int := 536870912
  let w_11_0 : Int := (w_9_0 * w_10_0)
  let w_12_0 : Int := (w_4_0 + w_11_0)
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : Int := 2
  let w_16_0 : Int := (w_14_0 * w_15_0)
  let w_17_0 : Int := 4611255024196841473
  let w_18_0 : Bool := decide (w_16_0 ≤ w_17_0)
  let w_19_0 : Int := if w_18_0 then 1 else 0
  let w_20_0 : Int := 4611255024196841473
  let w_21_0 : Int := (w_14_0 - w_20_0)
  let w_22_0 := witness.w_22_0
  let w_23_0 : Int := (w_12_0 + w_22_0)
  let w_24_0 : Int := 4294967296
  let w_25_0 : Int := (w_23_0 % w_24_0)
  generatedRoot.constraints_0 hashModel w_0_0 w_1_0 w_3_0 w_4_0 w_13_0 w_14_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 outputs

def generatedRoot (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : ByteArray × (Fin 630 → Int) × Int × Unit) (outputs : (Fin 630 → Int) × Int × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body hashModel params inputs outputs witness


end Stage_encrypt_left
