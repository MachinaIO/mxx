import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_decrypt

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_5_0 : Int
  w_16_0 : Int

abbrev generatedRoot.constraints_0 (w_4_0 : Fin 1 → Int) (w_5_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (w_13_0 : Int) (w_15_0 : Int) (w_16_0 : Int) (w_20_0 : Int) (outputs : Int × Int × Unit) : Prop :=
  MxxRuntime.familyGetStatic w_4_0 (0) w_5_0 ∧
  w_7_0 ≠ 0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_13_0 [w_15_0, w_8_0] w_16_0 ∧
  outputs = (w_20_0, (w_8_0, ()))

abbrev generatedRoot.body (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (inputs : Int × (Fin 630 → Int) × (Fin 630 → Int) × Unit) (outputs : Int × Int × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : Int := inputs.1
  let w_2_0 : Fin 630 → Int := inputs.2.1
  let w_3_0 : Fin 630 → Int := inputs.2.2.1
  let w_0_0 : Int := 1
  let w_4_0 : Fin 1 → Int := MxxRuntime.intMatrixVectorProduct false w_2_0 w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 : Int := (w_1_0 - w_5_0)
  let w_7_0 : Int := 4294967296
  let w_8_0 : Int := (w_6_0 % w_7_0)
  let w_9_0 : Int := 2
  let w_10_0 : Int := (w_8_0 * w_9_0)
  let w_11_0 : Int := 4294967296
  let w_12_0 : Bool := decide (w_10_0 ≤ w_11_0)
  let w_13_0 : Int := if w_12_0 then 1 else 0
  let w_14_0 : Int := 4294967296
  let w_15_0 : Int := (w_8_0 - w_14_0)
  let w_16_0 := witness.w_16_0
  let w_17_0 : Int := 0
  let w_18_0 : Bool := decide (w_16_0 < w_17_0)
  let w_19_0 : Int := if w_18_0 then 1 else 0
  let w_20_0 : Int := (w_0_0 - w_19_0)
  generatedRoot.constraints_0 w_4_0 w_5_0 w_7_0 w_8_0 w_13_0 w_15_0 w_16_0 w_20_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Int × (Fin 630 → Int) × (Fin 630 → Int) × Unit) (outputs : Int × Int × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Stage_decrypt
