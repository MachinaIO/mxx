import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_decode

structure Params where
  unit : Unit

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := 1152921504606584833

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 1

abbrev sequential_generatedRoot_3.constraints_0 (w_2_0 : Fin 65536 → Int) (w_1_0 : Int) (w_3_0 : Int) (w_6_0 : Int) (w_14_0 : Int) (w_23_0 : Int) (w_26_0 : Int) (outputs : Int) : Prop :=
  0 ≤ w_1_0 ∧
  w_1_0 < 65536 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_1_0 w_3_0 ∧
  w_6_0 ≠ 0 ∧
  w_14_0 ≠ 0 ∧
  w_23_0 ≠ 0 ∧
  outputs = w_26_0

def sequential_generatedRoot_3 (params : Params) (i_0 : Nat) (inputs : Int × (Fin 65536 → Int) × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_2_0 : Fin 65536 → Int := inputs.2.1
  let w_1_0 : Int := (Int.ofNat i_0)
  ∃ (w_3_0 : Int),
    let w_4_0 : Int := 2
    let w_5_0 : Int := (w_3_0 * w_4_0)
    let w_6_0 : Int := 1152921504606584833
    let w_7_0 : Int := (w_5_0 / w_6_0)
    let w_8_0 : Int := 1152921504606584833
    let w_9_0 : Int := (w_7_0 * w_8_0)
    let w_10_0 : Int := (w_3_0 - w_9_0)
    let w_11_0 : Int := 1
    let w_12_0 : Int := (w_10_0 + w_11_0)
    let w_13_0 : Int := (w_10_0 * w_12_0)
    let w_14_0 : Int := 2
    let w_15_0 : Int := (w_13_0 / w_14_0)
    let w_16_0 : Int := (w_1_0 * w_15_0)
    let w_17_0 : Int := (w_0_0 + w_16_0)
    let w_18_0 : Int := 65536
    let w_19_0 : Int := (w_1_0 + w_18_0)
    let w_20_0 : Int := 1
    let w_21_0 : Int := (w_10_0 - w_20_0)
    let w_22_0 : Int := (w_10_0 * w_21_0)
    let w_23_0 : Int := 2
    let w_24_0 : Int := (w_22_0 / w_23_0)
    let w_25_0 : Int := (w_19_0 * w_24_0)
    let w_26_0 : Int := (w_17_0 + w_25_0)
    sequential_generatedRoot_3.constraints_0 w_2_0 w_1_0 w_3_0 w_6_0 w_14_0 w_23_0 w_26_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Fin 65536 → Int
  w_3_0 : Int

abbrev generatedRoot.constraints_0 (params : Params) (w_1_0 : mxxMat0) (w_0_0 : Int) (w_2_0 : Fin 65536 → Int) (w_3_0 : Int) (w_4_0 : Int) (w_5_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.polynomialValues false w_1_0 w_2_0 ∧
  0 ≤ (65536) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Int) => sequential_generatedRoot_3 params i (current, (w_2_0, ())) next) (Int.toNat (65536)) w_0_0 w_3_0 ∧
  w_4_0 ≠ 0 ∧
  outputs = w_5_0

abbrev generatedRoot.body (params : Params) (inputs : mxxMat0) (outputs : Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : mxxMat0 := inputs
  let w_0_0 : Int := 0
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 : Int := 2
  let w_5_0 : Int := (w_3_0 / w_4_0)
  generatedRoot.constraints_0 params w_1_0 w_0_0 w_2_0 w_3_0 w_4_0 w_5_0 outputs

def generatedRoot (params : Params) (inputs : mxxMat0) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Stage_decode
