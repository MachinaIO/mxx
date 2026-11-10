import MxxIR
import MxxRuntime
import MxxSecurity.Sampling

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Plain

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_4.constraints_0 (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  outputs = w_4_0

def parallel_generatedRoot_4 (params : Params) (_ : Nat) (inputs : Int × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Int := inputs.2.1
  let w_3_0 : Int := inputs.2.2.1
  let w_2_0 : Int := (w_0_0 * w_1_0)
  let w_4_0 : Int := (w_2_0 % w_3_0)
  parallel_generatedRoot_4.constraints_0 w_3_0 w_4_0 outputs

abbrev parallel_generatedRoot_6.constraints_0 (w_3_0 : Int) (w_4_0 : Int) (outputs : Int) : Prop :=
  w_3_0 ≠ 0 ∧
  outputs = w_4_0

def parallel_generatedRoot_6 (params : Params) (_ : Nat) (inputs : Int × Int × Int × Unit) (outputs : Int) : Prop :=
  let _params := params
  let w_0_0 : Int := inputs.1
  let w_1_0 : Int := inputs.2.1
  let w_3_0 : Int := inputs.2.2.1
  let w_2_0 : Int := (w_0_0 * w_1_0)
  let w_4_0 : Int := (w_2_0 % w_3_0)
  parallel_generatedRoot_6.constraints_0 w_3_0 w_4_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_4_0 : Fin 8192 → Int
  w_6_0 : Fin 8192 → Int

abbrev generatedRoot.constraints_0 (params : Params) (w_0_0 : Fin 8192 → Int) (w_1_0 : Fin 8192 → Int) (w_2_0 : Fin 8192 → Int) (w_3_0 : Int) (w_4_0 : Fin 8192 → Int) (w_5_0 : Int) (w_6_0 : Fin 8192 → Int) (outputs : Fin 8192 → Int) : Prop :=
  (8192) = 8192 ∧
  (∀ i : Fin 8192, parallel_generatedRoot_4 params i ((w_1_0 i), ((w_2_0 i), (w_3_0, ()))) (w_4_0 i)) ∧
  (8192) = 8192 ∧
  (∀ i : Fin 8192, parallel_generatedRoot_6 params i ((w_0_0 i), ((w_4_0 i), (w_5_0, ()))) (w_6_0 i)) ∧
  outputs = w_6_0

abbrev generatedRoot.body (params : Params) (inputs : (Fin 8192 → Int) × (Fin 8192 → Int) × (Fin 8192 → Int) × Unit) (outputs : Fin 8192 → Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Fin 8192 → Int := inputs.1
  let w_1_0 : Fin 8192 → Int := inputs.2.1
  let w_2_0 : Fin 8192 → Int := inputs.2.2.1
  let w_3_0 : Int := 1032193
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := 1032193
  let w_6_0 := witness.w_6_0
  generatedRoot.constraints_0 params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 outputs

def generatedRoot (params : Params) (inputs : (Fin 8192 → Int) × (Fin 8192 → Int) × (Fin 8192 → Int) × Unit) (outputs : Fin 8192 → Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness

noncomputable def generatedRoot.comp (params : Params) (inputs : (Fin 8192 → Int) × (Fin 8192 → Int) × (Fin 8192 → Int) × Unit) : MxxSecurity.Comp (Fin 8192 → Int) := do
  pure (MxxSecurity.determined fun outputs : Fin 8192 → Int ↦
    ∃ witness : generatedRoot.Witness, generatedRoot.body params inputs outputs witness)


end Plain
