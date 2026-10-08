import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Ideal

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where

abbrev generatedRoot.constraints_0 (w_0_0 : Fin 8192 → Int) (outputs : Fin 8192 → Int) : Prop :=
  outputs = w_0_0

abbrev generatedRoot.body (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (inputs : Fin 8192 → Int) (outputs : Fin 8192 → Int) (_witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_0_0 : Fin 8192 → Int := inputs
  generatedRoot.constraints_0 w_0_0 outputs

def generatedRoot (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Fin 8192 → Int) (outputs : Fin 8192 → Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body tape path params inputs outputs witness


end Ideal
