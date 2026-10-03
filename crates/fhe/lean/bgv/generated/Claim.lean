import Backend
import Stage_keygen
import Stage_encrypt_x
import Stage_encrypt_y
import Stage_multiply
import Stage_relinearize
import Stage_modswitch
import Stage_decrypt
import Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace GeneratedClaim

structure ExternalInputs where
  input_0 : Fin 8192 → Int
  input_1 : Fin 8192 → Int

def ValidExternals (external : ExternalInputs) : Prop :=
  (∀ contract_i_0 : Fin 8192, ((0 : Int) ≤ (external.input_0 contract_i_0) ∧ (external.input_0 contract_i_0) ≤ (1032192 : Int))) ∧
  (∀ contract_i_0 : Fin 8192, ((0 : Int) ≤ (external.input_1 contract_i_0) ∧ (external.input_1 contract_i_0) ≤ (1032192 : Int)))

structure Execution where
  «stage_0» : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1 × Mxx.Primitives.ExactMatrix 421249166578543632464236954685046510773404157767471124412817604609 8192 2 3 × Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 1 1 × Unit
  «stage_1» : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
  «stage_2» : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
  «stage_3» : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 3 1
  «stage_4» : Mxx.Primitives.ExactMatrix 5846006548020969210596774788483421161649837621249 8192 2 1
  «stage_5» : Mxx.Primitives.ExactMatrix 324518553605595287786984016396289 8192 2 1
  «stage_6» : Fin 8192 → Int
  «ideal» : Fin 8192 → Int

def stage_0_params : Stage_keygen.Params := { «unit» := () }

def stage_1_params : Stage_encrypt_x.Params := { «unit» := () }

def stage_2_params : Stage_encrypt_y.Params := { «unit» := () }

def stage_3_params : Stage_multiply.Params := { «unit» := () }

def stage_4_params : Stage_relinearize.Params := { «unit» := () }

def stage_5_params : Stage_modswitch.Params := { «unit» := () }

def stage_6_params : Stage_decrypt.Params := { «unit» := () }

def ideal_params : Ideal.Params := { «unit» := () }

def Runs (_ : MxxRuntime.HashModel) (external : ExternalInputs)
    (execution : Execution) : Prop :=
  ValidExternals external ∧
  Stage_keygen.generatedRoot stage_0_params (()) execution.«stage_0» ∧
  Stage_encrypt_x.generatedRoot stage_1_params ((execution.«stage_0».1, external.input_0, ())) execution.«stage_1» ∧
  Stage_encrypt_y.generatedRoot stage_2_params ((execution.«stage_0».1, external.input_1, ())) execution.«stage_2» ∧
  Stage_multiply.generatedRoot stage_3_params ((execution.«stage_1», execution.«stage_2», ())) execution.«stage_3» ∧
  Stage_relinearize.generatedRoot stage_4_params ((execution.«stage_3», execution.«stage_0».2.1, ())) execution.«stage_4» ∧
  Stage_modswitch.generatedRoot stage_5_params (execution.«stage_4») execution.«stage_5» ∧
  Stage_decrypt.generatedRoot stage_6_params ((execution.«stage_5», execution.«stage_0».2.2.1, ())) execution.«stage_6» ∧
  Ideal.generatedRoot ideal_params ((external.input_0, external.input_1, ())) execution.«ideal»

/-- Every execution's endpoint equals the ideal one. -/
def CorrectnessClaim : Prop :=
  ∀ hashModel external execution, Runs hashModel external execution →
    execution.«stage_6» = execution.«ideal»

end GeneratedClaim
