import Backend
import Stage_mask
import Stage_encrypt
import Stage_online_eval
import Stage_decode
import Requirement_0
import Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace GeneratedClaim

structure ExternalInputs where
  input_0 : ByteArray
  input_1 : Fin 5 → Int
  input_2 : Fin 5 → Int
  input_3 : Fin 5 → Int
  input_4 : Int
  input_5 : Int

def ValidExternals (external : ExternalInputs) : Prop :=
  (external.input_0).size = 32 ∧
  (∀ contract_i_0 : Fin 5, ((0 : Int) ≤ (external.input_1 contract_i_0) ∧ (external.input_1 contract_i_0) ≤ (31 : Int))) ∧
  (∀ contract_i_0 : Fin 5, ((-1 : Int) ≤ (external.input_2 contract_i_0) ∧ (external.input_2 contract_i_0) ≤ (1 : Int))) ∧
  (∀ contract_i_0 : Fin 5, ((0 : Int) ≤ (external.input_3 contract_i_0) ∧ (external.input_3 contract_i_0) ≤ (31 : Int))) ∧
  ((-1 : Int) ≤ external.input_4 ∧ external.input_4 ≤ (1 : Int)) ∧
  ((0 : Int) ≤ external.input_5 ∧ external.input_5 ≤ (31 : Int))

structure Execution where
  «stage_0» : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1
  «stage_1» : (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4) × (Fin 5 → Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1) × Unit
  «stage_2» : Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 4 × Mxx.Primitives.ExactMatrix 1208925815113228574679041 32 1 1 × Unit
  «stage_3» : Int
  «requirement_0» : Bool
  «ideal» : Int

def stage_0_params : Stage_mask.Params := { «unit» := () }

def stage_1_params : Stage_encrypt.Params := { «unit» := () }

def stage_2_params : Stage_online_eval.Params := { «unit» := () }

def stage_3_params : Stage_decode.Params := { «unit» := () }

def requirement_0_params : Requirement_0.Params := { «unit» := () }

def ideal_params : Ideal.Params := { «unit» := () }

def Runs (hashModel : MxxRuntime.HashModel) (external : ExternalInputs) (tape : MxxRuntime.SampleTape)
    (execution : Execution) : Prop :=
  ValidExternals external ∧
  Stage_mask.generatedRoot tape [0] stage_0_params (()) execution.«stage_0» ∧
  Stage_encrypt.generatedRoot Backend.backend hashModel tape [1] stage_1_params ((execution.«stage_0», external.input_0, external.input_2, external.input_3, ())) execution.«stage_1» ∧
  Stage_online_eval.generatedRoot Backend.backend hashModel tape [2] stage_2_params ((execution.«stage_1».1, external.input_0, execution.«stage_1».2.1, execution.«stage_1».2.2.1, ())) execution.«stage_2» ∧
  Stage_decode.generatedRoot Backend.backend tape [3] stage_3_params ((execution.«stage_2».2.2.2.2.1, execution.«stage_0», execution.«stage_2».1, execution.«stage_2».2.2.1, external.input_4, external.input_5, execution.«stage_2».2.2.2.2.2.1, execution.«stage_2».2.1, execution.«stage_2».2.2.2.1, ())) execution.«stage_3» ∧
  Requirement_0.generatedRoot tape [4] requirement_0_params ((external.input_2, external.input_3, external.input_1, external.input_4, external.input_5, ())) execution.«requirement_0» ∧
  Ideal.generatedRoot tape [5] ideal_params (()) execution.«ideal» ∧
  execution.«requirement_0» = true

/-- Executions whose endpoint differs from the ideal one are rare. -/
def CorrectnessClaim : Prop :=
  ∀ hashModel external,
    MxxRuntime.tapeMeasure {tape | ∃ execution, Runs hashModel external tape execution ∧
      ¬ (execution.«stage_3» = execution.«ideal»)} ≤ (2 : ENNReal)⁻¹ ^ 100

end GeneratedClaim
