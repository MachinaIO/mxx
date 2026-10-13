import Backend
import Stage_setup_trapdoor
import Stage_setup_public
import Stage_payload
import Stage_prelude
import Stage_encrypt
import Stage_preprocess
import Stage_keygen
import Stage_online_eval
import Stage_decode
import Requirement_0
import Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace GeneratedClaim

structure ExternalInputs where
  input_0 : ByteArray
  input_1 : Fin 1 → Int
  input_2 : Fin 32768 → Int
  input_3 : Int
  input_4 : Int
  input_5 : Fin 1 → Int
  input_6 : Fin 2 → Int

def ValidExternals (external : ExternalInputs) : Prop :=
  (external.input_0).size = 32 ∧
  (∀ contract_i_0 : Fin 1, ((0 : Int) ≤ (external.input_1 contract_i_0) ∧ (external.input_1 contract_i_0) ≤ (1 : Int))) ∧
  (∀ contract_i_0 : Fin 32768, ((0 : Int) ≤ (external.input_2 contract_i_0) ∧ (external.input_2 contract_i_0) ≤ (1 : Int))) ∧
  ((1 : Int) ≤ external.input_3 ∧ external.input_3 ≤ (1 : Int)) ∧
  ((0 : Int) ≤ external.input_4 ∧ external.input_4 ≤ (0 : Int)) ∧
  (∀ contract_i_0 : Fin 1, ((0 : Int) ≤ (external.input_5 contract_i_0) ∧ (external.input_5 contract_i_0) ≤ (0 : Int))) ∧
  (∀ contract_i_0 : Fin 2, ((0 : Int) ≤ (external.input_6 contract_i_0) ∧ (external.input_6 contract_i_0) ≤ (1 : Int)))

structure Execution where
  «stage_0» : Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 14 × MxxRuntime.TrapdoorValue (Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 14) Unit × Unit
  «stage_1» : Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 14
  «stage_2» : Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 1
  «stage_3» : (Fin 1 → Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12) × (Fin 1 → Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12) × (Fin 1 → Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12) × (Fin 1 → Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12) × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 14 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 1 × Unit
  «stage_4» : (Fin 576 → Mxx.Primitives.ExactMatrix 2214732142572754471897509808394889702863060251185980682754060584134321625394313373321786988261522608528424961 32768 1 1) × (Fin 576 → Mxx.Primitives.ExactMatrix 2214732142572754471897509808394889702863060251185980682754060584134321625394313373321786988261522608528424961 32768 1 1) × (Fin 1 → Mxx.Primitives.ExactMatrix 2214732142572754471897509808394889702863060251185980682754060584134321625394313373321786988261522608528424961 32768 1 1) × (Fin 1 → Mxx.Primitives.ExactMatrix 2214732142572754471897509808394889702863060251185980682754060584134321625394313373321786988261522608528424961 32768 1 1) × Mxx.Primitives.ExactMatrix 2214732142572754471897509808394889702863060251185980682754060584134321625394313373321786988261522608528424961 32768 1 1 × Mxx.Primitives.ExactMatrix 2214732142572754471897509808394889702863060251185980682754060584134321625394313373321786988261522608528424961 32768 1 1 × (Fin 576 → Mxx.Primitives.ExactMatrix 2214732142572754471897509808394889702863060251185980682754060584134321625394313373321786988261522608528424961 32768 1 1) × (Fin 576 → Mxx.Primitives.ExactMatrix 2214732142572754471897509808394889702863060251185980682754060584134321625394313373321786988261522608528424961 32768 1 1) × Unit
  «stage_5» : Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 24 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 24 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Unit
  «stage_6» : Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 14 1 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 14 1 × (Fin 2 → Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 24 1) × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 24 1 × Unit
  «stage_7» : Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Mxx.Primitives.ExactMatrix 1509155612652322963894981558308021422531187968728956929 32768 1 12 × Unit
  «stage_8» : Fin 32768 → Int
  «requirement_0» : Bool
  «ideal» : Fin 32768 → Int

def stage_0_params : Stage_setup_trapdoor.Params := { «unit» := () }

def stage_1_params : Stage_setup_public.Params := { «unit» := () }

def stage_2_params : Stage_payload.Params := { «unit» := () }

def stage_3_params : Stage_prelude.Params := { «unit» := () }

def stage_4_params : Stage_encrypt.Params := { «unit» := () }

def stage_5_params : Stage_preprocess.Params := { «unit» := () }

def stage_6_params : Stage_keygen.Params := { «unit» := () }

def stage_7_params : Stage_online_eval.Params := { «unit» := () }

def stage_8_params : Stage_decode.Params := { «unit» := () }

def requirement_0_params : Requirement_0.Params := { «unit» := () }

def ideal_params : Ideal.Params := { «unit» := () }

def Runs (hashModel : MxxRuntime.HashModel) (external : ExternalInputs) (tape : MxxRuntime.SampleTape)
    (execution : Execution) : Prop :=
  ValidExternals external ∧
  Stage_setup_trapdoor.generatedRoot Backend.backend tape [0] stage_0_params (()) execution.«stage_0» ∧
  Stage_setup_public.generatedRoot tape [1] stage_1_params (execution.«stage_0».1) execution.«stage_1» ∧
  Stage_payload.generatedRoot tape [2] stage_2_params (()) execution.«stage_2» ∧
  Stage_prelude.generatedRoot Backend.backend hashModel tape [3] stage_3_params ((execution.«stage_1», external.input_0, execution.«stage_2», external.input_1, external.input_2, ())) execution.«stage_3» ∧
  Stage_encrypt.generatedRoot Backend.backend hashModel tape [4] stage_4_params ((execution.«stage_2», external.input_0, external.input_3, external.input_4, execution.«stage_3».1, execution.«stage_3».2.1, execution.«stage_3».2.2.1, execution.«stage_3».2.2.2.1, ())) execution.«stage_4» ∧
  Stage_preprocess.generatedRoot Backend.backend hashModel tape [5] stage_5_params ((external.input_5, external.input_0, external.input_6, ())) execution.«stage_5» ∧
  Stage_keygen.generatedRoot Backend.backend hashModel tape [6] stage_6_params ((execution.«stage_0».2.1, external.input_0, execution.«stage_5».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_5».2.2.2.2.2.2.2.2.2.2.2.1, ())) execution.«stage_6» ∧
  Stage_online_eval.generatedRoot Backend.backend hashModel tape [7] stage_7_params ((execution.«stage_5».2.2.2.2.2.2.2.2.2.2.1, execution.«stage_5».2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_5».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_5».2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.1, execution.«stage_4».2.2.2.2.2.1, execution.«stage_3».1, external.input_5, execution.«stage_3».2.1, execution.«stage_3».2.2.1, execution.«stage_3».2.2.2.1, execution.«stage_4».2.2.2.2.2.2.1, execution.«stage_4».2.2.2.2.2.2.2.1, external.input_0, external.input_6, execution.«stage_4».2.2.1, execution.«stage_4».2.2.2.1, execution.«stage_5».2.2.2.2.1, execution.«stage_5».2.2.2.2.2.1, execution.«stage_4».1, execution.«stage_4».2.1, execution.«stage_5».1, execution.«stage_5».2.1, execution.«stage_5».2.2.1, execution.«stage_5».2.2.2.1, ())) execution.«stage_7» ∧
  Stage_decode.generatedRoot Backend.backend hashModel tape [8] stage_8_params ((execution.«stage_3».2.2.2.2.2.1, external.input_0, execution.«stage_3».2.2.2.2.1, execution.«stage_6».1, execution.«stage_6».2.1, execution.«stage_4».2.2.1, execution.«stage_4».2.2.2.1, execution.«stage_6».2.2.1, execution.«stage_7».2.2.2.2.1, execution.«stage_7».2.2.2.2.2.1, execution.«stage_5».2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_5».2.2.2.2.2.2.2.2.2.2.2.2.2.1, execution.«stage_6».2.2.2.1, ())) execution.«stage_8» ∧
  Requirement_0.generatedRoot tape [9] requirement_0_params ((external.input_1, external.input_6, external.input_5, ())) execution.«requirement_0» ∧
  Ideal.generatedRoot tape [10] ideal_params (external.input_2) execution.«ideal» ∧
  execution.«requirement_0» = true

/-- Executions whose endpoint differs from the ideal one are rare. -/
def CorrectnessClaim : Prop :=
  ∀ hashModel external,
    MxxRuntime.tapeMeasure {tape | ∃ execution, Runs hashModel external tape execution ∧
      ¬ (execution.«stage_8» = execution.«ideal»)} ≤ (2 : ENNReal)⁻¹ ^ 128

end GeneratedClaim
