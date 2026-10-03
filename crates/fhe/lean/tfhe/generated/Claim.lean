import Backend
import Stage_keygen
import Stage_encrypt_left
import Stage_encrypt_right
import Stage_nand
import Stage_decrypt
import Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace GeneratedClaim

structure ExternalInputs where
  input_0 : ByteArray
  input_1 : Int
  input_2 : ByteArray
  input_3 : Int
  input_4 : ByteArray

def ValidExternals (external : ExternalInputs) : Prop :=
  (external.input_0).size = 32 ∧
  ((0 : Int) ≤ external.input_1 ∧ external.input_1 ≤ (1 : Int)) ∧
  (external.input_2).size = 32 ∧
  ((0 : Int) ≤ external.input_3 ∧ external.input_3 ≤ (1 : Int)) ∧
  (external.input_4).size = 32

structure Execution where
  «stage_0» : (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 630 → Mxx.Primitives.ExactMatrix 5234636801 1024 1 12) × (Fin 5160960 → Int) × (Fin 8192 → Int) × (Fin 630 → Int) × Unit
  «stage_1» : (Fin 630 → Int) × Int × Unit
  «stage_2» : (Fin 630 → Int) × Int × Unit
  «stage_3» : (Fin 630 → Int) × Int × Unit
  «stage_4» : Int
  «ideal» : Int

def stage_0_params : Stage_keygen.Params := { «unit» := () }

def stage_1_params : Stage_encrypt_left.Params := { «unit» := () }

def stage_2_params : Stage_encrypt_right.Params := { «unit» := () }

def stage_3_params : Stage_nand.Params := { «unit» := () }

def stage_4_params : Stage_decrypt.Params := { «unit» := () }

def ideal_params : Ideal.Params := { «unit» := () }

def Runs (hashModel : MxxRuntime.HashModel) (external : ExternalInputs) (tape : MxxRuntime.SampleTape)
    (execution : Execution) : Prop :=
  ValidExternals external ∧
  Stage_keygen.generatedRoot Backend.backend hashModel tape [0] stage_0_params (external.input_0) execution.«stage_0» ∧
  Stage_encrypt_left.generatedRoot hashModel tape [1] stage_1_params ((external.input_2, execution.«stage_0».2.2.2.2.1, external.input_1, ())) execution.«stage_1» ∧
  Stage_encrypt_right.generatedRoot hashModel tape [2] stage_2_params ((external.input_4, execution.«stage_0».2.2.2.2.1, external.input_3, ())) execution.«stage_2» ∧
  Stage_nand.generatedRoot Backend.backend tape [3] stage_3_params ((execution.«stage_0».2.2.1, execution.«stage_1».2.1, execution.«stage_2».2.1, execution.«stage_1».1, execution.«stage_2».1, execution.«stage_0».1, execution.«stage_0».2.1, execution.«stage_0».2.2.2.1, ())) execution.«stage_3» ∧
  Stage_decrypt.generatedRoot tape [4] stage_4_params ((execution.«stage_3».2.1, execution.«stage_3».1, execution.«stage_0».2.2.2.2.1, ())) execution.«stage_4» ∧
  Ideal.generatedRoot tape [5] ideal_params ((external.input_1, external.input_3, ())) execution.«ideal»

/-- Executions whose endpoint differs from the ideal one are rare. -/
def CorrectnessClaim : Prop :=
  ∀ hashModel external,
    MxxRuntime.tapeMeasure {tape | ∃ execution, Runs hashModel external tape execution ∧
      ¬ (execution.«stage_4» = execution.«ideal»)} ≤ (2 : ENNReal)⁻¹ ^ 128

end GeneratedClaim
