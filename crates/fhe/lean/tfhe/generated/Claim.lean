import Backend
import TfheSemantics
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
  «stage_0» : (Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) × (Fin 630 → Mxx.Primitives.ExactMatrix 4611255024196841473 2048 1 16) × (Fin 10321920 → Int) × (Fin 16384 → Int) × (Fin 630 → Int) × Unit
  «stage_1» : (Fin 630 → Int) × Int × Unit
  «stage_2» : (Fin 630 → Int) × Int × Unit
  «stage_3» : (Fin 630 → Int) × Int × Unit
  «stage_4» : Int × Int × Unit
  «ideal» : Int

def stage_0_params : Stage_keygen.Params := { «unit» := () }

def stage_1_params : Stage_encrypt_left.Params := { «unit» := () }

def stage_2_params : Stage_encrypt_right.Params := { «unit» := () }

def stage_3_params : Stage_nand.Params := { «unit» := () }

def stage_4_params : Stage_decrypt.Params := { «unit» := () }

def ideal_params : Ideal.Params := { «unit» := () }

def Runs (hashModel : MxxRuntime.HashModel) (external : ExternalInputs)
    (execution : Execution) : Prop :=
  ValidExternals external ∧
  Stage_keygen.generatedRoot FheBackend.backend hashModel stage_0_params (external.input_0) execution.«stage_0» ∧
  Stage_encrypt_left.generatedRoot hashModel stage_1_params ((external.input_2, execution.«stage_0».2.2.2.2.1, external.input_1, ())) execution.«stage_1» ∧
  Stage_encrypt_right.generatedRoot hashModel stage_2_params ((external.input_4, execution.«stage_0».2.2.2.2.1, external.input_3, ())) execution.«stage_2» ∧
  Stage_nand.generatedRoot FheBackend.backend stage_3_params ((execution.«stage_0».2.2.1, execution.«stage_1».2.1, execution.«stage_2».2.1, execution.«stage_1».1, execution.«stage_2».1, execution.«stage_0».1, execution.«stage_0».2.1, execution.«stage_0».2.2.2.1, ())) execution.«stage_3» ∧
  Stage_decrypt.generatedRoot stage_4_params ((execution.«stage_3».2.1, execution.«stage_3».1, execution.«stage_0».2.2.2.2.1, ())) execution.«stage_4» ∧
  Ideal.generatedRoot ideal_params ((external.input_1, external.input_3, ())) execution.«ideal»

noncomputable def observedResidual (execution : Execution) (index : Fin 1) : Int :=
  Mxx.Primitives.centeredLift 4294967296
    (((execution.«stage_4».2.1 : Int) : ZMod 4294967296) -
      (TfheSemantics.messageCenter 4294967296 (execution.«ideal») index : ZMod 4294967296))

/-- The application proof must establish this proposition; no noise premise is assumed. -/
def CorrectnessClaim : Prop :=
  ∀ hashModel external execution, Runs hashModel external execution →
    (∀ index, (observedResidual execution index).natAbs < TfheSemantics.decoderRadius 4294967296) ∧
    execution.«stage_4».1 = execution.«ideal»

end GeneratedClaim
