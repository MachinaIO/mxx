import Backend
import Stage_rlwe
import Ideal

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace GeneratedClaim

structure ExternalInputs where
  input_0 : ByteArray
  input_1 : Int

def ValidExternals (external : ExternalInputs) : Prop :=
  (external.input_0).size = 32 ∧
  ((0 : Int) ≤ external.input_1 ∧ external.input_1 ≤ (1 : Int))

structure Execution where
  «stage_0» : Mxx.Primitives.ExactMatrix 1532495540865518635130821056977027158796330141975560193 4096 1 1 × Bool × Unit
  «ideal» : Bool

def stage_0_params : Stage_rlwe.Params := { «crt_bits» := (60 : Int), «crt_depth» := (3 : Int), «cutoff» := (26 : Int), «gadget_base_bits» := (20 : Int), «sigma» := (4 / 1 : Rat) }

def ideal_params : Ideal.Params := { «crt_bits» := (60 : Int), «crt_depth» := (3 : Int), «cutoff» := (26 : Int), «gadget_base_bits» := (20 : Int), «sigma» := (4 / 1 : Rat) }

def Runs (hashModel : MxxRuntime.HashModel) (external : ExternalInputs)
    (execution : Execution) : Prop :=
  ValidExternals external ∧
  Stage_rlwe.generatedRoot hashModel stage_0_params ((external.input_0, external.input_1, ())) execution.«stage_0» ∧
  Ideal.generatedRoot ideal_params (external.input_1) execution.«ideal»

/-- Every execution's endpoint equals the ideal one. -/
def CorrectnessClaim : Prop :=
  ∀ hashModel external execution, Runs hashModel external execution →
    execution.«stage_0».2.1 = execution.«ideal»

end GeneratedClaim
