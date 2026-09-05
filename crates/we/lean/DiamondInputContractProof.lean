import DiamondProofParameters
import Claim

open DiamondProofParameters

namespace DiamondGeneratedProof

theorem generated_raw_witness_boolean
    (hashModel : MxxRuntime.HashModel) (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) (hrun : GeneratedClaim.Runs hashModel external execution)
    (index : Fin circuitWidth) : external.input_6 index = 0 ∨ external.input_6 index = 1 := by
  have h := hrun.1.2.2.2.2.2.2.1 index
  omega

theorem generated_raw_instance_boolean
    (hashModel : MxxRuntime.HashModel) (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) (hrun : GeneratedClaim.Runs hashModel external execution)
    (index : Fin circuitWidth) : external.input_5 index = 0 ∨ external.input_5 index = 1 := by
  have h := hrun.1.2.2.2.2.2.1 index
  omega

theorem generated_hash_key_length
    (hashModel : MxxRuntime.HashModel) (external : GeneratedClaim.ExternalInputs)
    (execution : GeneratedClaim.Execution) (hrun : GeneratedClaim.Runs hashModel external execution) :
    external.input_8.size = 32 := hrun.1.2.2.2.2.2.2.2.2

#print axioms generated_raw_witness_boolean
#print axioms generated_raw_instance_boolean
#print axioms generated_hash_key_length

end DiamondGeneratedProof
