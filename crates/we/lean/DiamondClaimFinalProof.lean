import DiamondProofParameters
import DiamondClaimStateProof
import DiamondFinalDecryptionProof
import DiamondFinalEncodingProof
import DiamondClaimCircuitProof

open Mxx.Primitives MxxRuntime GeneratedClaim
open DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxRecDepth 8192

theorem generated_claim_ideal {hashModel external execution}
    (hrun : Runs hashModel external execution) : execution.ideal = external.input_7 :=
  hrun.2.2.2.2.2.2.1

/-- The final whole-polynomial estimate supplies the observed residual and operational decoder.
    The numeric gate and estimate are local obligations, not a proof of CorrectnessClaim. -/
theorem generated_claim_decoder_of_approx {hashModel external execution B}
    (hrun : Runs hashModel external execution)
    (hbound : B < MxxWe.decoderRadius q)
    (happrox : Approx execution.stage_1.2.2.2.2.1
      (matrixPolynomial [(MxxWe.messageCenter q external.input_7 : Int)]) B) :
    (observedResidual execution).natAbs < MxxWe.decoderRadius q ∧
      execution.stage_1.2.2.2.2.2.1 = execution.ideal := by
  have hcenter :
      ((matrixPolynomial [(MxxWe.messageCenter q external.input_7 : Int)] :
        ExactMatrix q n 1 1) 0 0).coeff ⟨0, by decide⟩ =
        (MxxWe.messageCenter q external.input_7 : ZMod q) := by
    letI : Fact (1 < q) := ⟨by decide⟩
    change (((MxxWe.messageCenter q external.input_7 : Int) : ExactPoly q n) +
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) * 0).coeff _ = _
    rw [mul_zero, add_zero]
    simp only [Int.cast_natCast]
    have hc := Negacyclic.coeff_root_pow (R := ZMod q) (by decide : 0 < n)
      (⟨0, by decide⟩ : Fin n) (⟨0, by decide⟩ : Fin n)
    simp only [pow_zero, ite_true] at hc
    rw [show (MxxWe.messageCenter q external.input_7 : ExactPoly q n) =
      (MxxWe.messageCenter q external.input_7 : ZMod q) •
        (1 : ExactPoly q n) by simp [Algebra.smul_def]]
    rw [Algebra.smul_def, Negacyclic.coeff_smul, hc, mul_one]
  have h := final_approx_decoder (by decide : 4 ≤ q) (by decide : 0 < n)
    hbound external.input_7 _ _ happrox hcenter
  have hideal := generated_claim_ideal hrun
  constructor
  · simpa only [observedResidual, hideal] using h.1
  · rw [generated_final_decoder DiamondBackend.backend stage_1_params rfl hrun.2.2.1,
      hideal]
    exact h.2

#print axioms generated_claim_ideal
#print axioms generated_claim_decoder_of_approx

def claimFinalNoise : Nat :=
  2 * claimInitialNoise +
    a * (claimInitialNoise + factor ^ stage_1_params.depth.toNat * claimInitialNoise)

/-- Every approximation and shared-key equation is derived from the linked generated runs. -/
theorem generated_claim_polynomial_bound {hashModel external execution}
    (hrun : Runs hashModel external execution) :
    Approx execution.stage_1.2.2.2.2.1
      (matrixPolynomial [(MxxWe.messageCenter q external.input_7 : Int)])
      claimFinalNoise := by
  obtain ⟨w⟩ := generated_claim_injector hashModel external execution hrun
  obtain ⟨selector, hsecret, hmessage, hstate⟩ := claim_zero_state_encoding w
  obtain ⟨state, circuit, key, hget, hencoding, hkey, hresidual⟩ :=
    generated_claim_accepting_ciphertext w hrun
  obtain ⟨position, hposition, hget⟩ := hget
  have hp : position = 0 := Fin.ext (by change (position.val : Int) = 0 at hposition; omega)
  have hs : state = execution.stage_1.2.2.2.2.2.2.1 0 := by simpa only [hp] using hget
  have hc : Approx circuit
      (matrixMul ((fun _ _ ↦ selector 0 0) : ExactMatrix q n 1 1)
          execution.stage_0.2.2.2.2.2.2.2.2.2.1 -
        matrixMul ((fun _ _ ↦ selector 0 0) : ExactMatrix q n 1 1)
          w.finalPublic.gadget)
      (factor ^ stage_1_params.depth.toNat * claimInitialNoise) := by
    simpa only [BooleanEncodingWithin, one_smul, claimCircuitSecret, hsecret, hkey,
      initial_registered_gadget stage_0_params w.finalPublic, matrixMul] using hencoding
  have h := final_residual_from_encrypt_run hashModel stage_0_params hrun.2.1 w.finalPublic
    rfl selector external.input_7 hmessage state circuit
    (B := claimInjectorNoise) (BH := factor ^ stage_1_params.depth.toNat * claimInitialNoise)
    (hs ▸ hstate) hc
  rw [hresidual]
  simpa only [claimFinalNoise, claimInitialNoise] using h

#print axioms generated_claim_polynomial_bound

end DiamondGeneratedProof
