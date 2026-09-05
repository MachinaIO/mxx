import DiamondClaimFinalProof

open MxxWe GeneratedClaim DiamondProofParameters

namespace DiamondGeneratedProof

set_option maxRecDepth 8192

/-- The only remaining certificate premise is a closed numeric computation. -/
theorem generated_claim_correctness_of_capped_gate
    (hgate : cappedDiamondBound (decoderRadius q) n inner ell
      stage_0_params.diamond_error_max_coefficient_bound.toNat
      stage_0_params.diamond_preimage_max_coefficient_bound.toNat D
      stage_1_params.diamond_input_count.toNat stage_1_params.depth.toNat < decoderRadius q) :
    CorrectnessClaim := by
  have hraw := (cappedDiamondBound_lt_iff (decoderRadius q) n inner ell
    stage_0_params.diamond_error_max_coefficient_bound.toNat
    stage_0_params.diamond_preimage_max_coefficient_bound.toNat D
    stage_1_params.diamond_input_count.toNat stage_1_params.depth.toNat).mp hgate
  have hbound : claimFinalNoise < decoderRadius q := by
    have hb0 : claimInitialNoise = projectedInjectorBound n inner
        stage_0_params.diamond_error_max_coefficient_bound.toNat
        stage_0_params.diamond_preimage_max_coefficient_bound.toNat
        stage_1_params.diamond_input_count.toNat := by
      change (DiamondProofParameters.inner : Nat) * n * claimInjectorNoise *
        stage_0_params.diamond_preimage_max_coefficient_bound.toNat =
        DiamondProofParameters.inner * n * stage_0_params.diamond_preimage_max_coefficient_bound.toNat *
          claimInjectorNoise
      ac_rfl
    unfold claimFinalNoise
    rw [hb0]
    exact hraw
  intro hashModel external execution hrun
  exact generated_claim_decoder_of_approx hrun hbound (generated_claim_polynomial_bound hrun)

#print axioms generated_claim_correctness_of_capped_gate

end DiamondGeneratedProof
