import DiamondClaimCorrectnessProof
import NumericCertificate

namespace DiamondCertificate

/-- Final correctness theorem for the candidate's mechanically generated IR claim.
Audit `GeneratedClaim.CorrectnessClaim` and its `Runs` definition in the generated `Claim.lean`:
valid accepting executions have residual below the decoder radius and decode the ideal message.
The numeric gate is proved in `NumericCertificate.lean`, not assumed by this theorem. -/
theorem correctness : GeneratedClaim.CorrectnessClaim :=
  DiamondGeneratedProof.generated_claim_correctness_of_capped_gate
    DiamondNumericCertificate.numeric_gate

#print axioms correctness

end DiamondCertificate
