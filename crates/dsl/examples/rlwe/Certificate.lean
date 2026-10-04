import Claim
import RlweProof

/-! Checks that the handwritten proof has exactly the generated statement, and which axioms it
uses. -/

theorem certificate : GeneratedClaim.CorrectnessClaim := MxxRlweExample.correctness

#print axioms certificate
