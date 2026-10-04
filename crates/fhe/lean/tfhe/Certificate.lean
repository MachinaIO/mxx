import Claim
import TfheProof

/-! Checks that the handwritten proof has exactly the generated statement, and which axioms it
uses. -/

theorem certificate : GeneratedClaim.CorrectnessClaim := MxxFheTfhe.correctness

#print axioms certificate
