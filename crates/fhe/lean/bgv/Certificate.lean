import Claim
import BgvProof

/-! Checks that the handwritten proof has exactly the generated statement, and which axioms it
uses. -/

theorem certificate : GeneratedClaim.CorrectnessClaim := MxxFheBgv.correctness

#print axioms certificate
