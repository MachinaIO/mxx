import BgvReductions
import BgvDepth

/-! The certificate of the generated BGV scheme: the correctness of its circuit and its IND-CPA
statement with reductions whose simulators are IR programs of bounded cost, proven from the
declared assumptions only. `#print axioms` must list only `propext`,
`Classical.choice`, and `Quot.sound`. -/

namespace BgvCertificate

theorem correctness : GeneratedScheme.CorrectnessClaim := BgvCorrect.correctness

theorem bounded_security : GeneratedScheme.BoundedSecurityClaim := BgvSecurity.bounded_security

end BgvCertificate

#print axioms BgvCertificate.correctness
#print axioms BgvCertificate.bounded_security
