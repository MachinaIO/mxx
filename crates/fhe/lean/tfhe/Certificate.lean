import TfheCorrectness

/-! The certificate of the generated TFHE scheme: one NAND gate on fresh public-key encryptions
decrypts correctly except with probability at most `2^-100`. `#print axioms` must list only
`propext`, `Classical.choice`, and `Quot.sound`. -/

namespace TfheCertificate

theorem correctness : GeneratedScheme.CorrectnessClaim := MxxFheTfhe.correctness

end TfheCertificate

#print axioms TfheCertificate.correctness
