import MxxRuntime

namespace BgvSemantics

/-- No message is subtracted: the centered decryption phase is the residual. -/
def messageCenter (_ : Nat) (_ : Fin 8192 → Int) (_ : Fin 8192) : Int := 0

/-- One above the static bound on the centered decryption phase. -/
def decoderRadius (_ : Nat) : Nat := 56258131176

end BgvSemantics
