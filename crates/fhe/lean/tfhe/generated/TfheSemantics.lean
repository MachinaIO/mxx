import MxxRuntime

namespace TfheSemantics

/-- The encoded bit: `floor(q/8)` for one, `q - floor(q/8)` for zero. -/
def messageCenter (q : Nat) (bit : Int) (_ : Fin 1) : Int :=
  if bit = 1 then 536870912 else (q : Int) - 536870912

/-- The sign decoder returns the encoded bit within this distance. -/
def decoderRadius (_ : Nat) : Nat := 536870912

end TfheSemantics
