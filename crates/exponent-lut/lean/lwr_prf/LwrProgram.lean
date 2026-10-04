import Mathlib.Data.Int.Basic

/-!
# The sparse-LWR readout

Output `o` of `SparseLwrPrfProgram` reads `z = ⟨a, k_o⟩ mod Q` through the canonical readout
`ρ(z) = (-1)^b τ(r)` for `z = r + b Q/2`, with `τ(r) = ⌊p' r / (Q/2)⌋ + 1`. Decryption at the
terminal scale `⌊q/2⌋` returns the readout modulo `2`.
-/

namespace MxxExponentLutLwr

/-- The canonical readout table `τ(k) = ⌊p' k / w⌋ + 1` of width `w`. -/
def readout (p' w k : Nat) : Nat := p' * k / w + 1

end MxxExponentLutLwr
