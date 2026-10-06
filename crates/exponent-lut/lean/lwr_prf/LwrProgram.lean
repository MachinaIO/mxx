import Mathlib.Data.Int.Basic

/-!
# The sparse-LWR readout

`SparseLwrPrf::build` adds the sparse-LWR PRF to a Program. With one member (the `lwr_prf`
parameter sets, one key each) it reads `z = ⟨a, k⟩ mod Q` through its terminal LUT, whose table is
`ρ(z) = (-1)^b (2 τ(r) - 1)` for `z = r + b Q/2`, with `τ(r) = ⌊p' r / (Q/2)⌋ + 1` and `p' = P/2`.
`LutOutputRepresentation::Coefficient` returns it as the constant coefficient, and the
finalization `G⁻¹(k)` and decryption modulo `2P` return `ρ(z) mod 2P`. The value is odd, so the
decoder weights it by `2^{-1}` to get a half-integer digit.
-/

namespace MxxExponentLutLwr

/-- The readout `τ(k) = ⌊p' k / w⌋ + 1` of width `w`, with `p' = P / 2`. -/
def readout (p' w k : Nat) : Nat := p' * k / w + 1

/-- The PRF table `2 τ(k) - 1` of width `w`, an odd value in `[1, P - 1]`. -/
def halfIntegerReadout (p' w k : Nat) : Nat := 2 * readout p' w k - 1

/-- The half-integer readout is odd. -/
theorem halfIntegerReadout_odd (p' w k : Nat) : halfIntegerReadout p' w k % 2 = 1 := by
  unfold halfIntegerReadout readout
  generalize p' * k / w = x
  omega

end MxxExponentLutLwr
