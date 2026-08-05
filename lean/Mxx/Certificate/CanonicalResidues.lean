import Mathlib.Data.List.GetD
import Mxx.Ir.ExecutionFacts

namespace Mxx.Certificate

/-- A raw stored coefficient is the canonical representative modulo a positive modulus. -/
def CanonicalResidue (modulus value : Int) : Prop :=
  0 ≤ value ∧ value < modulus

/-- Runtime predicate underlying `canonicalResidues(q)`. It concerns the actual stored integers,
not their centered representatives. -/
def MatrixHasCanonicalResidues (modulus : Int) (matrix : Mxx.Matrix) : Prop :=
  matrix.modulus = modulus ∧ ∀ value ∈ matrix.coefficients, CanonicalResidue modulus value

end Mxx.Certificate
