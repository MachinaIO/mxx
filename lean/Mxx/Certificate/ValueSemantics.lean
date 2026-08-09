import Mxx.Certificate.MatrixTypeSemantics
import Mxx.Certificate.Bounds

namespace Mxx.Certificate

/-! # Closed runtime value semantics

These predicates are shared by input contracts, ordinary matrix facts, and sequential recurrence
states. They inspect only the concrete parameter environment and runtime value; no certificate or
caller-supplied predicate participates in their definition.
-/

/-- Runtime meaning of analyzer-tracked coefficient representation. A centered norm never proves
canonical raw residues; that property is checked against the actual stored coefficients. -/
def CoefficientRepresentation.Holds
    (representation : CoefficientRepresentation)
    (parameters : Mxx.Ir.ParamEnvironment)
    (matrix : Mxx.Matrix) : Prop :=
  match representation with
  | .unknown => True
  | .canonicalResidues modulus =>
      ∃ evaluatedModulus,
        evaluateIntExpr parameters modulus = .ok evaluatedModulus ∧
        0 < evaluatedModulus ∧
        MatrixHasCanonicalResidues evaluatedModulus matrix

end Mxx.Certificate
