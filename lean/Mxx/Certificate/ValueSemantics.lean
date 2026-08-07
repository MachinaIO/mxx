import Mxx.Certificate.CanonicalResidues
import Mxx.Certificate.Bounds

namespace Mxx.Certificate

/-! # Closed runtime value semantics

These predicates are shared by input contracts, ordinary matrix facts, and sequential recurrence
states. They inspect only the concrete parameter environment and runtime value; no certificate or
caller-supplied predicate participates in their definition.
-/

/-- A runtime matrix has exactly the type obtained by evaluating the frozen matrix-type
expression. The sampler-only coefficient bound is intentionally irrelevant to runtime shape. -/
def MatrixTypeExpr.Holds
    (matrixType : MatrixTypeExpr)
    (parameters : Mxx.Ir.ParamEnvironment)
    (matrix : Mxx.Matrix) : Prop :=
  ∃ evaluated,
    matrixType.evaluate parameters = some evaluated ∧
    matrix.modulus = evaluated.modulus ∧
    matrix.ringDimension = evaluated.ringDimension ∧
    matrix.rows = evaluated.rows ∧
    matrix.columns = evaluated.columns

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
