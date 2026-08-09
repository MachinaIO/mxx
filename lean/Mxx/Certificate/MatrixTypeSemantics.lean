import Mxx.Certificate.CanonicalResidues
import Mxx.Certificate.Identity

namespace Mxx.Certificate

/-! Runtime matrix-shape semantics shared by the operational checker and input contracts. -/

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
    matrix.columns = evaluated.columns ∧
    matrix.WellFormed

end Mxx.Certificate
