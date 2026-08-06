import Mxx.Certificate.Rules.MatrixRules

namespace Mxx.Certificate

/-- Fail-closed errors for the matrix fact combinations used by Diamond. -/
inductive MatrixAffineError where
  | typing (error : TypingError)
  | unsupportedScale (scalar : IntExpr)
  | generalAffineProduct
  | unknownCoefficientType (expression : MatrixExpr)
  | unknownBasisType (expression : MatrixExpr)

structure DerivedMatrixFact where
  type : MatrixTypeExpr
  fact : MatrixFact

private def productBound
    (leftType rightType : MatrixTypeExpr)
    (leftBound rightBound : BoundExpr) : Except MatrixAffineError (MatrixProductType × BoundExpr) := do
  let product ← inferMatrixProductType leftType rightType |>.mapError .typing
  let inner := match product.mode with
    | .ordinaryMatrixProduct => leftType.columns
    | .leftPolynomialScalarBroadcast | .rightPolynomialScalarBroadcast |
        .swappedRowVectorScalarProduct => .constant 1
  return (product, .matrixProduct leftType.ringDimension inner leftBound rightBound)

private def negateSignalTerm (term : SignalTerm) : SignalTerm := {
  term with
  coefficient := {
    term.coefficient with
    expression := .negate term.coefficient.expression
  }
}

/-- Negate every signal coefficient and the opaque noise witness while preserving all hard
bounds. `AffineForm.Holds` is quotient-valued, so this is the algebraic identity
`-(Σ cᵢBᵢ + e) = Σ (-cᵢ)Bᵢ + (-e)` in `R_q`, independent of stored representatives. -/
def negateAffineForm (form : AffineForm) : AffineForm := {
  terms := form.terms.map negateSignalTerm
  noiseBound := form.noiseBound
}

/-- Kernel-checked algebraic step used by signal-bearing affine negation: negating a signal
coefficient negates the resulting ordered product in the exact negacyclic quotient. -/
theorem negateSignalCoefficient_matrixValue
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (coefficient basis : Mxx.Matrix)
    (coefficientLayout : Mxx.Toolkit.MatrixLayout coefficient q ringDimension rows inner)
    (basisLayout : Mxx.Toolkit.MatrixLayout basis q ringDimension inner columns) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMultiply (Mxx.matrixNegate coefficient) basis) =
      -Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMultiply coefficient basis) := by
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension rows inner columns
      (Mxx.matrixNegate coefficient) basis
      (Mxx.Toolkit.matrixNegate_layout coefficient coefficientLayout) basisLayout,
    Mxx.Toolkit.matrixValue_negate q ringDimension rows inner coefficient
      ⟨coefficientLayout.modulus, coefficientLayout.ringDimension,
        coefficientLayout.rows, coefficientLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension rows inner columns
      coefficient basis coefficientLayout basisLayout]
  simp

/-- Negation is exact for exact inputs and coefficient-wise for affine inputs. The output drops
sampler relations, as required by the closed relation-transport table. -/
def deriveMatrixNegate
    (output : ValueInstanceRef)
    (type : MatrixTypeExpr)
    (input : MatrixFact) : Except MatrixAffineError DerivedMatrixFact := do
  let primary ← match input.primary with
    | .exact expression => pure (.exact (.negate expression))
    | .affine form => pure (.affine (negateAffineForm form))
  return {
    type
    fact := {
      subject := output
      primary
      relations := []
      totalNormBound := input.totalNormBound
    }
  }

/-- Diamond uses `MatrixScale(1)` to materialize a preimage/decomposition output as a matrix.
The operation preserves the hard bound and retargets sampler relations to the new subject. An
affine input is conservatively materialized as one bounded value; no signal is invented. -/
def deriveMatrixScaleOne
    (output : ValueInstanceRef)
    (type : MatrixTypeExpr)
    (scalar : IntExpr)
    (input : MatrixFact) : Except MatrixAffineError DerivedMatrixFact := do
  if scalar != .constant 1 then throw (.unsupportedScale scalar)
  let primary := match input.primary with
    | .exact expression => .exact (.scalarMultiply (.constant 1) expression)
    | .affine _ => .affine { terms := [], noiseBound := input.totalNormBound }
  return {
    type
    fact := {
      subject := output
      primary
      relations := input.relations.map (MatrixRelation.retargetSubject output)
      totalNormBound := input.totalNormBound
    }
  }

/-- Executable and hard-bound soundness for Diamond's only supported scale operation.  This is
separate from relation transport: sampler relations are equations in `R_q` and are retargeted by
the quotient theorems in `MatrixRules`. -/
theorem matrixScaleOneNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (input : Mxx.Matrix)
    (outputCount q bound : Nat)
    [NeZero q]
    (argumentsEvaluate :
      [inputRef].mapM (fun wire ↦ Mxx.Ir.lookupWire wire wires) = some [.matrix input])
    (modulus : input.modulus = q)
    (inputNorm : Mxx.maxCenteredCoefficientNorm input ≤ bound)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .matrixScale (.constant 1)
      arguments := [inputRef]
      outputCount
    }) :
    values = [.matrix (Mxx.matrixScale 1 input)] ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixScale 1 input) ≤ bound := by
  constructor
  · exact Mxx.Ir.mem_evaluateNode_matrixScale_of_arguments runChild samplers params inputs wires
      inputRef input (.constant 1) 1 outputCount argumentsEvaluate rfl member
  · exact le_trans (Mxx.Toolkit.matrixScale_norm_le q 1 input modulus) (by simpa using inputNorm)

private def boundedView (type : MatrixTypeExpr) (fact : MatrixFact) : BoundedMatrixExpr := {
  expression := .wire { value := fact.subject, type }
  normBound := fact.totalNormBound
}

private def multiplyAffineRight
    (leftType rightType : MatrixTypeExpr)
    (form : AffineForm)
    (right : BoundedMatrixExpr) : Except MatrixAffineError AffineForm := do
  let terms ← form.terms.mapM fun term ↦ do
    if term.coefficient.expression.inferType.isNone then
      throw (.unknownCoefficientType term.coefficient.expression)
    let basis := MatrixExpr.multiply term.basis right.expression
    if basis.inferType.isNone then throw (.unknownBasisType basis)
    mkSignalTerm term.coefficient basis |>.mapError .typing
  let (_, noiseBound) ← productBound leftType rightType form.noiseBound right.normBound
  return { terms, noiseBound }

private def multiplyAffineLeft
    (leftType rightType : MatrixTypeExpr)
    (left : BoundedMatrixExpr)
    (form : AffineForm) : Except MatrixAffineError AffineForm := do
  let terms ← form.terms.mapM fun term ↦ do
    let coefficientType ← match term.coefficient.expression.inferType with
      | some type => pure type
      | none => throw (.unknownCoefficientType term.coefficient.expression)
    let (_, coefficientBound) ← productBound leftType coefficientType
      left.normBound term.coefficient.normBound
    let coefficient := MatrixExpr.multiply left.expression term.coefficient.expression
    if coefficient.inferType.isNone then throw (.unknownCoefficientType coefficient)
    mkSignalTerm {
      expression := coefficient
      normBound := coefficientBound
    } term.basis |>.mapError .typing
  let (_, noiseBound) ← productBound leftType rightType left.normBound form.noiseBound
  return { terms, noiseBound }

/-- Closed Diamond multiplication combinations. At most one operand may carry signal terms.
Exact/exact stays exact; bounded-only products stay bounded; multiplying an affine signal on the
right or left transports its terms without inventing a bilinear signal decomposition. -/
def deriveMatrixMultiply
    (output : ValueInstanceRef)
    (leftType rightType : MatrixTypeExpr)
    (left right : MatrixFact) : Except MatrixAffineError DerivedMatrixFact := do
  let (product, totalBound) ← productBound leftType rightType
    left.totalNormBound right.totalNormBound
  let leftView := boundedView leftType left
  let rightView := boundedView rightType right
  let primary ← match left.primary, right.primary with
    | .exact leftExpression, .exact rightExpression =>
        pure (.exact (.multiply leftExpression rightExpression))
    | .affine leftForm, .exact rightExpression =>
        if leftForm.terms.isEmpty then
          pure (.affine {
            terms := [← mkSignalTerm leftView rightExpression |>.mapError .typing]
            noiseBound := .constant 0
          })
        else
          pure (.affine (← multiplyAffineRight leftType rightType leftForm {
            expression := rightExpression
            normBound := right.totalNormBound
          }))
    | .exact _, .affine rightForm =>
        if rightForm.terms.isEmpty then
          pure (.affine { terms := [], noiseBound := totalBound })
        else
          pure (.affine (← multiplyAffineLeft leftType rightType leftView rightForm))
    | .affine leftForm, .affine rightForm =>
        if leftForm.terms.isEmpty && rightForm.terms.isEmpty then
          pure (.affine { terms := [], noiseBound := totalBound })
        else if !leftForm.terms.isEmpty && rightForm.terms.isEmpty then
          pure (.affine (← multiplyAffineRight leftType rightType leftForm rightView))
        else if leftForm.terms.isEmpty && !rightForm.terms.isEmpty then
          pure (.affine (← multiplyAffineLeft leftType rightType leftView rightForm))
        else throw .generalAffineProduct
  return {
    type := product.output
    fact := {
      subject := output
      primary
      relations := []
      totalNormBound := totalBound
    }
  }

private def fixtureType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 2
  columns := .constant 2

private def fixtureBounded (name : String) : MatrixFact := {
  subject := .protocolInput ⟨name⟩
  primary := .affine { terms := [], noiseBound := .constant 3 }
  relations := []
  totalNormBound := .constant 3
}

private def fixtureSignal (name : String) : MatrixFact := {
  subject := .protocolInput ⟨name⟩
  primary := .affine {
    terms := [{
      coefficient := {
        expression := .identity fixtureType
        normBound := .constant 1
      }
      basis := .wire { value := .protocolInput ⟨name ++ "-basis"⟩, type := fixtureType }
      mode := .ordinaryMatrixProduct
    }]
    noiseBound := .constant 3
  }
  relations := []
  totalNormBound := .constant 7
}

example : (deriveMatrixMultiply (.protocolInput ⟨"out"⟩) fixtureType fixtureType
    (fixtureBounded "left") (fixtureBounded "right")).isOk := by decide

example : (deriveMatrixNegate (.protocolInput ⟨"out"⟩) fixtureType
    (fixtureSignal "input")).isOk := by decide

example : deriveMatrixScaleOne (.protocolInput ⟨"out"⟩) fixtureType (.constant 2)
    (fixtureBounded "input") = .error (.unsupportedScale (.constant 2)) := rfl

end Mxx.Certificate
