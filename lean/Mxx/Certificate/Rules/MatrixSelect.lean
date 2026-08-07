import Mxx.Certificate.Normalize
import Mxx.Certificate.Rules.MatrixRules

namespace Mxx.Certificate

inductive MatrixSelectError where
  | emptyBranches
  | mismatchedBranchType (expected actual : MatrixTypeExpr)
  | unsupportedBasis
  | duplicateBasis
  | incompatibleBasisCoefficient
  | invalidConcreteIndex (index : Int)
  | unknownCoefficientType
  | exactEmbeddingTyping (error : TypingError)

private structure NormalizedSignalTerm where
  key : MatrixExprKey
  term : SignalTerm
  coefficientType : MatrixTypeExpr

private def normalizeSignalTerm
    (term : SignalTerm) : Except MatrixSelectError NormalizedSignalTerm := do
  let basis := term.basis
  let key ← match basis.key with
    | some key => pure key
    | none => throw .unsupportedBasis
  let coefficientType ← match term.coefficient.expression.inferType with
    | some type => pure type
    | none => throw .unknownCoefficientType
  -- `term` is an analyzer-produced or schema-validated input fact.  Re-inferring the basis type
  -- here would incorrectly reject typed structural embeddings whose denotation is already carried
  -- by the input fact.  The selector only needs the checked coefficient type and structural key.
  return { key, term, coefficientType }

private def findTerm? (key : MatrixExprKey) :
    List NormalizedSignalTerm → Option NormalizedSignalTerm
  | [] => none
  | term :: tail => if term.key == key then some term else findTerm? key tail

private def normalizeTerms
    (terms : List SignalTerm) : Except MatrixSelectError (List NormalizedSignalTerm) := do
  let normalized ← terms.mapM normalizeSignalTerm
  let rec rejectDuplicates : List NormalizedSignalTerm → Except MatrixSelectError Unit
    | [] => pure ()
    | head :: tail => do
        if (findTerm? head.key tail).isSome then throw .duplicateBasis
        rejectDuplicates tail
  rejectDuplicates normalized
  return normalized

private structure BasisEntry where
  key : MatrixExprKey
  basis : MatrixExpr
  coefficientType : MatrixTypeExpr
  mode : SignalProductMode

private def findBasis? (key : MatrixExprKey) : List BasisEntry → Option BasisEntry
  | [] => none
  | entry :: tail => if entry.key == key then some entry else findBasis? key tail

private def insertBasis
    (entries : List BasisEntry)
    (term : NormalizedSignalTerm) : Except MatrixSelectError (List BasisEntry) :=
  match findBasis? term.key entries with
  | none => .ok (entries ++ [{
      key := term.key
      basis := term.term.basis
      coefficientType := term.coefficientType
      mode := term.term.mode
    }])
  | some entry =>
      if entry.coefficientType == term.coefficientType && entry.mode == term.term.mode then
        .ok entries
      else .error .incompatibleBasisCoefficient

private def collectBasesFrom
    (entries : List BasisEntry) :
    List (List NormalizedSignalTerm) → Except MatrixSelectError (List BasisEntry)
  | [] => pure entries
  | terms :: tail => do
      let next ← terms.foldlM insertBasis entries
      collectBasesFrom next tail

private def collectBases
    (branches : List (List NormalizedSignalTerm)) : Except MatrixSelectError (List BasisEntry) :=
  collectBasesFrom [] branches

private def maximumBounds : List BoundExpr → BoundExpr
  | [] => .constant 0
  | head :: tail => tail.foldl .maximum head

private def selectTerm
    (index : RuntimeExpr .integer)
    (branches : List (List NormalizedSignalTerm))
    (entry : BasisEntry) : Except MatrixSelectError SignalTerm := do
  let selections ← branches.mapM fun terms =>
    match findTerm? entry.key terms with
    | none => pure (MatrixExpr.zero entry.coefficientType, BoundExpr.constant 0)
    | some term =>
        if term.coefficientType == entry.coefficientType && term.term.mode == entry.mode then
          pure (term.term.coefficient.expression, term.term.coefficient.normBound)
        else throw .incompatibleBasisCoefficient
  let coefficient : BoundedMatrixExpr := {
    expression := .select index (selections.map (·.1))
    normBound := maximumBounds (selections.map (·.2))
  }
  return {
    coefficient
    basis := entry.basis
    mode := entry.mode
  }

private def allExact : List MatrixFact → Option (List MatrixExpr)
  | [] => some []
  | fact :: tail => match fact.primary with
      | .exact expression => return expression :: (← allExact tail)
      | .affine _ => none

private def allAffine : List MatrixFact → Option (List AffineForm)
  | [] => some []
  | fact :: tail => match fact.primary with
      | .exact _ => none
      | .affine form => return form :: (← allAffine tail)

/-- Embed an exact matrix into the affine language as `I * A`.  This is an exact signal
representation, not an opaque-noise approximation. -/
private def exactAsAffine
    (outputType : MatrixTypeExpr)
    (expression : MatrixExpr) : Except MatrixSelectError AffineForm := do
  let coefficientType : MatrixTypeExpr := {
    outputType with columns := outputType.rows
  }
  let coefficient : BoundedMatrixExpr := {
    expression := .identity coefficientType
    normBound := .constant 1
  }
  let product ← inferMatrixProductType coefficientType outputType
    |>.mapError .exactEmbeddingTyping
  let term : SignalTerm := {
    coefficient
    basis := expression
    mode := product.mode
  }
  return { terms := [term], noiseBound := .constant 0 }

private def asAffine
    (outputType : MatrixTypeExpr) : MatrixFact → Except MatrixSelectError AffineForm
  | { primary := .exact expression, .. } => exactAsAffine outputType expression
  | { primary := .affine form, .. } => pure form

private def validateBranchTypes
    (expected : MatrixTypeExpr) :
    List (MatrixTypeExpr × MatrixFact) → Except MatrixSelectError Unit
  | [] => pure ()
  | (actual, _) :: tail => do
      if actual != expected then throw (.mismatchedBranchType expected actual)
      validateBranchTypes expected tail

/-- Derive a fail-closed matrix select. Exact branches remain an exact select. Affine branches are
normalized to a deterministic union of basis identities; a missing basis receives a typed-zero
coefficient. In a mixed selection, each exact branch is embedded exactly as `I * A`; unsupported
dynamic basis expressions are still rejected. -/
def deriveMatrixSelect
    (output : ValueInstanceRef)
    (outputType : MatrixTypeExpr)
    (index : RuntimeExpr .integer)
    (branches : List (MatrixTypeExpr × MatrixFact)) :
    Except MatrixSelectError MatrixFact := do
  if branches.isEmpty then throw .emptyBranches
  validateBranchTypes outputType branches
  match index with
  | .intConstant value =>
      if value < 0 then throw (.invalidConcreteIndex value)
      match branches[value.toNat]? with
      | none => throw (.invalidConcreteIndex value)
      | some (_, selected) => return {
          selected with
          subject := output
          relations := []
        }
  | _ => pure ()
  let facts := branches.map (·.2)
  let totalBound := maximumBounds (facts.map (·.totalNormBound))
  let primary ← match allExact facts, allAffine facts with
    | some expressions, _ => pure (.exact (.select index expressions))
    | none, some forms => do
        let normalized ← forms.mapM fun form => normalizeTerms form.terms
        let bases ← collectBases normalized
        let terms ← bases.mapM (selectTerm index normalized)
        pure (.affine {
          terms
          noiseBound := maximumBounds (forms.map (·.noiseBound))
        })
    | none, none => do
        let forms ← facts.mapM (asAffine outputType)
        let normalized ← forms.mapM fun form => normalizeTerms form.terms
        let bases ← collectBases normalized
        let terms ← bases.mapM (selectTerm index normalized)
        pure (.affine {
          terms
          noiseBound := maximumBounds (forms.map (·.noiseBound))
        })
  return {
    subject := output
    primary
    relations := []
    totalNormBound := totalBound
  }

private def selectFixtureType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 2
  columns := .constant 2

private def selectFixtureFact : MatrixFact := {
  subject := .protocolInput ⟨"branch"⟩
  primary := .exact (.zero selectFixtureType)
  relations := []
  totalNormBound := .constant 0
}

example : deriveMatrixSelect (.protocolInput ⟨"output"⟩) selectFixtureType
    (.intConstant 0) [(selectFixtureType, selectFixtureFact)] = .ok {
      selectFixtureFact with subject := .protocolInput ⟨"output"⟩
    } := by rfl

example : deriveMatrixSelect (.protocolInput ⟨"output"⟩) selectFixtureType
    (.intConstant 0) [] = .error .emptyBranches := by rfl

end Mxx.Certificate
