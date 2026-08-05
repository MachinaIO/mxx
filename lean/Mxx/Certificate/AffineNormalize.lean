import Mxx.Certificate.Normalize
import Mxx.Certificate.Semantics
import Mxx.Certificate.Typing

namespace Mxx.Certificate

/-- Fail-closed errors from affine normalization.  In particular, basis equality is accepted only
when `MatrixExpr.sameSupported` returns an equality proof. -/
inductive AffineNormalizeError where
  | unknownCoefficientType
  | incompatibleBasisCoefficient
  | typing (error : TypingError)

/-- Kernel-checkable evidence emitted for every affine-normalization step.  This is analysis
evidence only; it is neither an executable IR node nor a certificate-supplied assertion. -/
inductive AffineNormalizationAction where
  | normalizeTerm
      (source result : SignalTerm)
      (coefficientRewrite : MatrixRewrite source.coefficient.expression
        result.coefficient.expression)
      (basisRewrite : MatrixRewrite source.basis result.basis)
      (modePreserved : result.mode = source.mode)
      (boundPreserved : result.coefficient.normBound = source.coefficient.normBound)
  | mergeTerms
      (left right result : SignalTerm)
      (basisEqual : left.basis = right.basis)
      (modeEqual : left.mode = right.mode)
      (coefficientRewrite : MatrixRewrite
        (.add left.coefficient.expression right.coefficient.expression)
        result.coefficient.expression)
      (boundIsSum : result.coefficient.normBound =
        .add left.coefficient.normBound right.coefficient.normBound)
  | eliminateZero
      (source : SignalTerm)
      (coefficientType : MatrixTypeExpr)
      (coefficientRewrite : MatrixRewrite source.coefficient.expression (.zero coefficientType))

structure AffineNormalizationResult where
  form : AffineForm
  actions : List AffineNormalizationAction

private structure TermEntry where
  term : SignalTerm
  coefficientType : MatrixTypeExpr

private def normalizeTerm
    (source : SignalTerm) :
    Except AffineNormalizeError (Option TermEntry × List AffineNormalizationAction) := do
  let coefficientType ← match source.coefficient.expression.inferType with
    | some type => pure type
    | none => throw .unknownCoefficientType
  let ⟨basis, basisRewrite⟩ := normalizeMatrixExprWithProof source.basis
  let ⟨coefficient, coefficientRewrite⟩ :=
    normalizeCoefficientWithProof coefficientType source.coefficient.expression
  let bounded : BoundedMatrixExpr := {
    expression := coefficient
    normBound := source.coefficient.normBound
  }
  let inferred ← mkSignalTerm bounded basis |>.mapError .typing
  if modeEqual : inferred.mode = source.mode then
    let result : SignalTerm := { coefficient := bounded, basis, mode := source.mode }
    let normalizeAction : AffineNormalizationAction :=
      .normalizeTerm source result coefficientRewrite basisRewrite rfl rfl
    match coefficient.sameTypedZero coefficientType with
    | .equal zeroEqual =>
      let resultZero : MatrixRewrite result.coefficient.expression (.zero coefficientType) := by
        exact zeroEqual ▸ .refl coefficient
      return (none, [normalizeAction, .eliminateZero result coefficientType resultZero])
    | .unknown => return (some { term := result, coefficientType }, [normalizeAction])
  else throw .incompatibleBasisCoefficient

private def mergeEntries
    (left right : TermEntry) :
    Except AffineNormalizeError (Option TermEntry × List AffineNormalizationAction) := do
  match left.term.basis.sameSupported right.term.basis with
  | .unknown => return (none, [])
  | .equal basisEqual =>
      if coefficientTypeEqual : left.coefficientType = right.coefficientType then
        if modeEqual : left.term.mode = right.term.mode then
          let coefficientExpression :=
            .add left.term.coefficient.expression right.term.coefficient.expression
          let ⟨coefficient, coefficientRewrite⟩ :=
            normalizeCoefficientWithProof left.coefficientType coefficientExpression
          let bounded : BoundedMatrixExpr := {
            expression := coefficient
            normBound := .add left.term.coefficient.normBound right.term.coefficient.normBound
          }
          let inferred ← mkSignalTerm bounded left.term.basis |>.mapError .typing
          if resultModeEqual : inferred.mode = left.term.mode then
            let result : SignalTerm := {
              coefficient := bounded
              basis := left.term.basis
              mode := left.term.mode
            }
            let action : AffineNormalizationAction :=
              .mergeTerms left.term right.term result basisEqual modeEqual coefficientRewrite rfl
            match coefficient.sameTypedZero left.coefficientType with
            | .equal zeroEqual =>
              let resultZero : MatrixRewrite result.coefficient.expression
                  (.zero left.coefficientType) := by
                exact zeroEqual ▸ .refl coefficient
              return (none, [action, .eliminateZero result left.coefficientType resultZero])
            | .unknown =>
                return (some { term := result, coefficientType := left.coefficientType }, [action])
          else throw .incompatibleBasisCoefficient
        else throw .incompatibleBasisCoefficient
      else throw .incompatibleBasisCoefficient

/-- Insert one normalized term into a deterministic accumulator.  An unsupported comparison is
not treated as inequality evidence: the search simply continues and keeps both terms. -/
private def insertTerm
    (entry : TermEntry) :
    List TermEntry →
    Except AffineNormalizeError (List TermEntry × List AffineNormalizationAction)
  | [] => return ([entry], [])
  | head :: tail => do
      let (merged, actions) ← mergeEntries head entry
      match merged, actions with
      | none, [] =>
          let (tail', tailActions) ← insertTerm entry tail
          return (head :: tail', tailActions)
      | none, actions => return (tail, actions)
      | some merged, actions => return (merged :: tail, actions)

private def normalizeTerms :
    List SignalTerm →
    List TermEntry →
    List AffineNormalizationAction →
    Except AffineNormalizeError (List TermEntry × List AffineNormalizationAction)
  | [], entries, actions => return (entries, actions)
  | term :: tail, entries, actions => do
      let (normalized, normalizeActions) ← normalizeTerm term
      match normalized with
      | none => normalizeTerms tail entries (actions ++ normalizeActions)
      | some entry =>
          let (entries', mergeActions) ← insertTerm entry entries
          normalizeTerms tail entries' (actions ++ normalizeActions ++ mergeActions)

/-- Normalize an affine form without changing its opaque-noise bound.  Equal normalized bases are
merged by coefficient addition; a coefficient is removed only after the syntax normalizer emits
an explicit proof that it is the correctly typed zero.  Addition of coefficient bounds is the
conservative triangle-inequality update. -/
def normalizeAffineForm
    (input : AffineForm) : Except AffineNormalizeError AffineNormalizationResult := do
  let (entries, actions) ← normalizeTerms input.terms [] []
  return {
    form := {
      terms := entries.map (·.term)
      noiseBound := input.noiseBound
    }
    actions
  }

/-- Quotient-algebra justification for merging two equal-basis signal terms.  It proves that the
runtime product of the summed coefficient has exactly the same value in `R_q` as the sum of the
two original products. -/
theorem mergedCoefficientProduct_matrixValue
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (left right basis : Mxx.Matrix)
    (leftLayout : Mxx.Toolkit.MatrixLayout left q ringDimension rows inner)
    (rightLayout : Mxx.Toolkit.MatrixLayout right q ringDimension rows inner)
    (basisLayout : Mxx.Toolkit.MatrixLayout basis q ringDimension inner columns) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMultiply (Mxx.matrixAdd left right) basis) =
      Mxx.Toolkit.matrixValue q ringDimension rows columns
          (Mxx.matrixMultiply left basis) +
        Mxx.Toolkit.matrixValue q ringDimension rows columns
          (Mxx.matrixMultiply right basis) := by
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension rows inner columns
      (Mxx.matrixAdd left right) basis
      (Mxx.Toolkit.matrixAdd_layout left right leftLayout rightLayout) basisLayout,
    Mxx.Toolkit.matrixValue_add q ringDimension rows inner left right
      ⟨leftLayout.modulus, leftLayout.ringDimension, leftLayout.rows, leftLayout.columns⟩
      ⟨rightLayout.modulus, rightLayout.ringDimension, rightLayout.rows, rightLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension rows inner columns
      left basis leftLayout basisLayout,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension rows inner columns
      right basis rightLayout basisLayout]
  ext row column
  change
    (∑ index,
      (Mxx.Toolkit.matrixValue q ringDimension rows inner left row index +
        Mxx.Toolkit.matrixValue q ringDimension rows inner right row index) *
        Mxx.Toolkit.matrixValue q ringDimension inner columns basis index column) =
      (∑ index,
        Mxx.Toolkit.matrixValue q ringDimension rows inner left row index *
          Mxx.Toolkit.matrixValue q ringDimension inner columns basis index column) +
      ∑ index,
        Mxx.Toolkit.matrixValue q ringDimension rows inner right row index *
          Mxx.Toolkit.matrixValue q ringDimension inner columns basis index column
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro index _
  exact add_mul _ _ _

/-- Quotient-algebra justification for deleting the signal term whose coefficient normalizes to
`x + (-x)`.  The product is the zero matrix in `R_q`, so deleting the term does not move any
bounded quantity into the opaque noise component. -/
theorem cancelledCoefficientProduct_matrixValue
    (q ringDimension rows inner columns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (coefficient basis : Mxx.Matrix)
    (coefficientLayout :
      Mxx.Toolkit.MatrixLayout coefficient q ringDimension rows inner)
    (basisLayout : Mxx.Toolkit.MatrixLayout basis q ringDimension inner columns) :
    Mxx.Toolkit.matrixValue q ringDimension rows columns
        (Mxx.matrixMultiply
          (Mxx.matrixAdd coefficient (Mxx.matrixNegate coefficient)) basis) = 0 := by
  rw [Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension rows inner columns
      (Mxx.matrixAdd coefficient (Mxx.matrixNegate coefficient)) basis
      (Mxx.Toolkit.matrixAdd_layout coefficient (Mxx.matrixNegate coefficient)
        coefficientLayout (Mxx.Toolkit.matrixNegate_layout coefficient coefficientLayout))
      basisLayout,
    Mxx.Toolkit.matrixValue_add q ringDimension rows inner
      coefficient (Mxx.matrixNegate coefficient)
      ⟨coefficientLayout.modulus, coefficientLayout.ringDimension,
        coefficientLayout.rows, coefficientLayout.columns⟩
      ⟨(Mxx.Toolkit.matrixNegate_layout coefficient coefficientLayout).modulus,
        (Mxx.Toolkit.matrixNegate_layout coefficient coefficientLayout).ringDimension,
        (Mxx.Toolkit.matrixNegate_layout coefficient coefficientLayout).rows,
        (Mxx.Toolkit.matrixNegate_layout coefficient coefficientLayout).columns⟩,
    Mxx.Toolkit.matrixValue_negate q ringDimension rows inner coefficient
      ⟨coefficientLayout.modulus, coefficientLayout.ringDimension,
        coefficientLayout.rows, coefficientLayout.columns⟩]
  simp

end Mxx.Certificate
