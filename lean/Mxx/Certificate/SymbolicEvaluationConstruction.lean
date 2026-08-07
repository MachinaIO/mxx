import Mxx.Certificate.SymbolicForm
import Mxx.Certificate.Workflow

namespace Mxx.Certificate

/-! # Analyzer-owned symbolic evaluation construction

This module converts an already normalized matrix wire fact into the immutable expression,
symbolic-form, and bound-witness arenas.  It accepts no certificate-provided symbolic metadata:
all expressions, signal terms, and bounds are copied from the analyzer's existing `MatrixFact`.
-/

inductive SymbolicEvaluationConstructionError where
  | nonMatrixFact
  | missingMatrixType
  | subjectMismatch
  | expressionRejected
  | symbolicFormRejected
  | boundWitnessRejected
  deriving BEq, DecidableEq, Repr

/-- Append-only state owned by the analyzer while symbolic facts are constructed. -/
structure SymbolicEvaluationConstructionState where
  expressionArena : ExpressionArena := {}
  symbolicFormArena : SymbolicMatrixFormArena := {}
  boundWitnessArena : BoundWitnessArena := {}
  /-- The final symbolic table is built by a single traversal.  An array keeps that construction
  linear while the public analysis result retains its proof-friendly list representation. -/
  symbolicMatrixFacts : Array MatrixSymbolicFact := #[]

structure SymbolicEvaluationConstructionState.Extends
    (older newer : SymbolicEvaluationConstructionState) : Prop where
  expressions : older.expressionArena.Extends newer.expressionArena
  forms : older.symbolicFormArena.Extends newer.symbolicFormArena
  witnesses : older.boundWitnessArena.Extends newer.boundWitnessArena

theorem SymbolicEvaluationConstructionState.Extends.refl
    (state : SymbolicEvaluationConstructionState) : state.Extends state :=
  ⟨⟨#[], by simp⟩, ⟨#[], by simp⟩, ⟨#[], by simp⟩⟩

theorem SymbolicEvaluationConstructionState.Extends.trans
    {first second third : SymbolicEvaluationConstructionState}
    (left : first.Extends second)
    (right : second.Extends third) : first.Extends third := by
  rcases left with ⟨⟨expressionLeft, expressionLeftEq⟩, ⟨formLeft, formLeftEq⟩,
    ⟨witnessLeft, witnessLeftEq⟩⟩
  rcases right with ⟨⟨expressionRight, expressionRightEq⟩, ⟨formRight, formRightEq⟩,
    ⟨witnessRight, witnessRightEq⟩⟩
  refine ⟨⟨expressionLeft ++ expressionRight, ?_⟩,
    ⟨formLeft ++ formRight, ?_⟩, ⟨witnessLeft ++ witnessRight, ?_⟩⟩
  · simp [expressionRightEq, expressionLeftEq, Array.append_assoc]
  · simp [formRightEq, formLeftEq, Array.append_assoc]
  · simp [witnessRightEq, witnessLeftEq, Array.append_assoc]

private def centeredRepresentativeBound (matrixType : MatrixTypeExpr) : BoundExpr :=
  .floorDivide (.absolute matrixType.modulus) 2

private def canonicalExactBound
    (matrixType : MatrixTypeExpr) : BoundExpr → Option BoundExpr
  | bound@(.floorDivide (.absolute modulus) 2) =>
      if modulus = matrixType.modulus then some bound else none
  | _ => none

private structure SymbolicEvaluationConstructionPlan where
  form : SymbolicMatrixForm
  summary : MatrixBoundSummary
  coefficientBound : BoundExpr
  noiseBound : BoundExpr
  totalBound : BoundExpr

private def planPrimary
    (matrixType : MatrixTypeExpr)
    (exactValue : MatrixExprRef)
    (primary : MatrixPrimaryForm)
    (totalNormBound : BoundExpr) :
    Except SymbolicEvaluationConstructionError SymbolicEvaluationConstructionPlan := do
  match primary with
  | .exact _ =>
      match canonicalExactBound matrixType totalNormBound with
      | some totalBound =>
          return {
            form := .signalAtom exactValue
            summary := .exactLarge totalBound
            coefficientBound := .constant 1
            noiseBound := .constant 0
            totalBound
          }
      | none =>
          return {
            form := .boundedAtom exactValue totalNormBound
            summary := .bounded totalNormBound
            coefficientBound := .constant 0
            noiseBound := totalNormBound
            totalBound := totalNormBound
          }
  | .affine affine =>
      let signal := if affine.terms.isEmpty then SignalPresence.none else .present
      let coefficient := affine.coefficientL1Bound
      return {
        form := .boundedAffineLeaf affine totalNormBound
        summary := {
          signal
          coefficientL1Bound := coefficient
          noiseBound := affine.noiseBound
          totalBound := totalNormBound
        }
        coefficientBound := coefficient
        noiseBound := affine.noiseBound
        totalBound := totalNormBound
      }

private def symbolicFormContext
    (expressionArena : ExpressionArena) : SymbolicFormWFContext where
  expressionArena
  preimageRelationCount := 0
  gadgetRelationCount := 0
  carriedArity := 0
  recurrences := []

private def boundWitnessContext
    (expressionArena : ExpressionArena) : BoundWitnessWFContext where
  runtimeExpressionCount := expressionArena.entries.size
  preimageRelationCount := 0
  gadgetRelationCount := 0
  recurrences := []

private def internExpression
    (state : SymbolicEvaluationConstructionState)
    (expression : MatrixExpr) :
    Except SymbolicEvaluationConstructionError
      (SymbolicEvaluationConstructionState × MatrixExprRef) := do
  let ⟨arena, reference⟩ ← match state.expressionArena.internMatrix expression with
    | some result => pure result
    | none => throw .expressionRejected
  return ({ state with expressionArena := arena }, reference)

private def internForm
    (state : SymbolicEvaluationConstructionState)
    (matrixType : MatrixTypeExpr)
    (form : SymbolicMatrixForm) :
    Except SymbolicEvaluationConstructionError
      (SymbolicEvaluationConstructionState × SymbolicMatrixFormRef) := do
  let context := symbolicFormContext state.expressionArena
  let ⟨arena, reference⟩ ← match state.symbolicFormArena.intern context { matrixType, form } with
    | some result => pure result
    | none => throw .symbolicFormRejected
  return ({ state with symbolicFormArena := arena }, reference)

private def internBoundWitness
    (state : SymbolicEvaluationConstructionState)
    (role : BoundRole)
    (bound : BoundExpr) :
    Except SymbolicEvaluationConstructionError
      (SymbolicEvaluationConstructionState × BoundWitnessRef) := do
  let context := boundWitnessContext state.expressionArena
  let entry : BoundWitnessEntry := { role, witness := .atom role bound }
  let ⟨arena, reference⟩ ← match state.boundWitnessArena.intern context entry with
    | some result => pure result
    | none => throw .boundWitnessRejected
  return ({ state with boundWitnessArena := arena }, reference)

private def appendBoundWitnesses
    (state : SymbolicEvaluationConstructionState)
    (coefficient noise total : BoundExpr) :
    Except SymbolicEvaluationConstructionError
      (SymbolicEvaluationConstructionState × MatrixBoundWitnessRefs) := do
  let ⟨state, coefficientL1⟩ ← internBoundWitness state .coefficient coefficient
  let ⟨state, noise⟩ ← internBoundWitness state .noise noise
  let ⟨state, total⟩ ← internBoundWitness state .total total
  return (state, { coefficientL1, noise, total })

/-- Convert one normalized matrix wire fact. Scalar/family facts and missing types fail closed.
Existing checked relations are preserved verbatim; this constructor does not synthesize a
relation rewrite or relation-local numeric witness. -/
def SymbolicEvaluationConstructionState.appendScopedMatrixFact
    (state : SymbolicEvaluationConstructionState)
    (scopedFact : ScopedWireFact) :
    Except SymbolicEvaluationConstructionError SymbolicEvaluationConstructionState := do
  let matrixType ← match scopedFact.matrixType with
    | some matrixType => pure matrixType
    | none => throw .missingMatrixType
  let fact ← match scopedFact.fact with
    | .matrix fact => pure fact
    | _ => throw .nonMatrixFact
  if fact.subject ≠ .ofCoreWire scopedFact.wire then throw .subjectMismatch

  let exactExpression := match fact.primary with
    | .exact expression => expression
    | .affine _ => .wire { value := fact.subject, type := matrixType }
  let ⟨state, exactValue⟩ ← internExpression state exactExpression

  let plan ← planPrimary matrixType exactValue fact.primary fact.totalNormBound
  let ⟨state, decomposition⟩ ← internForm state matrixType plan.form
  let ⟨state, boundWitnesses⟩ ←
    appendBoundWitnesses state plan.coefficientBound plan.noiseBound plan.totalBound
  let symbolicFact : MatrixSymbolicFact := {
    subject := fact.subject
    matrixType
    exactValue
    decomposition
    bounds := plan.summary
    boundWitnesses
    relations := fact.relations
    coefficientRepresentation := fact.coefficientRepresentation
  }
  return { state with symbolicMatrixFacts := state.symbolicMatrixFacts.push symbolicFact }

/-- Construct symbolic evaluations for the final analyzer fact table in table order.  Scalar,
Boolean, byte, and family facts deliberately have no matrix symbolic evaluation and are skipped.
Every matrix fact is passed to `appendScopedMatrixFact` exactly once, so any malformed matrix fact
fails the whole construction instead of disappearing from the symbolic result. -/
def SymbolicEvaluationConstructionState.appendMatrixFacts
    (state : SymbolicEvaluationConstructionState)
    (facts : ScopedWireFactTable) :
    Except SymbolicEvaluationConstructionError SymbolicEvaluationConstructionState :=
  match facts with
  | [] => .ok state
  | fact :: tail =>
      match fact.fact with
      | .matrix _ => do
          let state ← state.appendScopedMatrixFact fact
          state.appendMatrixFacts tail
      | .trapdoor _ | .integer _ | .boolean _ | .bytes _ | .family _ =>
          state.appendMatrixFacts tail

/-- Run the authoritative final-table pass without retaining symbolic facts from an earlier or
partial pass. Arena entries remain append-only and interned, but output facts are reconstructed
exactly once from `facts`. -/
def SymbolicEvaluationConstructionState.rebuildMatrixFacts
    (state : SymbolicEvaluationConstructionState)
    (facts : ScopedWireFactTable) :
    Except SymbolicEvaluationConstructionError SymbolicEvaluationConstructionState :=
  { state with symbolicMatrixFacts := #[] }.appendMatrixFacts facts

namespace SymbolicEvaluationConstructionFixtures

private def matrixType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 2
  columns := .constant 2

private def stage : StageId := ⟨"symbolic-construction"⟩
private def wire (node : Nat) : CoreWireRef := {
  stage
  scope := ⟨[]⟩
  node := ⟨node⟩
  port := 0
}

private def scopedFact (node : Nat) (primary : MatrixPrimaryForm) (total : BoundExpr) :
    ScopedWireFact := {
  wire := wire node
  matrixType := some matrixType
  fact := .matrix {
    subject := .ofCoreWire (wire node)
    primary
    relations := []
    totalNormBound := total
  }
}

private def centered : BoundExpr := .floorDivide (.absolute matrixType.modulus) 2

/-- Fixture: exact values become one signal atom with the canonical independent bounds. -/
example : planPrimary matrixType ⟨7⟩
    (.exact (.wire { value := .ofCoreWire (wire 0), type := matrixType })) centered = .ok {
      form := .signalAtom ⟨7⟩
      summary := .exactLarge centered
      coefficientBound := .constant 1
      noiseBound := .constant 0
      totalBound := centered
    } := by
  rfl

/-- Fixture: an exact zero remains a bounded atom instead of becoming a Large signal. -/
example : planPrimary matrixType ⟨7⟩ (.exact (.zero matrixType)) (.constant 0) = .ok {
    form := .boundedAtom ⟨7⟩ (.constant 0)
    summary := .bounded (.constant 0)
    coefficientBound := .constant 0
    noiseBound := .constant 0
    totalBound := .constant 0
  } := by
  rfl

/-- Fixture: an exact identity uses its proved unit bound rather than the centered-radius cap. -/
example : planPrimary matrixType ⟨7⟩ (.exact (.identity matrixType)) (.constant 1) = .ok {
    form := .boundedAtom ⟨7⟩ (.constant 1)
    summary := .bounded (.constant 1)
    coefficientBound := .constant 0
    noiseBound := .constant 1
    totalBound := .constant 1
  } := by
  rfl

/-- Fixture: bounded-only affine input remains signal-free and keeps its actual noise formula. -/
example :
    (({} : SymbolicEvaluationConstructionState).appendScopedMatrixFact
      (scopedFact 1 (.affine { terms := [], noiseBound := .constant 3 }) (.constant 3))
    ).map (fun state => state.symbolicMatrixFacts.toList.head?.map (fun fact => fact.bounds.signal)) =
      .ok (some .none) := by
  rfl

private def scalarFact : ScopedWireFact := {
  wire := wire 8
  matrixType := none
  fact := .integer {
    expression := .intConstant 4
    lower := .integer (.constant 4)
    upper := .integer (.constant 4)
  }
}

/-- Fixture: rebuilding after an earlier partial construction does not duplicate its matrix fact. -/
example :
    let left := scopedFact 6 (.affine { terms := [], noiseBound := .constant 2 }) (.constant 2)
    let right := scopedFact 7 (.affine { terms := [], noiseBound := .constant 3 }) (.constant 3)
    (Except.bind (({} : SymbolicEvaluationConstructionState).appendScopedMatrixFact left)
      (fun previous => previous.rebuildMatrixFacts [left, scalarFact, right])).map
        (fun state => state.symbolicMatrixFacts.toList.map (fun fact => fact.subject)) =
      .ok [.ofCoreWire (wire 6), .ofCoreWire (wire 7)] := by
  rfl

/-- Fixture: the final table pass skips a scalar between two matrices, constructs each matrix
once, and preserves matrix-table order.  The two forms and six role-specific witnesses also rule
out accidental duplicate construction. -/
example :
    let left := scopedFact 6 (.affine { terms := [], noiseBound := .constant 2 }) (.constant 2)
    let right := scopedFact 7 (.affine { terms := [], noiseBound := .constant 3 }) (.constant 3)
    (({} : SymbolicEvaluationConstructionState).appendMatrixFacts [left, scalarFact, right]).map
      (fun state =>
        (state.symbolicMatrixFacts.toList.map (fun fact => fact.subject),
          state.expressionArena.entries.size,
          state.symbolicFormArena.entries.size,
          state.boundWitnessArena.entries.size)) =
      .ok ([.ofCoreWire (wire 6), .ofCoreWire (wire 7)], 2, 2, 6) := by
  rfl

private def firstTerm : SignalTerm := {
  coefficient := {
    expression := .identity matrixType
    normBound := .constant 2
  }
  basis := .wire { value := .ofCoreWire (wire 2), type := matrixType }
  mode := .ordinaryMatrixProduct
}

private def secondTerm : SignalTerm := {
  coefficient := {
    expression := .identity matrixType
    normBound := .constant 5
  }
  basis := .wire { value := .ofCoreWire (wire 3), type := matrixType }
  mode := .ordinaryMatrixProduct
}

/-- Fixture: affine signal terms remain ordered and their coefficient bounds are summed. -/
example : planPrimary matrixType ⟨7⟩ (.affine {
    terms := [firstTerm, secondTerm]
    noiseBound := .constant 3
  }) (.constant 8) = .ok {
    form := .boundedAffineLeaf
      { terms := [firstTerm, secondTerm], noiseBound := .constant 3 } (.constant 8)
    summary := {
      signal := .present
      coefficientL1Bound := .add (.constant 2) (.constant 5)
      noiseBound := .constant 3
      totalBound := .constant 8
    }
    coefficientBound := .add (.constant 2) (.constant 5)
    noiseBound := .constant 3
    totalBound := .constant 8
  } := by
  rfl

/-- Fixture: checked relations are preserved without manufacturing relation rewrite witnesses. -/
example :
    let relation := MatrixRelation.gadgetDecomposition
      (.ofCoreWire (wire 4)) { value := .ofCoreWire (wire 4), type := matrixType }
      (.constant 2) (.constant 3)
    (({} : SymbolicEvaluationConstructionState).appendScopedMatrixFact {
      wire := wire 4
      matrixType := some matrixType
      fact := .matrix {
        subject := .ofCoreWire (wire 4)
        primary := .affine { terms := [], noiseBound := .constant 3 }
        relations := [relation]
        totalNormBound := .constant 3
      }
    }).map (fun state => state.symbolicMatrixFacts.toList.head?.map (·.relations)) =
      .ok (some [relation]) := by
  rfl

end SymbolicEvaluationConstructionFixtures

end Mxx.Certificate
