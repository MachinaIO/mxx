import Mxx.Certificate.ExpressionArena
import Mxx.Certificate.Facts

namespace Mxx.Certificate

/-- The analyzer-owned context needed to compare matrix provenance relative to one sequential
loop. It is created only while descending through that frozen loop; it is not protocol or
certificate input. -/
structure MatrixOriginNormalizationContext where
  arena : ExpressionArena
  selectedLoop : LoopRef
  indexSlot : Nat

/-- Closed failures while normalizing recurrence provenance. These failures decline an aligned
rewrite; they never create an assumption or an approximate identity. -/
inductive RecurrenceBasisAlignmentError where
  | missingExpression (id : Nat)
  | cyclicArenaReference (id : Nat)
  | wrongLoopBinder (expected actual : LoopRef)
  | wrongLoopIndexSlot (expected actual : Nat)
  | unsupportedIndex
  | negativeOffset (value : Int)
  | unsupportedArithmetic
  | unsupportedOrigin
  | missingAlias
  | ambiguousAlias
  | aliasTypeMismatch
  | escapedAnalyzerPlaceholder
  | wrongInputTermCount
  | rightSubjectMismatch
  | noPreimageRelation
  | ambiguousPreimageRelation
  | sourceTypeMismatch
  | targetTypeMismatch
  | targetNotOneTermAffine
  | successorOriginMismatch
  | missingSuccessorBasisFact
  | ambiguousSuccessorBasisFact
  | nonUniformSuccessorBasisBound
  | schemaMismatch
  | rangeObligationUnavailable
  deriving BEq, DecidableEq, Repr

private def checkedOffset (value : Int) : Except RecurrenceBasisAlignmentError Nat :=
  if 0 ≤ value then .ok value.toNat else .error (.negativeOffset value)

private def normalizeRelativeIntExpr
    (selectedSlot : Nat) : IntExpr → Except RecurrenceBasisAlignmentError (Option Nat)
  | .loopIndex slot =>
      if slot = selectedSlot then .ok (some 0) else .error (.wrongLoopIndexSlot selectedSlot slot)
  | .constant _ | .parameter _ => .ok none
  | .add left right => do
      let leftOffset ← normalizeRelativeIntExpr selectedSlot left
      let rightOffset ← normalizeRelativeIntExpr selectedSlot right
      match leftOffset, rightOffset with
      | none, none => .ok none
      | some _, some _ => .error .unsupportedArithmetic
      | some offset, none =>
          match right with
          | .constant value => return some (offset + (← checkedOffset value))
          | _ => throw .unsupportedArithmetic
      | none, some offset =>
          match left with
          | .constant value => return some (offset + (← checkedOffset value))
          | _ => throw .unsupportedArithmetic
  | .subtract left right | .multiply left right | .divide left right | .roundDivide left right => do
      let leftOffset ← normalizeRelativeIntExpr selectedSlot left
      let rightOffset ← normalizeRelativeIntExpr selectedSlot right
      if leftOffset.isSome || rightOffset.isSome then throw .unsupportedArithmetic
      return none
  | .log2Ceil value => do
      if (← normalizeRelativeIntExpr selectedSlot value).isSome then throw .unsupportedArithmetic
      return none

/-- Normalize only the initial closed grammar: a selected loop index, optionally plus a
nonnegative integer literal, or an expression which contains no loop index. -/
private def normalizeRelativeExpression
    (selected : LoopRef)
    (selectedSlot : Nat) : RuntimeExpr .integer → Except RecurrenceBasisAlignmentError (Option Nat)
  | .loopIndex loop =>
      if loop = selected then .ok (some 0) else .error (.wrongLoopBinder selected loop)
  | .intConstant _ | .intWire _ => .ok none
  | .parameter expression => normalizeRelativeIntExpr selectedSlot expression
  | .intBinary .add left right => do
      let leftOffset ← normalizeRelativeExpression selected selectedSlot left
      let rightOffset ← normalizeRelativeExpression selected selectedSlot right
      match leftOffset, rightOffset with
      | none, none => .ok none
      | some _, some _ => .error .unsupportedArithmetic
      | some offset, none =>
          match right with
          | .intConstant value => return some (offset + (← checkedOffset value))
          | _ => throw .unsupportedArithmetic
      | none, some offset =>
          match left with
          | .intConstant value => return some (offset + (← checkedOffset value))
          | _ => throw .unsupportedArithmetic
  | .intBinary _ left right => do
      let leftOffset ← normalizeRelativeExpression selected selectedSlot left
      let rightOffset ← normalizeRelativeExpression selected selectedSlot right
      if leftOffset.isSome || rightOffset.isSome then throw .unsupportedArithmetic
      return none
  | .boolToInt _ | .extractCoefficient .. | .familyElement .. | .select .. =>
      .error .unsupportedArithmetic
  | .carriedInput _ => .error .escapedAnalyzerPlaceholder

/-- Resolve the arena-owned expression exactly once. The arena is immutable and each entry only
references older entries, so this normalizer does not introduce a second integer evaluator. -/
def normalizeLoopRelativeIntExpr
    (arena : ExpressionArena)
    (selected : LoopRef)
    (selectedSlot : Nat)
    (reference : RuntimeExprRef .integer) :
    Except RecurrenceBasisAlignmentError RelativeIndexNF := do
  let expression ← match arena.lookupInteger reference with
    | some expression => pure expression
    | none => throw (.missingExpression reference.id)
  match ← normalizeRelativeExpression selected selectedSlot expression with
  | some offset => pure (.loopOffset offset)
  | none => pure (.invariant reference)

/-- Normalize a frozen execution path relative to one selected sequential loop. -/
def normalizeInstanceFrame
    (arena : ExpressionArena)
    (selected : LoopRef)
    (selectedSlot : Nat) : InstanceFrame → Except RecurrenceBasisAlignmentError RelativeInstanceFrame
  | .subgraphCall site => pure (.subgraphCall site)
  | .parallelLane site index => do
      let normalized ← normalizeLoopRelativeIntExpr arena selected selectedSlot index
      pure (.parallelLane site normalized)
  | .sequentialIteration site index => do
      if site = selected.site then
        match ← normalizeLoopRelativeIntExpr arena selected selectedSlot index with
        | .loopOffset 0 => pure (.selectedSequentialIteration site)
        | .loopOffset _ => throw .unsupportedArithmetic
        | .invariant _ => throw (.wrongLoopBinder selected { site })
      else
        match ← normalizeLoopRelativeIntExpr arena selected selectedSlot index with
        | .invariant invariant => pure (.invariantSequentialIteration site invariant)
        | .loopOffset _ => throw (.wrongLoopBinder selected { site })

def normalizeInstancePath
    (arena : ExpressionArena)
    (selected : LoopRef)
    (selectedSlot : Nat)
    (path : InstancePathExpr) : Except RecurrenceBasisAlignmentError (List RelativeInstanceFrame) :=
  path.mapM (normalizeInstanceFrame arena selected selectedSlot)

def normalizeAggregateOrigin
    (arena : ExpressionArena)
    (selected : LoopRef)
    (selectedSlot : Nat) : FamilyAggregateRef →
      Except RecurrenceBasisAlignmentError NormalizedAggregateOrigin
  | .joint family outputSlot path =>
      return .joint family outputSlot (← normalizeInstancePath arena selected selectedSlot path)
  | .familyElement parent index =>
      return .familyElement (← normalizeAggregateOrigin arena selected selectedSlot parent)
        (← normalizeLoopRelativeIntExpr arena selected selectedSlot index)
  | .carriedInput _ | .recurrenceResult .. => throw .unsupportedOrigin

def normalizeValueOrigin
    (arena : ExpressionArena)
    (selected : LoopRef)
    (selectedSlot : Nat)
    (type : MatrixTypeExpr) : ValueInstanceRef →
      Except RecurrenceBasisAlignmentError IndexedMatrixOrigin
  | .concrete wire => pure (.invariant { value := .concrete wire, type })
  | .template wire => pure (.instantiatedTemplate wire [] type)
  | .instantiatedTemplate wire path =>
      return .instantiatedTemplate wire (← normalizeInstancePath arena selected selectedSlot path) type
  | .familyElement aggregate index =>
      return .familyElement (← normalizeAggregateOrigin arena selected selectedSlot aggregate)
        (← normalizeLoopRelativeIntExpr arena selected selectedSlot index) type
  | .protocolInput _ | .recurrenceResult .. => throw .unsupportedOrigin

def normalizeMatrixOrigin
    (arena : ExpressionArena)
    (selected : LoopRef)
    (selectedSlot : Nat)
    (expression : MatrixExpr) : Except RecurrenceBasisAlignmentError IndexedMatrixOrigin :=
  match expression with
  | .wire reference => normalizeValueOrigin arena selected selectedSlot reference.type reference.value
  | .scalarMultiply (.constant 1) value => normalizeMatrixOrigin arena selected selectedSlot value
  | .carriedInput .. | .loopResult .. => throw .escapedAnalyzerPlaceholder
  | _ => throw .unsupportedOrigin

def shiftRelativeIndex : RelativeIndexNF → RelativeIndexNF
  | .invariant expression => .invariant expression
  | .loopOffset offset => .loopOffset (offset + 1)

def shiftIndexedMatrixOrigin : IndexedMatrixOrigin → IndexedMatrixOrigin
  | .invariant matrix => .invariant matrix
  | .instantiatedTemplate wire path type => .instantiatedTemplate wire path type
  | .familyElement aggregate index type => .familyElement aggregate (shiftRelativeIndex index) type

def sameInitialOrigin (left right : IndexedMatrixOrigin) : Bool := left == right

def sameSuccessorOrigin (current successor : IndexedMatrixOrigin) : Bool :=
  shiftIndexedMatrixOrigin current == successor

private def normalizerTestSite (node : Nat) : CoreNodeRef := {
  stage := ⟨"recurrence-normalizer"⟩
  scope := ⟨[]⟩
  node := ⟨node⟩
}

private def normalizerTestLoop : LoopRef := ⟨normalizerTestSite 0⟩

private def normalizerTestArena : ExpressionArena := {
  entries := #[
    .integer (.loopIndex normalizerTestLoop),
    .integer (.intBinary .add (.loopIndex normalizerTestLoop) (.intConstant 1)),
    .integer (.intBinary .add (.intConstant 1) (.loopIndex normalizerTestLoop)),
    .integer (.intBinary .add
      (.intBinary .add (.loopIndex normalizerTestLoop) (.intConstant 1)) (.intConstant 2)),
    .integer (.parameter (.constant 7))
  ]
}

example : normalizeLoopRelativeIntExpr normalizerTestArena normalizerTestLoop 0 ⟨0⟩ =
    .ok (.loopOffset 0) := by
  simp [normalizeLoopRelativeIntExpr, normalizeRelativeExpression, normalizerTestArena,
    normalizerTestLoop, normalizerTestSite, ExpressionArena.lookupInteger]
  rfl

example : normalizeLoopRelativeIntExpr normalizerTestArena normalizerTestLoop 0 ⟨1⟩ =
    .ok (.loopOffset 1) := by
  simp [normalizeLoopRelativeIntExpr, normalizeRelativeExpression, normalizerTestArena,
    normalizerTestLoop, normalizerTestSite, ExpressionArena.lookupInteger, checkedOffset]
  rfl

example : normalizeLoopRelativeIntExpr normalizerTestArena normalizerTestLoop 0 ⟨2⟩ =
    .ok (.loopOffset 1) := by
  simp [normalizeLoopRelativeIntExpr, normalizeRelativeExpression, normalizerTestArena,
    normalizerTestLoop, normalizerTestSite, ExpressionArena.lookupInteger, checkedOffset]
  rfl

example : normalizeLoopRelativeIntExpr normalizerTestArena normalizerTestLoop 0 ⟨3⟩ =
    .ok (.loopOffset 3) := by
  simp [normalizeLoopRelativeIntExpr, normalizeRelativeExpression, normalizerTestArena,
    normalizerTestLoop, normalizerTestSite, ExpressionArena.lookupInteger, checkedOffset]
  rfl

example : normalizeLoopRelativeIntExpr normalizerTestArena normalizerTestLoop 0 ⟨4⟩ =
    .ok (.invariant ⟨4⟩) := by
  simp [normalizeLoopRelativeIntExpr, normalizeRelativeExpression, normalizerTestArena,
    normalizerTestLoop, normalizerTestSite, ExpressionArena.lookupInteger]
  rfl

end Mxx.Certificate
