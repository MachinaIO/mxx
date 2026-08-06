import Mxx.Certificate.Syntax

namespace Mxx.Certificate

structure ConcatPartKey where
  matrixType : MatrixTypeExpr
  rowOffset : IntExpr
  columnOffset : IntExpr
  deriving BEq

structure ConcatLayoutKey where
  axis : ConcatAxis
  parts : List ConcatPartKey
  output : MatrixTypeExpr
  deriving BEq

def ConcatLayout.key (layout : ConcatLayout) : ConcatLayoutKey := {
  axis := layout.axis
  parts := layout.parts.map fun part => {
    matrixType := part.matrixType
    rowOffset := part.rowOffset
    columnOffset := part.columnOffset
  }
  output := layout.output
}

def ConcatPartKey.toPart (key : ConcatPartKey) : ConcatPart := {
  matrixType := key.matrixType
  rowOffset := key.rowOffset
  columnOffset := key.columnOffset
}

def ConcatLayoutKey.toLayout (key : ConcatLayoutKey) : ConcatLayout := {
  axis := key.axis
  parts := key.parts.map ConcatPartKey.toPart
  output := key.output
}

/-- Closed structural identity for matrix expressions. Dynamic matrix selects are excluded: two
select expressions require their own branchwise proof rather than heuristic key equality. -/
inductive MatrixExprKey where
  | wire (value : ValueInstanceRef) (type : MatrixTypeExpr)
  | zero (type : MatrixTypeExpr)
  | identity (type : MatrixTypeExpr)
  | gadget (type : MatrixTypeExpr) (base : IntExpr)
  | add (left right : MatrixExprKey)
  | negate (value : MatrixExprKey)
  | multiply (left right : MatrixExprKey)
  | scalarMultiply (scalar : IntExpr) (value : MatrixExprKey)
  | rowSlice (value : MatrixExprKey) (start stop : IntExpr)
  | rowConcat (parts : List MatrixExprKey)
  | columnSlice (value : MatrixExprKey) (start stop : IntExpr)
  | columnConcat (parts : List MatrixExprKey)
  | diagonalConcat (parts : List MatrixExprKey)
  | rowCoefficientEmbed (layout : ConcatLayoutKey) (part : Nat) (value : MatrixExprKey)
  | columnBasisEmbed (layout : ConcatLayoutKey) (part : Nat) (value : MatrixExprKey)
  | diagonalCoefficientEmbed (layout : ConcatLayoutKey) (part : Nat) (value : MatrixExprKey)
  | diagonalBasisEmbed (layout : ConcatLayoutKey) (part : Nat) (value : MatrixExprKey)
  | loopResult
      (type : MatrixTypeExpr)
      (summary : SequentialRecurrenceInstanceRef)
      (path : MatrixFactPath)
  | carriedInput (type : MatrixTypeExpr) (path : MatrixFactPath)
  deriving BEq

def MatrixExprKey.toExpr : MatrixExprKey → MatrixExpr
  | .wire value type => .wire { value, type }
  | .zero type => .zero type
  | .identity type => .identity type
  | .gadget type base => .gadget type base
  | .add left right => .add left.toExpr right.toExpr
  | .negate value => .negate value.toExpr
  | .multiply left right => .multiply left.toExpr right.toExpr
  | .scalarMultiply scalar value => .scalarMultiply scalar value.toExpr
  | .rowSlice value start stop => .rowSlice value.toExpr start stop
  | .rowConcat parts => .rowConcat (parts.map toExpr)
  | .columnSlice value start stop => .columnSlice value.toExpr start stop
  | .columnConcat parts => .columnConcat (parts.map toExpr)
  | .diagonalConcat parts => .diagonalConcat (parts.map toExpr)
  | .rowCoefficientEmbed layout part value =>
      .rowCoefficientEmbed layout.toLayout part value.toExpr
  | .columnBasisEmbed layout part value => .columnBasisEmbed layout.toLayout part value.toExpr
  | .diagonalCoefficientEmbed layout part value =>
      .diagonalCoefficientEmbed layout.toLayout part value.toExpr
  | .diagonalBasisEmbed layout part value =>
      .diagonalBasisEmbed layout.toLayout part value.toExpr
  | .loopResult type summary path => .loopResult type summary path
  | .carriedInput type path => .carriedInput type path

def MatrixExpr.key : MatrixExpr → Option MatrixExprKey
  | .wire reference => some (.wire reference.value reference.type)
  | .zero type => some (.zero type)
  | .identity type => some (.identity type)
  | .gadget type base => some (.gadget type base)
  | .add left right => return .add (← left.key) (← right.key)
  | .negate value => return .negate (← value.key)
  | .multiply left right => return .multiply (← left.key) (← right.key)
  | .scalarMultiply scalar value => return .scalarMultiply scalar (← value.key)
  | .rowSlice value start stop => return .rowSlice (← value.key) start stop
  | .rowConcat parts => return .rowConcat (← parts.mapM key)
  | .columnSlice value start stop => return .columnSlice (← value.key) start stop
  | .columnConcat parts => return .columnConcat (← parts.mapM key)
  | .diagonalConcat parts => return .diagonalConcat (← parts.mapM key)
  | .rowCoefficientEmbed layout part value =>
      return .rowCoefficientEmbed layout.key part (← value.key)
  | .columnBasisEmbed layout part value =>
      return .columnBasisEmbed layout.key part (← value.key)
  | .diagonalCoefficientEmbed layout part value =>
      return .diagonalCoefficientEmbed layout.key part (← value.key)
  | .diagonalBasisEmbed layout part value =>
      return .diagonalBasisEmbed layout.key part (← value.key)
  | .select .. => none
  | .loopResult type summary path => some (.loopResult type summary path)
  | .carriedInput type path => some (.carriedInput type path)

inductive EqualityAttempt {α : Type} (left right : α) where
  | equal (proof : left = right)
  | unknown

/-- Proof-producing equality for the closed expression fragment used as Diamond signal bases.
Unsupported structural constructors fail closed rather than falling back to hash or textual
equality. -/
def MatrixExpr.sameSupported : (left right : MatrixExpr) → EqualityAttempt left right
  | .wire left, .wire right =>
      if valueEq : left.value = right.value then
        if typeEq : left.type = right.type then
          .equal (by cases left; cases right; cases valueEq; cases typeEq; rfl)
        else .unknown
      else .unknown
  | .zero left, .zero right =>
      if equal : left = right then .equal (equal ▸ rfl) else .unknown
  | .identity left, .identity right =>
      if equal : left = right then .equal (equal ▸ rfl) else .unknown
  | .gadget leftType leftBase, .gadget rightType rightBase =>
      if typeEq : leftType = rightType then
        if baseEq : leftBase = rightBase then .equal (typeEq ▸ baseEq ▸ rfl)
        else .unknown
      else .unknown
  | .add leftA leftB, .add rightA rightB =>
      match leftA.sameSupported rightA, leftB.sameSupported rightB with
      | .equal leftEq, .equal rightEq => .equal (leftEq ▸ rightEq ▸ rfl)
      | _, _ => .unknown
  | .multiply leftA leftB, .multiply rightA rightB =>
      match leftA.sameSupported rightA, leftB.sameSupported rightB with
      | .equal leftEq, .equal rightEq => .equal (leftEq ▸ rightEq ▸ rfl)
      | _, _ => .unknown
  | .negate left, .negate right =>
      match left.sameSupported right with
      | .equal equal => .equal (equal ▸ rfl)
      | .unknown => .unknown
  | .scalarMultiply leftScalar left, .scalarMultiply rightScalar right =>
      if scalarEq : leftScalar = rightScalar then
        match left.sameSupported right with
        | .equal valueEq => .equal (scalarEq ▸ valueEq ▸ rfl)
        | .unknown => .unknown
      else .unknown
  | .loopResult leftType leftSummary leftPath,
      .loopResult rightType rightSummary rightPath =>
      if typeEq : leftType = rightType then
        if summaryEq : leftSummary = rightSummary then
          if pathEq : leftPath = rightPath then
            .equal (typeEq ▸ summaryEq ▸ pathEq ▸ rfl)
          else .unknown
        else .unknown
      else .unknown
  | .carriedInput leftType leftPath, .carriedInput rightType rightPath =>
      if typeEq : leftType = rightType then
        if pathEq : leftPath = rightPath then .equal (typeEq ▸ pathEq ▸ rfl)
        else .unknown
      else .unknown
  | _, _ => .unknown

/-- Proof-producing recognition of a zero with one exact matrix type. -/
def MatrixExpr.sameTypedZero (type : MatrixTypeExpr) :
    (expression : MatrixExpr) → EqualityAttempt expression (.zero type)
  | .zero actual =>
      if equal : actual = type then .equal (equal ▸ rfl) else .unknown
  | _ => .unknown

/-- Algebraic rewrite evidence produced by the matrix-expression normalizer.
It is deliberately independent of an executable matrix denotation. -/
inductive MatrixRewrite : MatrixExpr → MatrixExpr → Prop where
  | refl (expression : MatrixExpr) : MatrixRewrite expression expression
  | trans {left middle right : MatrixExpr} :
      MatrixRewrite left middle → MatrixRewrite middle right → MatrixRewrite left right
  | addCongr :
      {left left' right right' : MatrixExpr} →
      MatrixRewrite left left' →
      MatrixRewrite right right' →
      MatrixRewrite (.add left right) (.add left' right')
  | multiplyCongr :
      {left left' right right' : MatrixExpr} →
      MatrixRewrite left left' →
      MatrixRewrite right right' →
      MatrixRewrite (.multiply left right) (.multiply left' right')
  | scalarMultiplyCongr (scalar : IntExpr) :
      {value value' : MatrixExpr} →
      MatrixRewrite value value' →
      MatrixRewrite (.scalarMultiply scalar value) (.scalarMultiply scalar value')
  | negateCongr {value value' : MatrixExpr} :
      MatrixRewrite value value' → MatrixRewrite (.negate value) (.negate value')
  | addZeroLeft (type : MatrixTypeExpr) (right : MatrixExpr) :
      MatrixRewrite (.add (.zero type) right) right
  | addZeroRight (left : MatrixExpr) (type : MatrixTypeExpr) :
      MatrixRewrite (.add left (.zero type)) left
  | negateZero (type : MatrixTypeExpr) : MatrixRewrite (.negate (.zero type)) (.zero type)
  | doubleNegate (value : MatrixExpr) : MatrixRewrite (.negate (.negate value)) value
  | addNegateRight (type : MatrixTypeExpr) (value : MatrixExpr) :
      MatrixRewrite (.add value (.negate value)) (.zero type)
  | addNegateLeft (type : MatrixTypeExpr) (value : MatrixExpr) :
      MatrixRewrite (.add (.negate value) value) (.zero type)

private def normalizeAddWithProof
    (left right : MatrixExpr) :
    { normalized : MatrixExpr // MatrixRewrite (.add left right) normalized } :=
  match left, right with
  | .zero type, right => ⟨right, .addZeroLeft type right⟩
  | left, .zero type => ⟨left, .addZeroRight left type⟩
  | left, right => ⟨.add left right, .refl _⟩

private def normalizeNegateWithProof
    (value : MatrixExpr) :
    { normalized : MatrixExpr // MatrixRewrite (.negate value) normalized } :=
  match value with
  | .zero type => ⟨.zero type, .negateZero type⟩
  | .negate inner => ⟨inner, .doubleNegate inner⟩
  | value => ⟨.negate value, .refl _⟩

/-- Normalize the universally valid additive rewrites currently representable by `MatrixExpr`.
Identity multiplication is left intact because its dimensional side condition belongs to the
typed analyzer, rather than this syntax-only normalizer. -/
def normalizeMatrixExprWithProof :
    (expression : MatrixExpr) → { normalized : MatrixExpr // MatrixRewrite expression normalized }
  | .add left right =>
      let ⟨left', leftProof⟩ := normalizeMatrixExprWithProof left
      let ⟨right', rightProof⟩ := normalizeMatrixExprWithProof right
      let childrenProof : MatrixRewrite (.add left right) (.add left' right') :=
        .addCongr leftProof rightProof
      let ⟨result, resultProof⟩ := normalizeAddWithProof left' right'
      ⟨result, .trans childrenProof resultProof⟩
  | .negate value =>
      let ⟨value', valueProof⟩ := normalizeMatrixExprWithProof value
      let childProof : MatrixRewrite (.negate value) (.negate value') :=
        .negateCongr valueProof
      let ⟨result, resultProof⟩ := normalizeNegateWithProof value'
      ⟨result, .trans childProof resultProof⟩
  | .multiply left right =>
      let ⟨left', leftProof⟩ := normalizeMatrixExprWithProof left
      let ⟨right', rightProof⟩ := normalizeMatrixExprWithProof right
      ⟨.multiply left' right', .multiplyCongr leftProof rightProof⟩
  | .scalarMultiply scalar value =>
      let ⟨value', valueProof⟩ := normalizeMatrixExprWithProof value
      ⟨.scalarMultiply scalar value', .scalarMultiplyCongr scalar valueProof⟩
  | expression => ⟨expression, .refl expression⟩

def MatrixExpr.normalize (expression : MatrixExpr) : MatrixExpr :=
  (normalizeMatrixExprWithProof expression).val

theorem MatrixExpr.normalize_preserves (expression : MatrixExpr) :
    MatrixRewrite expression expression.normalize :=
  (normalizeMatrixExprWithProof expression).property

/-- Normalize a coefficient and eliminate a syntactically proved `x + (-x)` pair. The caller
supplies the coefficient type used by the zero expression; semantic soundness separately checks
that type against the coefficient denotation. -/
private def eliminateCoefficientCancellationWithProof
    (type : MatrixTypeExpr)
    (expression : MatrixExpr) :
    { normalized : MatrixExpr // MatrixRewrite expression normalized } :=
  match expression with
  | .add left (.negate right) =>
      match left.sameSupported right with
      | .equal equal =>
          equal ▸ ⟨.zero type, .addNegateRight type left⟩
      | .unknown => ⟨.add left (.negate right), .refl _⟩
  | .add (.negate left) right =>
      match left.sameSupported right with
      | .equal equal => equal ▸ ⟨.zero type, .addNegateLeft type left⟩
      | .unknown => ⟨.add (.negate left) right, .refl _⟩
  | expression => ⟨expression, .refl expression⟩

def normalizeCoefficientWithProof
    (type : MatrixTypeExpr)
    (expression : MatrixExpr) :
    { normalized : MatrixExpr // MatrixRewrite expression normalized } :=
  let ⟨first, firstProof⟩ := normalizeMatrixExprWithProof expression
  let ⟨result, resultProof⟩ := eliminateCoefficientCancellationWithProof type first
  ⟨result, .trans firstProof resultProof⟩

private def testType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 1
  columns := .constant 1

private def testWire : MatrixExpr :=
  .wire {
    value := .protocolInput ⟨"normalization-test"⟩
    type := testType
  }

example : MatrixExpr.normalize (.add (.zero testType) testWire) = testWire := rfl

example : MatrixExpr.normalize (.negate (.negate testWire)) = testWire := rfl

example :
    MatrixRewrite (.add (.zero testType) (.negate (.negate testWire))) testWire :=
  MatrixExpr.normalize_preserves _

end Mxx.Certificate
