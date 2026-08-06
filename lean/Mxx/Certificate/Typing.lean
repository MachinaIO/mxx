import Mxx.Certificate.Facts

namespace Mxx.Certificate

inductive TypingError where
  | unknownExpressionType
  | incompatibleMatrixProduct (left right : MatrixTypeExpr)

private def sameRing (left right : MatrixTypeExpr) : Bool :=
  left.modulus == right.modulus && left.ringDimension == right.ringDimension

private def isOne (value : IntExpr) : Bool := value == .constant 1

/-- Remove only arithmetic neutral elements emitted by the DSL. -/
private def normalizeDimension : IntExpr → IntExpr
  | .add left right =>
      match normalizeDimension left, normalizeDimension right with
      | .constant left, .constant right => .constant (left + right)
      | .constant 0, value | value, .constant 0 => value
      | left, right => .add left right
  | .multiply left right =>
      match normalizeDimension left, normalizeDimension right with
      | .constant left, .constant right => .constant (left * right)
      | .constant 1, value | value, .constant 1 => value
      | left, right => .multiply left right
  | .subtract left right =>
      let left := normalizeDimension left
      let right := normalizeDimension right
      if left == right then .constant 0
      else match left, right with
        | .constant left, .constant right => .constant (left - right)
        | .add first extra, right =>
            if first == right then extra
            else if extra == right then first
            else .subtract left right
        | _, _ => .subtract left right
  | .divide left right => .divide (normalizeDimension left) (normalizeDimension right)
  | .roundDivide left right => .roundDivide (normalizeDimension left) (normalizeDimension right)
  | .log2Ceil value => .log2Ceil (normalizeDimension value)
  | value => value

private def sameDimension (left right : IntExpr) : Bool :=
  normalizeDimension left == normalizeDimension right

private def addDimensions (values : List IntExpr) : IntExpr :=
  values.foldl .add (.constant 0)

structure MatrixProductType where
  output : MatrixTypeExpr
  mode : SignalProductMode

/-- Derive the same multiplication branch and result shape as executable `matrixMultiply`.
The left scalar branch has priority, exactly as in the runtime definition. -/
def inferMatrixProductType
    (left right : MatrixTypeExpr) : Except TypingError MatrixProductType := do
  if !sameRing left right then throw (.incompatibleMatrixProduct left right)
  if isOne left.rows && isOne left.columns then
    if isOne right.rows then
      return {
        output := { left with columns := right.columns }
        mode := .ordinaryMatrixProduct
      }
    return { output := right, mode := .leftPolynomialScalarBroadcast }
  if isOne right.rows && isOne right.columns then
    if isOne left.rows then
      return { output := left, mode := .swappedRowVectorScalarProduct }
    return { output := left, mode := .rightPolynomialScalarBroadcast }
  if !sameDimension left.columns right.rows then throw (.incompatibleMatrixProduct left right)
  return {
    output := { left with columns := right.columns }
    mode := .ordinaryMatrixProduct
  }

private def equivalentMatrixType (left right : MatrixTypeExpr) : Bool :=
  sameRing left right && sameDimension left.rows right.rows &&
    sameDimension left.columns right.columns

private def allSameType : List MatrixTypeExpr → Option MatrixTypeExpr
  | [] => none
  | first :: rest => if rest.all (equivalentMatrixType · first) then some first else none

/-- Conservative, syntax-directed matrix typing. Recurrence results and analyzer-only carried
inputs store the type obtained from their checked schema path. -/
def MatrixExpr.inferType : MatrixExpr → Option MatrixTypeExpr
  | .wire reference => some reference.type
  | .zero type | .identity type | .gadget type _ => some type
  | .add left right => do
      let leftType ← left.inferType
      let rightType ← right.inferType
      if equivalentMatrixType leftType rightType then some leftType else none
  | .negate value | .scalarMultiply _ value => value.inferType
  | .multiply left right => do
      let leftType ← left.inferType
      let rightType ← right.inferType
      (inferMatrixProductType leftType rightType).toOption.map (·.output)
  | .rowSlice value start stop => do
      let type ← value.inferType
      return { type with rows := .subtract stop start }
  | .columnSlice value start stop => do
      let type ← value.inferType
      return { type with columns := .subtract stop start }
  | .rowConcat parts => do
      let types ← parts.mapM MatrixExpr.inferType
      let first ← types.head?
      if types.all (fun type => sameRing type first && type.columns == first.columns) then
        return { first with rows := addDimensions (types.map (·.rows)) }
      none
  | .columnConcat parts => do
      let types ← parts.mapM MatrixExpr.inferType
      let first ← types.head?
      if types.all (fun type => sameRing type first && type.rows == first.rows) then
        return { first with columns := addDimensions (types.map (·.columns)) }
      none
  | .diagonalConcat parts => do
      let types ← parts.mapM MatrixExpr.inferType
      let first ← types.head?
      if types.all (sameRing · first) then
        return {
          first with
          rows := addDimensions (types.map (·.rows))
          columns := addDimensions (types.map (·.columns))
        }
      none
  | .rowCoefficientEmbed layout _ value => do
      let type ← value.inferType
      if sameRing type layout.output then
        return { type with rows := layout.output.rows }
      none
  | .columnBasisEmbed layout _ value => do
      let type ← value.inferType
      if sameRing type layout.output then
        return { type with columns := layout.output.columns }
      none
  | .diagonalCoefficientEmbed layout _ value => do
      let type ← value.inferType
      if sameRing type layout.output then
        return { type with rows := layout.output.rows }
      none
  | .diagonalBasisEmbed layout _ value => do
      let type ← value.inferType
      if sameRing type layout.output then
        return { type with columns := layout.output.columns }
      none
  | .select _ branches => do
      let types ← branches.mapM MatrixExpr.inferType
      allSameType types
  | .loopResult type .. => some type
  | .carriedInput type _ => some type

/-- The only constructor for signal terms. Its mode is derived from the final transformed
coefficient and basis types, never copied from an outer executable node. -/
def mkSignalTerm
    (coefficient : BoundedMatrixExpr)
    (basis : MatrixExpr) : Except TypingError SignalTerm := do
  let coefficientType ← match coefficient.expression.inferType with
    | some type => pure type
    | none => throw .unknownExpressionType
  let basisType ← match basis.inferType with
    | some type => pure type
    | none => throw .unknownExpressionType
  let product ← inferMatrixProductType coefficientType basisType
  return { coefficient, basis, mode := product.mode }

private def scalarType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 1
  columns := .constant 1

private def matrixType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 3
  columns := .constant 5

example : (inferMatrixProductType scalarType matrixType).map (·.mode) =
    .ok .leftPolynomialScalarBroadcast := rfl

example : (inferMatrixProductType matrixType scalarType).map (·.mode) =
    .ok .rightPolynomialScalarBroadcast := rfl

example : MatrixExpr.inferType (.carriedInput matrixType (.exactExpression 0)) =
    some matrixType := by simp [MatrixExpr.inferType]

end Mxx.Certificate
