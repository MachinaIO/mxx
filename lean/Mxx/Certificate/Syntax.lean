import Mxx.Certificate.Identity

namespace Mxx.Certificate

inductive MatrixFactPath where
  | exactExpression (carriedSlot : Nat)
  | affineCoefficient (carriedSlot termIndex : Nat)
  | affineBasis (carriedSlot termIndex : Nat)
  | familyElement
      (carriedSlot : Nat)
      (index : RuntimeExprRef .integer)
      (nested : MatrixFactPath)
  deriving BEq, DecidableEq, Repr

inductive BoundFactPath where
  | affineCoefficientBound (carriedSlot termIndex : Nat)
  | affineNoiseBound (carriedSlot : Nat)
  | matrixTotalBound (carriedSlot : Nat)
  | familyElement
      (carriedSlot : Nat)
      (index : RuntimeExprRef .integer)
      (nested : BoundFactPath)
  deriving BEq, DecidableEq, Repr

/-- Typed path to a scalar runtime expression in a carried value. -/
inductive RuntimeFactPath : RuntimeScalarType → Type where
  | integerValue (carriedSlot : Nat) : RuntimeFactPath .integer
  | booleanValue (carriedSlot : Nat) : RuntimeFactPath .boolean
  | familyElement {type : RuntimeScalarType}
      (carriedSlot : Nat)
      (index : RuntimeExprRef .integer)
      (nested : RuntimeFactPath type) : RuntimeFactPath type
  deriving BEq, DecidableEq, Repr

/-- Typed path to an integer interval endpoint in a carried value. -/
inductive IntBoundFactPath where
  | lower (carriedSlot : Nat)
  | upper (carriedSlot : Nat)
  | familyElement
      (carriedSlot : Nat)
      (index : RuntimeExprRef .integer)
      (nested : IntBoundFactPath)
  deriving BEq, DecidableEq, Repr

/-- Equality of lane-uniform family-bound locations. Family indices are intentionally ignored:
the recurrence resolver accepts a family bound only when its element template evaluates without a
lane index, so every lane has the same numeric bound. -/
def BoundFactPath.sameUniformLocation : BoundFactPath → BoundFactPath → Bool
  | .affineCoefficientBound slot term, .affineCoefficientBound otherSlot otherTerm =>
      slot == otherSlot && term == otherTerm
  | .affineNoiseBound slot, .affineNoiseBound otherSlot => slot == otherSlot
  | .matrixTotalBound slot, .matrixTotalBound otherSlot => slot == otherSlot
  | .familyElement slot _ nested, .familyElement otherSlot _ otherNested =>
      slot == otherSlot && nested.sameUniformLocation otherNested
  | _, _ => false

/-- Signed-bound counterpart of `BoundFactPath.sameUniformLocation`. -/
def IntBoundFactPath.sameUniformLocation : IntBoundFactPath → IntBoundFactPath → Bool
  | .lower slot, .lower otherSlot | .upper slot, .upper otherSlot => slot == otherSlot
  | .familyElement slot _ nested, .familyElement otherSlot _ otherNested =>
      slot == otherSlot && nested.sameUniformLocation otherNested
  | _, _ => false

inductive ConcatAxis where
  | rows
  | columns
  | diagonal
  deriving BEq, DecidableEq, Repr

structure ConcatPart where
  matrixType : MatrixTypeExpr
  rowOffset : IntExpr
  columnOffset : IntExpr

/-- The checked matrix types and placements of a structural concatenation. -/
structure ConcatLayout where
  axis : ConcatAxis
  parts : List ConcatPart
  output : MatrixTypeExpr

inductive RuntimeExpr : RuntimeScalarType → Type where
  | intWire (wire : ValueInstanceRef) : RuntimeExpr .integer
  | boolWire (wire : ValueInstanceRef) : RuntimeExpr .boolean
  | intConstant (value : Int) : RuntimeExpr .integer
  | boolConstant (value : Bool) : RuntimeExpr .boolean
  | parameter (value : IntExpr) : RuntimeExpr .integer
  | intBinary (operation : IntBinaryOp)
      (left right : RuntimeExpr .integer) : RuntimeExpr .integer
  | compare (operation : IntCompareOp)
      (left right : RuntimeExpr .integer) : RuntimeExpr .boolean
  | bitExtract (value : RuntimeExpr .integer) (position : IntExpr) : RuntimeExpr .boolean
  | boolToInt (value : RuntimeExpr .boolean) : RuntimeExpr .integer
  | thresholdDecodeBool
      (matrix : ValueInstanceRef)
      (ciphertextModulus plaintextModulus position : IntExpr) : RuntimeExpr .boolean
  | extractCoefficient
      (matrix : MatrixExprRef)
      (position : IntExpr) : RuntimeExpr .integer
  | familyElement
      (elementType : RuntimeScalarType)
      (aggregate : FamilyAggregateRef)
      (indexRef : RuntimeExprRef .integer)
      (index : RuntimeExpr .integer) : RuntimeExpr elementType
  | select
      (resultType : RuntimeScalarType)
      (index : RuntimeExpr .integer)
      (branches : List (RuntimeExprRef resultType)) : RuntimeExpr resultType
  | loopIndex (loop : LoopRef) : RuntimeExpr .integer
  /-- Analyzer-only placeholder. It must be eliminated when a sequential template is instantiated. -/
  | carriedInput {type : RuntimeScalarType} (path : RuntimeFactPath type) : RuntimeExpr type

inductive MatrixExpr where
  | wire (reference : MatrixInstanceRef)
  | zero (type : MatrixTypeExpr)
  /-- A symbolic identity with a checked square matrix type. This is not an executable IR node. -/
  | identity (type : MatrixTypeExpr)
  /-- The deterministic gadget matrix produced by the executable IR node. Its digit count is
  derived from the evaluated matrix dimensions, exactly as in `Mxx.Ir.evaluateNode`. -/
  | gadget (type : MatrixTypeExpr) (base : IntExpr)
  | add (left right : MatrixExpr)
  | negate (value : MatrixExpr)
  | multiply (left right : MatrixExpr)
  | scalarMultiply (scalar : IntExpr) (value : MatrixExpr)
  | rowSlice (value : MatrixExpr) (start stop : IntExpr)
  | rowConcat (parts : List MatrixExpr)
  | columnSlice (value : MatrixExpr) (start stop : IntExpr)
  | columnConcat (parts : List MatrixExpr)
  | diagonalConcat (parts : List MatrixExpr)
  | rowCoefficientEmbed (layout : ConcatLayout) (part : Nat) (value : MatrixExpr)
  | columnBasisEmbed (layout : ConcatLayout) (part : Nat) (value : MatrixExpr)
  | diagonalCoefficientEmbed (layout : ConcatLayout) (part : Nat) (value : MatrixExpr)
  | diagonalBasisEmbed (layout : ConcatLayout) (part : Nat) (value : MatrixExpr)
  | select (index : RuntimeExpr .integer) (branches : List MatrixExpr)
  | loopResult
      (type : MatrixTypeExpr)
      (summary : FactRecurrenceInstanceRef)
      (path : MatrixFactPath)
  /-- Analyzer-only typed placeholder. It has no ordinary matrix denotation. -/
  | carriedInput (type : MatrixTypeExpr) (path : MatrixFactPath)

end Mxx.Certificate
