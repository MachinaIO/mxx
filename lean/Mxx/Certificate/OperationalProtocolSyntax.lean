import Mxx.Certificate.Identity

namespace Mxx.Certificate

/-! Generic protocol data consumed by the operational checker. Endpoint semantics stay owner-side. -/

inductive DeclaredBoundExpr where
  | constant (value : Nat)
  | parameter (value : IntExpr)
  | add (left right : DeclaredBoundExpr)
  | multiply (left right : DeclaredBoundExpr)
  | maximum (left right : DeclaredBoundExpr)
  | absolute (value : IntExpr)
  | floorDivide (value : DeclaredBoundExpr) (positiveDivisor : Nat)
  | matrixProduct
      (ringDimension innerDimension : IntExpr)
      (left right : DeclaredBoundExpr)
  | minimum (left right : DeclaredBoundExpr)

inductive InputValueContract where
  | matrixExact (type : MatrixTypeExpr)
  | matrixBounded (type : MatrixTypeExpr) (bound : DeclaredBoundExpr)
  | integerRange (lower upper : IntExpr)
  | boolean
  | bytes (length : IntExpr)
  | family (count : IntExpr) (element : InputValueContract)

structure InputContract where
  inputs : List (ProtocolInputId × String × InputValueContract)

/-- The only protocol-level data inspected by operational workflow evaluation. -/
structure OperationalWorkflowSpec where
  workflow : Mxx.Ir.Workflow
  inputContract : InputContract

end Mxx.Certificate
