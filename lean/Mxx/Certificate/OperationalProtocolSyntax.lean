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
  | matrixExact
      (type : MatrixTypeExpr)
      (canonicalExclusiveUpper : Option IntExpr)
      (isConstantPolynomial : Bool)
  | matrixBounded (type : MatrixTypeExpr) (bound : DeclaredBoundExpr)
  | integerRange (lower upper : IntExpr)
  | boolean
  | bytes (length : IntExpr)
  | family (count : IntExpr) (element : InputValueContract)

structure InputContract where
  inputs : List (ProtocolInputId × String × InputValueContract)

/-- Closed executable decoder semantics selected by an operational target. -/
inductive OperationalDecoderKind where
  | thresholdDecode (plaintextModulus : IntExpr)
  | booleanInterval

structure OperationalDecoderTarget where
  targetId : String
  residualStage : StageId
  residualOutput : String
  decoderStage : StageId
  decoderNode : Nat
  kind : OperationalDecoderKind

/-- The only protocol-level data inspected by operational workflow evaluation. -/
structure OperationalWorkflowSpec where
  workflow : Mxx.Ir.Workflow
  inputContract : InputContract
  operationalDecoderTargets : List OperationalDecoderTarget

private def decoderNodeAt (scope : Mxx.Ir.Scope) (wire : Mxx.Ir.WireRef) : Option Mxx.Ir.Node :=
  if wire.port = 0 then scope.nodes[wire.node]? else none

private def decoderOneArgument (node : Mxx.Ir.Node) : Option Mxx.Ir.WireRef :=
  match node.arguments with
  | [argument] => some argument
  | _ => none

private def decoderTwoArguments (node : Mxx.Ir.Node) : Option (Mxx.Ir.WireRef × Mxx.Ir.WireRef) :=
  match node.arguments with
  | [left, right] => some (left, right)
  | _ => none

private def decoderConstantInt
    (scope : Mxx.Ir.Scope) (wire : Mxx.Ir.WireRef) (expected : Int) : Bool :=
  match decoderNodeAt scope wire with
  | some { kind := .constantInt actual, arguments := [], .. } => actual == expected
  | _ => false

/-- Match the closed Boolean interval decoder shape and recover its residual wire and q. This
matcher is protocol-independent: it validates only the frozen executable node chain selected by
an `OperationalDecoderTarget`. -/
def matchBooleanIntervalDecoder
    (scope : Mxx.Ir.Scope)
    (decoderNode : Nat) : Option (Mxx.Ir.WireRef × IntExpr) := do
  let result ← scope.nodes[decoderNode]?
  guard (result.kind == .intCompare .equal)
  let (sumRef, twoRef) ← decoderTwoArguments result
  guard (decoderConstantInt scope twoRef 2)

  let sum ← decoderNodeAt scope sumRef
  guard (sum.kind == .intBinary .add)
  let (lowerIntRef, upperIntRef) ← decoderTwoArguments sum
  let lowerInt ← decoderNodeAt scope lowerIntRef
  guard (lowerInt.kind == .boolToInt)
  let lowerBoolRef ← decoderOneArgument lowerInt
  let upperInt ← decoderNodeAt scope upperIntRef
  guard (upperInt.kind == .boolToInt)
  let upperBoolRef ← decoderOneArgument upperInt

  let lower ← decoderNodeAt scope lowerBoolRef
  guard (lower.kind == .intCompare .lessEqual)
  let (quarterRef, coefficientRef) ← decoderTwoArguments lower
  let upper ← decoderNodeAt scope upperBoolRef
  guard (upper.kind == .intCompare .lessEqual)
  let (sameCoefficientRef, upperRef) ← decoderTwoArguments upper
  guard (coefficientRef == sameCoefficientRef)

  let upperValue ← decoderNodeAt scope upperRef
  guard (upperValue.kind == .intBinary .multiply)
  let (sameQuarterRef, threeRef) ← decoderTwoArguments upperValue
  guard (quarterRef == sameQuarterRef)
  guard (decoderConstantInt scope threeRef 3)

  let coefficient ← decoderNodeAt scope coefficientRef
  let position ← match coefficient.kind with
    | .extractCoefficient position => pure position
    | _ => none
  guard (position == .constant 0)
  let residual ← decoderOneArgument coefficient

  let quarter ← decoderNodeAt scope quarterRef
  match quarter.kind, quarter.arguments with
  | .evaluateInt (.roundDivide (.subtract modulus (.constant 2)) (.constant 4)), [] =>
      pure (residual, modulus)
  | _, _ => none

end Mxx.Certificate
