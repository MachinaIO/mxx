import Mxx.Certificate.Identity

/-! Syntax-only declarations shared by generated protocols and the active operational checker. -/

namespace Mxx.Certificate

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

inductive EndpointSpecId where
  | toyThresholdDecode
  | diamondBooleanInterval
  deriving BEq, DecidableEq, Repr

/-- A construction-time DSL label resolved to frozen wires before certificate emission. -/
structure SemanticAnchorRef where
  stage : StageId
  label : String
  deriving BEq, DecidableEq, Repr

inductive ProtocolInputDestination where
  | workflowStage (stage : StageId) (inputName : String)
  | requirement (index : Nat) (inputName : String)
  | ideal (inputName : String)
  deriving BEq, DecidableEq, Repr

structure ProtocolInputBinding where
  input : ProtocolInputId
  destinations : List ProtocolInputDestination

structure ComparatorEndpointBinding where
  endpoint : EndpointSpecId
  actualInput : String
  idealInput : String
  resultOutput : String
  failureValue : Bool

inductive ComparatorSpec where
  | equality (endpointBindings : List ComparatorEndpointBinding)
  | equalityAfterMap
      (program : Mxx.Ir.Prog)
      (endpointBindings : List ComparatorEndpointBinding)

inductive EndpointSemanticBinding where
  | thresholdDecode
  | diamondBoolean
      (residual carrier : SemanticAnchorRef)
      (message : ProtocolInputId)

structure EndpointAnchor where
  specification : EndpointSpecId
  stage : StageId
  semanticAnchor : SemanticAnchorRef
  semantics : EndpointSemanticBinding
  workflowOutput : String
  idealOutput : String

structure EndpointAnchors where
  entries : List EndpointAnchor

structure SemanticAnchorBinding where
  anchor : SemanticAnchorRef
  wires : List CoreWireRef

structure ProtocolPreconditionSpec where
  requirementOutputs : List String

structure ClosedProtocolBundle where
  workflow : Mxx.Ir.Workflow
  ideal : Mxx.Ir.Prog
  requirements : List Mxx.Ir.Prog
  comparator : ComparatorSpec
  endpoints : EndpointAnchors
  anchorBindings : List SemanticAnchorBinding
  endpointSpecs : List EndpointSpecId
  inputContract : InputContract
  inputBindings : List ProtocolInputBinding
  preconditionSpec : ProtocolPreconditionSpec

inductive ParameterKind where
  | dimension
  | integer
  | rational
  deriving BEq, DecidableEq, Repr

structure ParameterDecl where
  name : String
  kind : ParameterKind
  deriving BEq, DecidableEq, Repr

structure ClosedProtocolDecl where
  parameters : List ParameterDecl
  bundle : ClosedProtocolBundle

end Mxx.Certificate
