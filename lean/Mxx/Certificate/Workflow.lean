import Mxx.Certificate.Registry
import Mxx.Certificate.ExpressionArena
import Mxx.Certificate.SymbolicForm
import Mxx.Certificate.SymbolicRecurrence

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

def DeclaredBoundExpr.toBoundExpr : DeclaredBoundExpr → BoundExpr
  | .constant value => .constant value
  | .parameter value => .parameter value
  | .add left right => .add left.toBoundExpr right.toBoundExpr
  | .multiply left right => .multiply left.toBoundExpr right.toBoundExpr
  | .maximum left right => .maximum left.toBoundExpr right.toBoundExpr
  | .absolute value => .absolute value
  | .floorDivide value divisor => .floorDivide value.toBoundExpr divisor
  | .matrixProduct ringDimension innerDimension left right =>
      .matrixProduct ringDimension innerDimension left.toBoundExpr right.toBoundExpr
  | .minimum left right => .minimum left.toBoundExpr right.toBoundExpr

inductive InputValueContract where
  | matrixExact (type : MatrixTypeExpr)
  | matrixBounded (type : MatrixTypeExpr) (bound : DeclaredBoundExpr)
  | integerRange (lower upper : IntExpr)
  | boolean
  | bytes (length : IntExpr)
  | family (count : IntExpr) (element : InputValueContract)

structure InputContract where
  inputs : List (ProtocolInputId × String × InputValueContract)

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

/-- Closed semantic identities required by an endpoint rule. These anchors are resolved to frozen
wire identities before analysis; endpoint rules never search for labels or accept symbolic
expressions supplied by a certificate. -/
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

/-- The single canonical protocol declaration consumed by the Lean analyzer. -/
structure ClosedProtocolDecl where
  parameters : List ParameterDecl
  bundle : ClosedProtocolBundle

inductive StaticObligation where
  | positiveModulus (value : IntExpr)
  | positiveDivisor (value : Nat)
  | matchingMatrixTypes (left right : MatrixTypeExpr)
  | intBoundNonnegative (value : IntBoundExpr)
  | intBoundPositive (value : IntBoundExpr)
  | intBoundsOrdered (lower upper : IntBoundExpr)
  | thresholdNoise
      (noise : BoundExpr)
      (ciphertextModulus plaintextModulus : IntExpr)
  | diamondFalseInterval (noise : BoundExpr) (ciphertextModulus : IntExpr)
  | diamondTrueInterval (noise : BoundExpr) (ciphertextModulus : IntExpr)

inductive InputObligation where
  | matrixNorm (input : ProtocolInputId) (bound : BoundExpr)
  | integerRange (input : ProtocolInputId) (lower upper : IntExpr)

inductive SemanticObligation where
  | lemma (id : SemanticLemmaId) (anchor : SemanticAnchorRef)

structure DerivedObligations where
  static : List StaticObligation
  input : List InputObligation
  semantic : List SemanticObligation

structure EndpointFact where
  anchor : SemanticAnchorRef
  specification : EndpointSpecId
  resolvedEndpoint : ValueInstanceRef
  stage : StageId
  workflowOutput : String
  idealOutput : String
  comparatorActualInput : String
  comparatorIdealInput : String
  comparatorResultOutput : String
  failureValue : Bool

structure ScopedWireFact where
  wire : CoreWireRef
  matrixType : Option MatrixTypeExpr
  fact : ValueFact

abbrev ScopedWireFactTable := List ScopedWireFact

structure AnalysisResult where
  expressionArena : ExpressionArena := { entries := [] }
  symbolicFormArena : SymbolicMatrixFormArena := {}
  boundWitnessArena : BoundWitnessArena := {}
  symbolicMatrixFacts : List MatrixSymbolicFact := []
  facts : ScopedWireFactTable
  families : List (JointFamilyId × JointFamilyFact)
  recurrences : List (FactRecurrenceInstanceRef × FactRecurrence)
  symbolicRecurrences : List SymbolicRecurrenceTransfer := []
  staticObligations : List StaticObligation
  inputObligations : List InputObligation
  semanticObligations : List SemanticObligation
  endpointFacts : List EndpointFact
  usedRules : List RuleUse

end Mxx.Certificate
