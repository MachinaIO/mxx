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

/-- Analyzer-owned provenance for one parallel-family summary.  This is emitted by the same
analysis branch that constructs the corresponding `JointFamilyFact`; it is not certificate or
protocol input.  Retaining the exact frozen loop and the body-analysis products lets semantic
soundness replay the actual parallel trace instead of trusting `elementTuple` as an assertion. -/
structure ParallelFamilyDerivationSource where
  family : JointFamilyId
  loopSite : CoreNodeRef
  childScope : StaticScopeId
  definition : String
  count : IntExpr
  indexSlot : Nat
  /-- The analyzer-owned arena reference for this loop's index expression.  Retaining the
  reference is necessary when the body output template is instantiated at an actual lane; it is
  not certificate input. -/
  indexReference : RuntimeExprRef .integer
  indexExpression : RuntimeExpr .integer
  bindings : List (String × IntExpr)
  modes : List Mxx.Ir.LoopInputMode
  argumentRefs : List Mxx.Ir.WireRef
  outputCount : Nat
  outputTypes : List Mxx.Ir.WireTypeExpr
  body : Mxx.Ir.Scope
  seededFacts : ScopedWireFactTable
  analyzedFacts : ScopedWireFactTable
  outputFacts : ScopedWireFactTable
  elementTemplates : List ValueFactTemplate

/-- Internal consistency required before a retained parallel-family source may be used by a
semantic proof.  In particular, the element templates are recovered from the exact analyzed body
outputs rather than trusted independently from the family summary. -/
def ParallelFamilyDerivationSource.MatchesFamily
    (source : ParallelFamilyDerivationSource)
    (family : JointFamilyFact) : Prop :=
  source.family = family.id ∧
    source.count = family.count ∧
    source.indexSlot = family.indexVariable.slot ∧
    source.outputCount = family.outputArity ∧
    source.elementTemplates = family.elementTuple.toList

/-- The retained output facts are exactly the frozen child scope outputs, in port order. -/
def ParallelFamilyDerivationSource.OutputFactsMatchBody
    (source : ParallelFamilyDerivationSource) : Prop :=
  source.childScope = ⟨source.loopSite.scope.path ++ [source.definition]⟩ ∧
    source.body.outputs.length = source.outputCount ∧
    source.outputFacts.map (fun fact => fact.wire) =
      source.body.outputs.map (fun output =>
        { stage := source.loopSite.stage
          scope := source.childScope
          node := ⟨output.2.node⟩
          port := output.2.port })

/-- Analyzer-owned public summary of a requirement acceptance wrapper. The detailed dependent
wrapper proof stays in the analyzer; this summary carries only frozen identities needed for later
coupling lookup. -/
structure RequirementAcceptanceSummary where
  requirementIndex : Nat
  outputName : String
  outputWire : CoreWireRef
  selectedRecurrence : SequentialRecurrenceInstanceRef
  selectedSlot : Nat
  deriving BEq, DecidableEq, Repr

structure AnalysisResult where
  expressionArena : ExpressionArena := { entries := [] }
  symbolicFormArena : SymbolicMatrixFormArena := {}
  boundWitnessArena : BoundWitnessArena := {}
  symbolicMatrixFacts : List MatrixSymbolicFact := []
  facts : ScopedWireFactTable
  families : List (JointFamilyId × JointFamilyFact)
  parallelFamilyDerivations : List ParallelFamilyDerivationSource := []
  symbolicRecurrences : List SymbolicRecurrenceTransfer := []
  requirementAcceptances : List RequirementAcceptanceSummary := []
  staticObligations : List StaticObligation
  inputObligations : List InputObligation
  semanticObligations : List SemanticObligation
  endpointFacts : List EndpointFact
  usedRules : List RuleUse

end Mxx.Certificate
