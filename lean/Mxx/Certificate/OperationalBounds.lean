import Mxx.Certificate.Derivation
import Mxx.Certificate.OperationalProtocolSyntax

/-! # Linear operational hard-bound estimator

This module is deliberately separate from the symbolic certificate analyzer.  It evaluates a
frozen scope in node order, after `ProgramDerivation` has checked the generator-selected rule
and operands.  The resulting values are estimates used for parameter search; they are not yet
runtime-bound theorems.

The only recursive bound language is for a sequential-loop numeric state transition.  Ordinary
scope evaluation stores concrete integer bounds, so it neither reconstructs symbolic expressions
nor searches the graph.
-/

namespace Mxx.Certificate

open Mxx.Ir

inductive OperationalBoundPath where
  | matrixMaximum (depth slot : Nat)
  | integerLower (depth slot : Nat)
  | integerUpper (depth slot : Nat)
  deriving BEq, DecidableEq, Repr

inductive OperationalParameterDomain where
  | loopIndex (slot count : Nat)
  | parameter
      (name : String)
      (environment : ParamEnvironment)
      (domains : List OperationalParameterDomain)
      (expression : IntExpr)
  deriving BEq, Repr

inductive ContextualExtremum where
  | minimum
  | maximum
  | maximumAbsolute
  deriving BEq, Repr

inductive OperationalBoundExpr where
  | closedInt (value : IntExpr)
  | contextual
      (kind : ContextualExtremum)
      (environment : ParamEnvironment)
      (domains : List OperationalParameterDomain)
      (value : IntExpr)
  | previous (path : OperationalBoundPath)
  | negate (value : OperationalBoundExpr)
  | add (left right : OperationalBoundExpr)
  | subtract (left right : OperationalBoundExpr)
  | multiply (left right : OperationalBoundExpr)
  | divide (left right : OperationalBoundExpr)
  | minimum (left right : OperationalBoundExpr)
  | maximum (left right : OperationalBoundExpr)
  | centeredCap (modulus value : OperationalBoundExpr)
  | matrixProduct
      (ringDimension innerDimension left right : OperationalBoundExpr)
  | recurrence
      (count : Nat)
      (initial transition : List OperationalBoundExpr)
      (slot : Nat)
  | recurrenceState
      (count : Nat)
      (paths : List OperationalBoundPath)
      (initial transition : List OperationalBoundExpr)
      (output : OperationalBoundPath)
  deriving BEq, Repr

inductive CanonicalRange where
  | unknown
  | below (upperExclusive : Nat)
  deriving BEq, DecidableEq, Repr

inductive ProgramInstanceKey where
  | temporary
  | workflowStage (stage : StageId)
  | ideal
  | requirement (index : Nat)
  | standalone (checkedProgramOrdinal : Nat)
  deriving BEq, DecidableEq, Repr

inductive ScopeTemplateKey where
  | root (program : ProgramInstanceKey)
  | callBody (parent : ScopeTemplateKey) (callNode : Nat)
  | parallelBody (parent : ScopeTemplateKey) (loopNode : Nat)
  | sequentialBody (parent : ScopeTemplateKey) (loopNode : Nat)
  deriving BEq, DecidableEq, Repr

inductive LoopCoordinate where
  | loopBinder (owner : ScopeTemplateKey) (loopNode binderSlot : Nat)
  | loopBinderOffset (owner : ScopeTemplateKey) (loopNode binderSlot offset : Nat)
  deriving BEq, DecidableEq, Repr

structure FamilyTemplateBinder where
  owner : ScopeTemplateKey
  producerNode : Nat
  binderSlot : Nat
  deriving BEq, DecidableEq, Repr

inductive OperationalValueOrigin where
  | local (scope : ScopeTemplateKey) (wire : WireRef)
  | protocolInput (input : ProtocolInputId)
  | protocolFamilyElement (input : ProtocolInputId) (index : Nat)
  | loopInstance (slot index : Nat) (source : OperationalValueOrigin)
  | selected
      (binder : FamilyTemplateBinder)
      (index : OperationalValueOrigin)
      (source : OperationalValueOrigin)
  deriving BEq, DecidableEq, Repr

structure DynamicSelectionIdentity where
  index : OperationalValueOrigin
  deriving BEq, DecidableEq, Repr

structure DeterministicHashIdentity where
  keyOrigin : OperationalValueOrigin
  matrixType : MatrixTypeExpr
  parameterEnvironment : ParamEnvironment
  parameterDomains : List OperationalParameterDomain
  tagPrefix : List Nat
  tagExpressions : List IntExpr
  tagDecimalExpressions : List IntExpr
  tagU64LeExpressions : List IntExpr
  trailingIntegerOrigins : List OperationalValueOrigin
  deriving BEq, Repr

inductive MatrixOriginIdentity where
  | value (scope : ScopeTemplateKey) (wire : WireRef)
  | protocolInput (input : ProtocolInputId)
  | protocolFamilyElement (input : ProtocolInputId) (index : Nat)
  | deterministicHash (query : DeterministicHashIdentity)
  | loopInstance (slot index : Nat) (source : MatrixOriginIdentity)
  | selected
      (binder : FamilyTemplateBinder)
      (selection : DynamicSelectionIdentity)
      (source : MatrixOriginIdentity)
  deriving BEq, Repr

inductive PublicMatrixIdentity where
  | sampledTrapdoor (scope : ScopeTemplateKey) (wire : WireRef)
  | gadget
      (paramsId : Mxx.SamplerParamsId)
      (params : Mxx.SamplerParams)
      (inputRows : Nat)
      (base : Int)
      (small : Bool)
      (digitCount : Nat)
  | selected
      (binder : FamilyTemplateBinder)
      (selection : DynamicSelectionIdentity)
      (source : PublicMatrixIdentity)
  | loopInstance (slot index : Nat) (source : PublicMatrixIdentity)
  deriving BEq, DecidableEq, Repr

private def temporaryScope : ScopeTemplateKey := .root .temporary

/-! ## Flat operational polynomial

The parameter-search checker keeps exact sums of ordered products.  This is intentionally flat:
there is no recursive matrix expression and no stored coefficient/carrier split. -/

inductive OperationalFactorRole where
  | bounded
  | large
  deriving BEq, DecidableEq, Repr

structure OperationalMatrixMetadata where
  isConstantPolynomial : Bool := false
  knownZeroRows : Option IntExpr := none
  deriving BEq, DecidableEq, Repr

inductive OperationalProductMode where
  | ordinaryMatrixProduct
  | leftPolynomialScalarBroadcast
  | rightPolynomialScalarBroadcast
  | swappedRowVectorScalarProduct
  deriving BEq, DecidableEq, Repr

inductive OperationalFactorTransform where
  | negate
  | transpose
  | rowSlice (start stop : IntExpr)
  | columnSlice (start stop : IntExpr)
  | rowEmbed (axis : ConcatAxis) (part : Nat)
  | columnEmbed (axis : ConcatAxis) (part : Nat)
  | reshape (rows columns : IntExpr)
  | constantCoefficient (index : IntExpr)
  deriving BEq, Repr

inductive OperationalPrimitiveIdentity where
  | matrix (identity : MatrixOriginIdentity)
  | publicMatrix (identity : PublicMatrixIdentity)
  | value (identity : OperationalValueOrigin)
  | parameterScalar
      (environment : ParamEnvironment)
      (domains : List OperationalParameterDomain)
      (value : IntExpr)
  | identityMatrix (type : MatrixTypeExpr)
  | selectionIndicator
      (binder : FamilyTemplateBinder)
      (selection : DynamicSelectionIdentity)
      (branch : Nat)
  | indexedArtifact (input : ProtocolInputId) (index : IntExpr)
  | recurrenceResult (scope : ScopeTemplateKey) (node path : Nat)
  | carriedInput (path : Nat)
  deriving BEq, Repr

inductive OperationalCompressionKind where
  | boundedRun
  | boundedNoiseSum
  deriving BEq, DecidableEq, Repr

inductive OperationalProvenanceSegmentKind where
  | primitiveRun
  | boundedNoiseSum
  deriving BEq, DecidableEq, Repr

inductive OperationalCompressionToken where
  | primitive (identity : OperationalPrimitiveIdentity)
  | transform (value : OperationalFactorTransform)
  | productMode (value : OperationalProductMode)
  | intermediateType (value : MatrixTypeExpr)
  | productStart
  | productEnd
  | groupStart
  | groupEnd
  | sumStart
  | sumEnd
  | termStart (coefficient : Int)
  | termEnd
  | summaryBound (bound : OperationalBoundExpr)
  | summaryMetadata (metadata : OperationalMatrixMetadata)
  | segmentStart (kind : OperationalProvenanceSegmentKind) (length : Nat)
  | segmentEnd
  deriving BEq, Repr

structure OperationalCompressionOrigin where
  kind : OperationalCompressionKind
  tokens : List OperationalCompressionToken
  deriving BEq, Repr

structure OperationalBoundedFactorSummary where
  matrixType : MatrixTypeExpr
  hardBound : OperationalBoundExpr
  metadata : OperationalMatrixMetadata
  provenance : List OperationalCompressionToken
  deriving BEq, Repr

inductive OperationalCompressionProtection where
  | relationOwner
  | decompositionOwner
  | exactOneIndicator
  | endpointIdentity
  | originPreservingArtifact
  deriving BEq, DecidableEq, Repr

/-- A relation owns a relation-free copy of its exact target polynomial. This separate flat type
breaks the recursive cycle that would arise if a target snapshot could itself own relations. -/
inductive RelationSnapshotFactorLeaf where
  | primitive (identity : OperationalPrimitiveIdentity)
  | boundedSummary
      (origin : OperationalCompressionOrigin)
      (summary : OperationalBoundedFactorSummary)
  | exactTransform (tokens : List OperationalCompressionToken) (type : MatrixTypeExpr)
  deriving BEq, Repr

structure RelationSnapshotFactor where
  leaf : RelationSnapshotFactorLeaf
  transforms : List OperationalFactorTransform := []
  inputType : MatrixTypeExpr
  outputType : MatrixTypeExpr
  role : OperationalFactorRole
  boundedSummary : Option OperationalBoundedFactorSummary := none
  protections : List OperationalCompressionProtection := []
  deriving BEq, Repr

structure RelationSnapshotProduct where
  factors : List RelationSnapshotFactor
  modes : List OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq, Repr

structure RelationSnapshotTerm where
  coefficient : Int
  product : RelationSnapshotProduct
  deriving BEq, Repr

abbrev RelationSnapshotPolynomial := List RelationSnapshotTerm

structure RelationTargetSummary where
  origin : MatrixOriginIdentity
  matrixType : MatrixTypeExpr
  matrixParams : Mxx.SamplerParams
  totalHardBound : OperationalBoundExpr
  canonicalRange : CanonicalRange
  polynomial : RelationSnapshotPolynomial
  deriving BEq, Repr

inductive ReconstructionStatus where
  | available
  | smallRangeMissing (requiredExclusiveUpper : Nat)
  deriving BEq, DecidableEq, Repr

structure DecompositionRelation where
  producer : MatrixOriginIdentity
  publicIdentity : PublicMatrixIdentity
  inputOrigin : MatrixOriginIdentity
  inputSummary : RelationTargetSummary
  base : Int
  small : Bool
  digitCount : Nat
  status : ReconstructionStatus
  deriving BEq, Repr

structure PreimageRelation where
  producer : MatrixOriginIdentity
  publicIdentity : PublicMatrixIdentity
  targetOrigin : MatrixOriginIdentity
  targetSummary : RelationTargetSummary
  deriving BEq, Repr

inductive OperationalMatrixRelation where
  | decomposition (relation : DecompositionRelation)
  | preimage (relation : PreimageRelation)
  deriving BEq, Repr

inductive OperationalFactorLeaf where
  | primitive (identity : OperationalPrimitiveIdentity)
  | boundedSummary
      (origin : OperationalCompressionOrigin)
      (summary : OperationalBoundedFactorSummary)
  | exactTransform (tokens : List OperationalCompressionToken) (type : MatrixTypeExpr)
  deriving BEq, Repr

structure OperationalFactorKey where
  leaf : OperationalFactorLeaf
  transforms : List OperationalFactorTransform := []
  inputType : MatrixTypeExpr
  outputType : MatrixTypeExpr
  role : OperationalFactorRole
  boundedSummary : Option OperationalBoundedFactorSummary := none
  protections : List OperationalCompressionProtection := []
  relations : List OperationalMatrixRelation := []
  deriving BEq, Repr

structure OperationalProductKey where
  factors : List OperationalFactorKey
  modes : List OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq, Repr

structure OperationalTerm where
  coefficient : Int
  product : OperationalProductKey
  deriving BEq, Repr

abbrev OperationalPolynomial := List OperationalTerm

private def relationSnapshotLeaf : OperationalFactorLeaf → RelationSnapshotFactorLeaf
  | .primitive identity => .primitive identity
  | .boundedSummary origin summary => .boundedSummary origin summary
  | .exactTransform tokens type => .exactTransform tokens type

private def relationSnapshotFactor (factor : OperationalFactorKey) : RelationSnapshotFactor := {
  leaf := relationSnapshotLeaf factor.leaf
  transforms := factor.transforms
  inputType := factor.inputType
  outputType := factor.outputType
  role := factor.role
  boundedSummary := factor.boundedSummary
  protections := factor.protections.filter (· != .relationOwner)
}

private def relationSnapshotPolynomial
    (polynomial : OperationalPolynomial) : RelationSnapshotPolynomial :=
  polynomial.map fun term => {
    coefficient := term.coefficient
    product := {
      factors := term.product.factors.map relationSnapshotFactor
      modes := term.product.modes
      outputType := term.product.outputType
    }
  }

private def operationalLeafFromSnapshot : RelationSnapshotFactorLeaf → OperationalFactorLeaf
  | .primitive identity => .primitive identity
  | .boundedSummary origin summary => .boundedSummary origin summary
  | .exactTransform tokens type => .exactTransform tokens type

private def operationalPolynomialFromSnapshot
    (polynomial : RelationSnapshotPolynomial) : OperationalPolynomial :=
  polynomial.map fun term => {
    coefficient := term.coefficient
    product := {
      factors := term.product.factors.map fun factor => {
        leaf := operationalLeafFromSnapshot factor.leaf
        transforms := factor.transforms
        inputType := factor.inputType
        outputType := factor.outputType
        role := factor.role
        boundedSummary := factor.boundedSummary
        protections := factor.protections
        relations := []
      }
      modes := term.product.modes
      outputType := term.product.outputType
    }
  }

inductive OperationalFlatError where
  | incompatibleProduct (left right : MatrixTypeExpr)
  | malformedProduct
  | missingBoundedSummary
  | invalidKnownZeroRows
  | cannotPreserveNoiseSeparation
  | analysisLimitExceeded
  deriving BEq, DecidableEq, Repr

private def operationalAbsoluteCoefficient (value : Int) : Int :=
  if value < 0 then -value else value

private def operationalCoefficientContent : List OperationalTerm → Nat
  | [] => 1
  | head :: tail =>
      let content := tail.foldl (fun current term => Nat.gcd current term.coefficient.natAbs)
        head.coefficient.natAbs
      if content = 0 then 1 else content

private def insertCanonicalOperationalTerm
    (term : OperationalTerm) : OperationalPolynomial → OperationalPolynomial
  | [] => [term]
  | head :: tail =>
      if reprStr term.product < reprStr head.product then term :: head :: tail
      else head :: insertCanonicalOperationalTerm term tail

private def sortOperationalTerms (terms : OperationalPolynomial) : OperationalPolynomial :=
  terms.foldl (fun sorted term => insertCanonicalOperationalTerm term sorted) []

private def normalizeOperationalDimension : IntExpr → IntExpr
  | .add left right =>
      let left := normalizeOperationalDimension left
      let right := normalizeOperationalDimension right
      match left, right with
      | .constant 0, value | value, .constant 0 => value
      | .constant left, .constant right => .constant (left + right)
      | left, right => .add left right
  | .subtract left right =>
      let left := normalizeOperationalDimension left
      let right := normalizeOperationalDimension right
      match left, right with
      | value, .constant 0 => value
      | .constant left, .constant right => .constant (left - right)
      | left, right => .subtract left right
  | .multiply left right =>
      let left := normalizeOperationalDimension left
      let right := normalizeOperationalDimension right
      match left, right with
      | .constant 0, _ | _, .constant 0 => .constant 0
      | .constant 1, value | value, .constant 1 => value
      | .constant left, .constant right => .constant (left * right)
      | left, right => .multiply left right
  | .divide left right =>
      .divide (normalizeOperationalDimension left) (normalizeOperationalDimension right)
  | .roundDivide left right =>
      .roundDivide (normalizeOperationalDimension left) (normalizeOperationalDimension right)
  | .log2Ceil value => .log2Ceil (normalizeOperationalDimension value)
  | value => value

private def operationalDimensionEqual (left right : IntExpr) : Bool :=
  normalizeOperationalDimension left == normalizeOperationalDimension right

private def operationalSameRing (left right : MatrixTypeExpr) : Bool :=
  operationalDimensionEqual left.modulus right.modulus &&
    operationalDimensionEqual left.ringDimension right.ringDimension

private def operationalIsOne : IntExpr → Bool
  | value => normalizeOperationalDimension value == .constant 1

def inferOperationalProductMode
    (left right : MatrixTypeExpr) : Except OperationalFlatError
      (OperationalProductMode × MatrixTypeExpr) := do
  if !operationalSameRing left right then throw (.incompatibleProduct left right)
  if operationalDimensionEqual left.columns right.rows then
    pure (.ordinaryMatrixProduct, {
      modulus := left.modulus
      ringDimension := left.ringDimension
      rows := left.rows
      columns := right.columns
    })
  else if operationalIsOne left.rows && operationalIsOne left.columns then
    pure (.leftPolynomialScalarBroadcast, right)
  else if operationalIsOne right.rows && operationalIsOne right.columns then
    pure (.rightPolynomialScalarBroadcast, left)
  else if operationalIsOne left.rows && operationalIsOne right.rows &&
      operationalDimensionEqual left.columns right.columns then
    pure (.swappedRowVectorScalarProduct, {
      modulus := left.modulus
      ringDimension := left.ringDimension
      rows := .constant 1
      columns := left.columns
    })
  else throw (.incompatibleProduct left right)

private def operationalInnerDimension
    (mode : OperationalProductMode)
    (left : OperationalBoundedFactorSummary)
    (right : OperationalBoundedFactorSummary) : Except OperationalFlatError IntExpr := do
  match mode with
  | .ordinaryMatrixProduct =>
      match right.metadata.knownZeroRows with
      | none => pure left.matrixType.columns
      | some zeroRows => pure (.subtract left.matrixType.columns zeroRows)
  | .leftPolynomialScalarBroadcast | .rightPolynomialScalarBroadcast |
      .swappedRowVectorScalarProduct => pure (.constant 1)

def multiplyOperationalBoundedSummaries
    (mode : OperationalProductMode)
    (left right : OperationalBoundedFactorSummary) :
    Except OperationalFlatError OperationalBoundedFactorSummary := do
  let inner ← operationalInnerDimension mode left right
  let ringFactor := if left.metadata.isConstantPolynomial ||
      right.metadata.isConstantPolynomial then .closedInt (.constant 1)
    else .closedInt left.matrixType.ringDimension
  let outputType ← (inferOperationalProductMode left.matrixType right.matrixType).map (·.2)
  pure {
    matrixType := outputType
    hardBound := .multiply (.closedInt inner)
      (.multiply ringFactor (.multiply left.hardBound right.hardBound))
    metadata := {
      isConstantPolynomial := left.metadata.isConstantPolynomial &&
        right.metadata.isConstantPolynomial
      knownZeroRows := none
    }
    provenance := left.provenance ++
      [.productMode mode, .intermediateType outputType] ++ right.provenance
  }

def OperationalTerm.negate (term : OperationalTerm) : OperationalTerm :=
  { term with coefficient := -term.coefficient }

private def insertOperationalTerm
    (term : OperationalTerm) : OperationalPolynomial → OperationalPolynomial
  | [] => if term.coefficient = 0 then [] else [term]
  | head :: tail =>
      if head.product == term.product then
        let coefficient := head.coefficient + term.coefficient
        if coefficient = 0 then tail else { head with coefficient } :: tail
      else head :: insertOperationalTerm term tail

def normalizeOperationalTerms (terms : OperationalPolynomial) : OperationalPolynomial :=
  terms.foldl (fun result term ↦ insertOperationalTerm term result) []

def addOperationalPolynomials
    (left right : OperationalPolynomial) : OperationalPolynomial :=
  normalizeOperationalTerms (left ++ right)

def subtractOperationalPolynomials
    (left right : OperationalPolynomial) : OperationalPolynomial :=
  normalizeOperationalTerms (left ++ right.map OperationalTerm.negate)

def scaleOperationalPolynomial
    (scalar : Int) (terms : OperationalPolynomial) : OperationalPolynomial :=
  normalizeOperationalTerms (terms.map fun term ↦ {
    term with coefficient := scalar * term.coefficient
  })

private def multiplyOperationalTerms
    (left right : OperationalTerm) : Except OperationalFlatError OperationalTerm := do
  let leftLast ← match left.product.factors.getLast? with
    | some factor => pure factor
    | none => throw .malformedProduct
  let rightFirst ← match right.product.factors.head? with
    | some factor => pure factor
    | none => throw .malformedProduct
  let (mode, outputType) ← inferOperationalProductMode leftLast.outputType rightFirst.inputType
  pure {
    coefficient := left.coefficient * right.coefficient
    product := {
      factors := left.product.factors ++ right.product.factors
      modes := left.product.modes ++ [mode] ++ right.product.modes
      outputType
    }
  }

def multiplyOperationalPolynomials
    (left right : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let rows ← left.mapM fun leftTerm ↦ right.mapM (multiplyOperationalTerms leftTerm)
  pure (normalizeOperationalTerms rows.flatten)

def operationalLargeFactorCount (term : OperationalTerm) : Nat :=
  term.product.factors.countP fun factor ↦ factor.role == .large

def operationalTermIsNoise (term : OperationalTerm) : Bool :=
  operationalLargeFactorCount term = 0

def operationalTermIsSignal (term : OperationalTerm) : Bool :=
  0 < operationalLargeFactorCount term

private def operationalTermIsCompressionProtected (term : OperationalTerm) : Bool :=
  term.product.factors.any fun factor => !factor.protections.isEmpty

private def factorBoundedSummary
    (factor : OperationalFactorKey) : Except OperationalFlatError OperationalBoundedFactorSummary :=
  match factor.role, factor.boundedSummary with
  | .bounded, some summary => pure summary
  | _, _ => throw .missingBoundedSummary

private def boundedRunTokens
    (factors : List OperationalFactorKey)
    (modes : List OperationalProductMode)
    (summary : OperationalBoundedFactorSummary) : List OperationalCompressionToken :=
  [.productStart] ++
    (factors.flatMap fun factor ↦ match factor.leaf with
      | .primitive identity =>
          [.segmentStart .primitiveRun (1 + factor.transforms.length), .primitive identity] ++
            factor.transforms.map OperationalCompressionToken.transform ++ [.segmentEnd]
      | .boundedSummary origin _ =>
          [.segmentStart (match origin.kind with
            | .boundedRun => .primitiveRun
            | .boundedNoiseSum => .boundedNoiseSum) origin.tokens.length] ++
            origin.tokens ++ [.segmentEnd]
      | .exactTransform tokens _ =>
          [.segmentStart .primitiveRun tokens.length] ++ tokens ++ [.segmentEnd]) ++
    modes.map OperationalCompressionToken.productMode ++
    [.intermediateType summary.matrixType, .summaryBound summary.hardBound,
      .summaryMetadata summary.metadata, .productEnd]

private def summarizeEntireBoundedProduct
    (product : OperationalProductKey) :
    Except OperationalFlatError OperationalBoundedFactorSummary := do
  if product.factors.isEmpty || product.factors.any fun factor ↦ factor.role == .large then
    throw .cannotPreserveNoiseSeparation
  let summaries ← product.factors.mapM factorBoundedSummary
  let first ← match summaries.head? with
    | some summary => pure summary
    | none => throw .malformedProduct
  let pairs := product.modes.zip (summaries.drop 1)
  pairs.foldlM (init := first) fun current pair ↦
    multiplyOperationalBoundedSummaries pair.1 current pair.2

def compressEntireBoundedProduct
    (product : OperationalProductKey) : Except OperationalFlatError OperationalFactorKey := do
  if product.factors.any fun factor ↦ !factor.protections.isEmpty then
    throw .cannotPreserveNoiseSeparation
  let summary ← summarizeEntireBoundedProduct product
  let firstFactor ← match product.factors.head? with
    | some factor => pure factor
    | none => throw .malformedProduct
  if product.factors.length = 1 then
    pure firstFactor
  else
    let tokens := boundedRunTokens product.factors product.modes summary
    let origin : OperationalCompressionOrigin := { kind := .boundedRun, tokens }
    pure {
      leaf := .boundedSummary origin { summary with provenance := tokens }
      inputType := firstFactor.inputType
      outputType := product.outputType
      role := .bounded
      boundedSummary := some { summary with provenance := tokens }
    }

private def boundedNoiseTermSummary
    (term : OperationalTerm) : Except OperationalFlatError OperationalBoundedFactorSummary := do
  if !operationalTermIsNoise term then throw .cannotPreserveNoiseSeparation
  summarizeEntireBoundedProduct term.product

private def boundedNoiseTermTokens
    (term : OperationalTerm)
    (summary : OperationalBoundedFactorSummary) : List OperationalCompressionToken :=
  [.termStart term.coefficient, .segmentStart .boundedNoiseSum summary.provenance.length] ++
    summary.provenance ++ [.segmentEnd, .termEnd]

/-- Replace a sum of bounded-only products by one bounded summary.  The signed content is kept as
the sole additive coefficient, while the summary bound uses the triangle inequality.  The
summary is never reopened by multiplication; subsequent products use its stored hard bound. -/
def compressBoundedNoiseSum
    (terms : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let terms := sortOperationalTerms (normalizeOperationalTerms terms)
  if terms.isEmpty then return []
  if terms.any operationalTermIsSignal then throw .cannotPreserveNoiseSeparation
  if terms.any operationalTermIsCompressionProtected then
    throw .cannotPreserveNoiseSeparation
  match terms with
  | [{ product := { factors := [{ leaf := .boundedSummary origin _, .. }], .. }, .. }] =>
      if origin.kind == OperationalCompressionKind.boundedNoiseSum then return terms
  | _ => pure ()
  let summaries ← terms.mapM boundedNoiseTermSummary
  let firstTerm ← match terms.head? with
    | some term => pure term
    | none => throw .malformedProduct
  let content := operationalCoefficientContent terms
  let unsignedContent := Int.ofNat content
  let contentInt := if firstTerm.coefficient < 0 then -unsignedContent else unsignedContent
  let scaledBounds := (terms.zip summaries).map fun (term, summary) =>
    OperationalBoundExpr.multiply
      (.closedInt (.constant (operationalAbsoluteCoefficient (term.coefficient / contentInt))))
      summary.hardBound
  let hardBound := scaledBounds.foldl OperationalBoundExpr.add (.closedInt (.constant 0))
  let tokens := [.sumStart] ++
    (terms.zip summaries).flatMap fun (term, summary) =>
      boundedNoiseTermTokens { term with coefficient := term.coefficient / contentInt } summary ++
    [.summaryBound hardBound, .sumEnd]
  let metadata : OperationalMatrixMetadata := {
    isConstantPolynomial := summaries.all (·.metadata.isConstantPolynomial)
    knownZeroRows := none
  }
  let summary : OperationalBoundedFactorSummary := {
    matrixType := firstTerm.product.outputType
    hardBound
    metadata
    provenance := tokens
  }
  let origin : OperationalCompressionOrigin := { kind := .boundedNoiseSum, tokens }
  let factor : OperationalFactorKey := {
    leaf := .boundedSummary origin summary
    inputType := summary.matrixType
    outputType := summary.matrixType
    role := .bounded
    boundedSummary := some summary
  }
  pure [{
    coefficient := contentInt
    product := { factors := [factor], modes := [], outputType := summary.matrixType }
  }]

/-- Canonicalize exact products first, then collapse only the bounded-only subset.  Signal terms,
including terms with multiple Large factors, stay distributed and retain their ordered factors. -/
def normalizeOperationalPolynomial
    (terms : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let merged := normalizeOperationalTerms terms
  let noise := merged.filter operationalTermIsNoise
  let signal := merged.filter operationalTermIsSignal
  let protectedNoise := noise.filter operationalTermIsCompressionProtected
  let compressibleNoise := noise.filter fun term => !operationalTermIsCompressionProtected term
  let compressedNoise ← compressBoundedNoiseSum compressibleNoise
  pure (normalizeOperationalTerms (signal ++ protectedNoise ++ compressedNoise))

structure OperationalMatrixFact where
  subject : WireRef
  origin : MatrixOriginIdentity
  matrixType : MatrixTypeExpr
  matrixParams : Mxx.SamplerParams
  totalHardBound : OperationalBoundExpr
  polynomial : OperationalPolynomial := []
  metadata : OperationalMatrixMetadata := {}
  canonicalRange : CanonicalRange := .unknown
  identity : Option PublicMatrixIdentity := none
  relations : List OperationalMatrixRelation := []
  deriving BEq

abbrev OperationalExprId := Nat

private structure UniformBoundedSchema where
  matrixType : MatrixTypeExpr
  hardBound : OperationalBoundExpr
  metadata : OperationalMatrixMetadata
  deriving BEq

private structure UniformSnapshotFactorSchema where
  transforms : List OperationalFactorTransform
  inputType : MatrixTypeExpr
  outputType : MatrixTypeExpr
  role : OperationalFactorRole
  bounded : Option UniformBoundedSchema
  deriving BEq

private structure UniformSnapshotTermSchema where
  coefficient : Int
  factors : List UniformSnapshotFactorSchema
  modes : List OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq

private structure UniformTargetSchema where
  matrixType : MatrixTypeExpr
  matrixParams : Mxx.SamplerParams
  totalHardBound : OperationalBoundExpr
  canonicalRange : CanonicalRange
  polynomial : List UniformSnapshotTermSchema
  deriving BEq

private structure UniformRelationSchema where
  isPreimage : Bool
  target : UniformTargetSchema
  base : Int := 0
  small : Bool := false
  digitCount : Nat := 0
  status : ReconstructionStatus := .available
  deriving BEq

private structure UniformFactorSchema where
  transforms : List OperationalFactorTransform
  inputType : MatrixTypeExpr
  outputType : MatrixTypeExpr
  role : OperationalFactorRole
  bounded : Option UniformBoundedSchema
  protections : List OperationalCompressionProtection
  relations : List UniformRelationSchema
  deriving BEq

private structure UniformTermSchema where
  coefficient : Int
  factors : List UniformFactorSchema
  modes : List OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq

private structure UniformMatrixSchema where
  matrixType : MatrixTypeExpr
  matrixParams : Mxx.SamplerParams
  totalHardBound : OperationalBoundExpr
  polynomial : List UniformTermSchema
  metadata : OperationalMatrixMetadata
  canonicalRange : CanonicalRange
  hasPublicIdentity : Bool
  relations : List UniformRelationSchema
  deriving BEq

private def uniformBoundedSchema
    (summary : OperationalBoundedFactorSummary) : UniformBoundedSchema := {
  matrixType := summary.matrixType
  hardBound := summary.hardBound
  metadata := summary.metadata
}

private def uniformSnapshotFactorSchema
    (factor : RelationSnapshotFactor) : UniformSnapshotFactorSchema := {
  transforms := factor.transforms
  inputType := factor.inputType
  outputType := factor.outputType
  role := factor.role
  bounded := factor.boundedSummary.map uniformBoundedSchema
}

private def uniformSnapshotTermSchema
    (term : RelationSnapshotTerm) : UniformSnapshotTermSchema := {
  coefficient := term.coefficient
  factors := term.product.factors.map uniformSnapshotFactorSchema
  modes := term.product.modes
  outputType := term.product.outputType
}

private def uniformTargetSchema (target : RelationTargetSummary) : UniformTargetSchema := {
  matrixType := target.matrixType
  matrixParams := target.matrixParams
  totalHardBound := target.totalHardBound
  canonicalRange := target.canonicalRange
  polynomial := target.polynomial.map uniformSnapshotTermSchema
}

private def uniformRelationSchema : OperationalMatrixRelation → UniformRelationSchema
  | .decomposition relation => {
      isPreimage := false
      target := uniformTargetSchema relation.inputSummary
      base := relation.base
      small := relation.small
      digitCount := relation.digitCount
      status := relation.status
    }
  | .preimage relation => {
      isPreimage := true
      target := uniformTargetSchema relation.targetSummary
    }

private def uniformFactorSchema (factor : OperationalFactorKey) : UniformFactorSchema := {
  transforms := factor.transforms
  inputType := factor.inputType
  outputType := factor.outputType
  role := factor.role
  bounded := factor.boundedSummary.map uniformBoundedSchema
  protections := factor.protections
  relations := factor.relations.map uniformRelationSchema
}

private def uniformTermSchema (term : OperationalTerm) : UniformTermSchema := {
  coefficient := term.coefficient
  factors := term.product.factors.map uniformFactorSchema
  modes := term.product.modes
  outputType := term.product.outputType
}

private def operationalUniformSchema (fact : OperationalMatrixFact) : UniformMatrixSchema := {
  matrixType := fact.matrixType
  matrixParams := fact.matrixParams
  totalHardBound := fact.totalHardBound
  polynomial := fact.polynomial.map uniformTermSchema
  metadata := fact.metadata
  canonicalRange := fact.canonicalRange
  hasPublicIdentity := fact.identity.isSome
  relations := fact.relations.map uniformRelationSchema
}

private def matrixFactHasRelation (fact : OperationalMatrixFact) : Bool :=
  !fact.relations.isEmpty || fact.polynomial.any fun term =>
    term.product.factors.any fun factor => !factor.relations.isEmpty

private def boundaryLastPublicIdentity?
    (fact : OperationalMatrixFact) : Option PublicMatrixIdentity := do
  let term ← match fact.polynomial with | [term] => some term | _ => none
  let factor ← term.product.factors.getLast?
  match factor.leaf with
  | .primitive (.publicMatrix identity) => some identity
  | _ => none

private def boundaryFirstRelationPublicIdentity?
    (fact : OperationalMatrixFact) : Option PublicMatrixIdentity := do
  let term ← match fact.polynomial with | [term] => some term | _ => none
  let factor ← term.product.factors.head?
  if !factor.transforms.isEmpty then none else pure ()
  let producer ← match factor.leaf with
    | .primitive (.matrix origin) => some origin
    | _ => none
  let identities := factor.relations.filterMap fun relation => match relation with
    | .decomposition value =>
        if value.producer == producer && value.status == .available then
          some value.publicIdentity
        else none
    | .preimage value =>
        if value.producer == producer then some value.publicIdentity else none
  match identities with
  | [identity] => some identity
  | _ => none

private def publicIdentityTemplateEqual : PublicMatrixIdentity → PublicMatrixIdentity → Bool
  | .sampledTrapdoor leftScope leftWire, .sampledTrapdoor rightScope rightWire =>
      leftScope == rightScope && leftWire == rightWire
  | .gadget leftId leftParams leftRows leftBase leftSmall leftDigits,
      .gadget rightId rightParams rightRows rightBase rightSmall rightDigits =>
      leftId == rightId && leftParams == rightParams && leftRows == rightRows &&
        leftBase == rightBase && leftSmall == rightSmall && leftDigits == rightDigits
  | .loopInstance leftSlot _ leftSource, .loopInstance rightSlot _ rightSource =>
      leftSlot == rightSlot && publicIdentityTemplateEqual leftSource rightSource
  | .selected _ leftSelection leftSource, .selected _ rightSelection rightSource =>
      leftSelection == rightSelection && publicIdentityTemplateEqual leftSource rightSource
  | _, _ => false

private def selectionBinderInPublicIdentity?
    (selection : OperationalValueOrigin) : PublicMatrixIdentity → Option FamilyTemplateBinder
  | .selected binder identity source =>
      if identity.index == selection then some binder
      else selectionBinderInPublicIdentity? selection source
  | .loopInstance _ _ source => selectionBinderInPublicIdentity? selection source
  | .sampledTrapdoor .. | .gadget .. => none

private def selectedRelationPublicIdentity : OperationalMatrixRelation → PublicMatrixIdentity
  | .decomposition relation => relation.publicIdentity
  | .preimage relation => relation.publicIdentity

/-- Return the one binder proving that a matrix schema has already absorbed `selection`.  The
caller may then replace another uniform family with a representative wrapped by this exact binder,
without equating two unrelated selection origins. -/
private def absorbedSelectionBinder?
    (fact : OperationalMatrixFact)
    (selection : OperationalValueOrigin) : Option FamilyTemplateBinder :=
  let direct := fact.identity.toList ++ fact.relations.map selectedRelationPublicIdentity
  let factors := fact.polynomial.flatMap fun term => term.product.factors.flatMap fun factor =>
    let leaf := match factor.leaf with
      | .primitive (.publicMatrix identity) => [identity]
      | _ => []
    leaf ++ factor.relations.map selectedRelationPublicIdentity
  let binders := (direct ++ factors).filterMap (selectionBinderInPublicIdentity? selection)
  match binders with
  | [] => none
  | first :: rest => if rest.all (· == first) then some first else none

structure SelectedMatrixSummary where
  uniformSchema : Option UniformMatrixSchema
  relationFree : Bool
  sharedLastPublicIdentity : Option PublicMatrixIdentity
  sharedFirstRelationPublicIdentity : Option PublicMatrixIdentity
  deriving BEq

private structure SelectedMatrixFamily where
  selection : OperationalValueOrigin
  branches : Array OperationalMatrixFact
  count : Nat
  representsLoopLanes : Bool
  summary : SelectedMatrixSummary
  deriving BEq

private def selectedMatrixSummary
    (branches : Array OperationalMatrixFact) : SelectedMatrixSummary :=
  match branches[0]? with
  | none => {
      uniformSchema := none
      relationFree := false
      sharedLastPublicIdentity := none
      sharedFirstRelationPublicIdentity := none
    }
  | some first =>
      let schema := operationalUniformSchema first
      let uniform := branches.all fun branch => operationalUniformSchema branch == schema
      let relationFree := branches.all fun branch => !matrixFactHasRelation branch
      let lastIdentity := boundaryLastPublicIdentity? first
      let sharedLast := if lastIdentity.isSome &&
          branches.all (fun branch => match lastIdentity, boundaryLastPublicIdentity? branch with
            | some expected, some actual => publicIdentityTemplateEqual expected actual
            | _, _ => false) then
        lastIdentity else none
      let relationIdentity := boundaryFirstRelationPublicIdentity? first
      let sharedRelation := if relationIdentity.isSome && branches.all
          (fun branch => match relationIdentity, boundaryFirstRelationPublicIdentity? branch with
            | some expected, some actual => publicIdentityTemplateEqual expected actual
            | _, _ => false) then
        relationIdentity else none
      {
        uniformSchema := if uniform then some schema else none
        relationFree
        sharedLastPublicIdentity := sharedLast
        sharedFirstRelationPublicIdentity := sharedRelation
      }

/-- Recompute every identity-sensitive envelope field after deterministic selection
instantiation. `source.uniformSchema` is the prior all-branches proof; the representative is used
only to calculate the transformed schema, never to infer uniformity by itself. -/
private def selectedMatrixSummaryAfterInstantiation
    (source : SelectedMatrixSummary)
    (representative : OperationalMatrixFact) : Option SelectedMatrixSummary := do
  let _ ← source.uniformSchema
  let recomputed := selectedMatrixSummary #[representative]
  if recomputed.relationFree != source.relationFree then none
  else some {
    uniformSchema := some (operationalUniformSchema representative)
    relationFree := source.relationFree
    sharedLastPublicIdentity := boundaryLastPublicIdentity? representative
    sharedFirstRelationPublicIdentity := boundaryFirstRelationPublicIdentity? representative
  }

/-- Recompute a deterministic operation's complete summary from its transformed representative.
The source summary supplies the prior all-branches uniformity proof; without it this operation is
not permitted to create an envelope. -/
private def recomputeSelectedMatrixSummary
    (source : SelectedMatrixSummary)
    (representative : OperationalMatrixFact) : Option SelectedMatrixSummary := do
  let _ ← source.uniformSchema
  pure (selectedMatrixSummary #[representative])

private def selectedMatrixFamily
    (selection : OperationalValueOrigin)
    (branches : Array OperationalMatrixFact) : SelectedMatrixFamily := {
  selection
  branches
  count := branches.size
  representsLoopLanes := false
  summary := selectedMatrixSummary branches
}

private def selectedMatrixEnvelope
    (selection : OperationalValueOrigin)
    (count : Nat)
    (representative : OperationalMatrixFact)
    (summary : SelectedMatrixSummary := selectedMatrixSummary #[representative])
    (representsLoopLanes : Bool := false) :
    SelectedMatrixFamily := {
  selection
  branches := #[representative]
  count
  representsLoopLanes
  summary
}

private def SelectedMatrixFamily.isEnvelope (family : SelectedMatrixFamily) : Bool :=
  family.count != family.branches.size

private def SelectedMatrixFamily.map
    (family : SelectedMatrixFamily)
    (selection : OperationalValueOrigin)
    (transform : OperationalMatrixFact → OperationalMatrixFact) : SelectedMatrixFamily :=
  let branches := family.branches.map transform
  if family.isEnvelope then
    match branches[0]? with
    | some representative => selectedMatrixEnvelope selection family.count representative
        (representsLoopLanes := family.representsLoopLanes)
    | none => selectedMatrixFamily selection branches
  else
    { selectedMatrixFamily selection branches with
      representsLoopLanes := family.representsLoopLanes }

private def primitiveOperationalPolynomial
    (origin : MatrixOriginIdentity)
    (matrixType : MatrixTypeExpr)
    (totalHardBound : OperationalBoundExpr)
    (role : OperationalFactorRole)
    (identity : Option PublicMatrixIdentity)
    (relations : List OperationalMatrixRelation)
    (metadata : OperationalMatrixMetadata) : OperationalPolynomial :=
  let summary := match role with
    | .bounded => some {
        matrixType
        hardBound := totalHardBound
        metadata
        provenance := [.primitive (.matrix origin)]
      }
    | .large => none
  let primitive := match identity with
    | some publicIdentity => OperationalPrimitiveIdentity.publicMatrix publicIdentity
    | none => .matrix origin
  let protections := if relations.isEmpty then [] else
    [OperationalCompressionProtection.relationOwner]
  [{
    coefficient := 1
    product := {
      factors := [{
        leaf := .primitive primitive
        inputType := matrixType
        outputType := matrixType
        role
        boundedSummary := summary
        protections
        relations
      }]
      modes := []
      outputType := matrixType
    }
  }]

private def OperationalMatrixFact.initializePrimitivePolynomial
    (fact : OperationalMatrixFact)
    (role : OperationalFactorRole) : OperationalMatrixFact := {
  fact with polynomial := (primitiveOperationalPolynomial fact.origin fact.matrixType
    fact.totalHardBound role fact.identity fact.relations fact.metadata)
}

private def OperationalMatrixFact.primitiveRole (fact : OperationalMatrixFact) :
    OperationalFactorRole :=
  match fact.polynomial.head? >>= fun term => term.product.factors.head? with
  | some factor => factor.role
  | none => .bounded

private def OperationalMatrixFact.refreshPrimitivePolynomial
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  fact.initializePrimitivePolynomial fact.primitiveRole

structure OperationalTrapdoorFact where
  subject : WireRef
  matrixType : MatrixTypeExpr
  matrixParams : Mxx.SamplerParams
  maximum : OperationalBoundExpr
  publicIdentity : PublicMatrixIdentity
  deriving BEq

structure OperationalIntegerFact where
  subject : WireRef
  origin : OperationalValueOrigin
  lower : Int
  upper : Int
  lowerExpression : OperationalBoundExpr
  upperExpression : OperationalBoundExpr
  deriving BEq

structure OperationalBytesFact where
  subject : WireRef
  origin : OperationalValueOrigin
  length : Int
  deriving BEq, DecidableEq, Repr

inductive OperationalFact where
  | matrix (fact : OperationalMatrixFact)
  | matrixExpr (root : OperationalExprId)
  | integer (fact : OperationalIntegerFact)
  | boolean
  | real
  | trapdoor (fact : OperationalTrapdoorFact)
  | familyUniform
      (binder : FamilyTemplateBinder)
      (binderCoordinate : Option LoopCoordinate)
      (element : OperationalFact)
      (count : Int)
  | familyPacked
      (elements : Array OperationalFact)
      (count : Nat)
      (matrixSummary : Option SelectedMatrixSummary)
  | selectedMatrices (family : SelectedMatrixFamily)
  | bytes (fact : OperationalBytesFact)
  | typedBlob (typeName : String)
  | unknown (wireType : WireTypeExpr)
  deriving BEq

private def packedMatrixSummary? (elements : Array OperationalFact) : Option SelectedMatrixSummary := do
  let matrices ← elements.mapM fun element => match element with
    | .matrix matrix => some matrix
    | _ => none
  let summary := selectedMatrixSummary matrices
  if summary.uniformSchema.isSome then some summary else none

private def packedOperationalFamily
    (elements : Array OperationalFact)
    (count : Nat := elements.size) : OperationalFact :=
  .familyPacked elements count (packedMatrixSummary? elements)

abbrev OperationalState := Array OperationalFact

/-- Structure-only information validated once before numeric operational requests are evaluated. -/
structure PreparedOperationalScope where
  scope : Scope
  derivation : ScopeDerivation
  inputIndices : Array (Option Nat)
  definitionIndices : Array (Option Nat)
  attachmentBuckets : Array (Array DerivationAttachment)

/-- One frozen program with all scope/derivation alignment resolved to array indices. -/
structure PreparedOperationalProgram where
  root : PreparedOperationalScope
  definitions : Array (String × PreparedOperationalScope)

inductive OperationalError where
  | inScope (scope : ScopeTemplateKey) (error : OperationalError)
  | missingOutputType (node : Nat) (port : Nat)
  | missingOperand (node : Nat) (operand : WireRef)
  | operandNotMatrix (node : Nat) (operand : WireRef)
  | operandNotInteger (node : Nat) (operand : WireRef)
  | operandNotBoolean (node : Nat) (operand : WireRef)
  | operandNotReal (node : Nat) (operand : WireRef)
  | invalidMatrixParameters (node : Nat)
  | flat (node : Nat) (error : OperationalFlatError)
  | invalidBound (node : Nat) (bound : Int)
  | invalidCount (node : Nat) (count : Int)
  | missingGadgetLayout (node : Nat)
  | ambiguousGadgetLayout (node : Nat)
  | invalidGadgetLayout (node : Nat)
  | gadgetLayoutMismatch (node : Nat)
  | missingPublicIdentity (node : Nat) (wire : WireRef)
  | publicIdentityMismatch (node : Nat)
  | missingRelation (node : Nat) (wire : WireRef)
  | ambiguousRelation (node : Nat) (wire : WireRef)
  | unavailableRelation (node : Nat) (wire : WireRef)
  | malformedRelation (node : Nat)
  | missingDefinition (name : String)
  | definitionFuelExhausted
  | childInputMismatch (node expected actual : Nat)
  | duplicateInputName (name : String)
  | missingInputNode (name : String)
  | unexpectedInputNode (name : String)
  | missingChildOutput (node port : Nat)
  | loopInputModeMismatch (node argument : Nat)
  | selectedFamilyOperationUnsupported (node : Nat)
  | relationBearingCarriedValue (scope : ScopeTemplateKey) (node slot : Nat)
  | sequentialSchemaMismatch
      (scope : ScopeTemplateKey)
      (node slot : Nat)
      (initialLargeCounts outputLargeCounts : List Nat)
  | divisionByZero
  | negativeDenominator (value : Int)
  | invalidPreviousPath (path : OperationalBoundPath)
  | nonClosedExpression
  | derivation (error : DerivationError)
  | unsupportedOutputArity (node : Nat) (actual : Nat)
  | outputTypeMismatch (node : Nat)
  | missingStageDerivation (stage : String)
  | missingStageResult (stage output : String)
  | missingProtocolContract (name : String)
  | inputContractMismatch (name : String)
  | unknownDerivationAttachment (ownerNamespace ruleName : String)
  | missingDerivationAttachmentRole (ownerNamespace ruleName roleName : String)
  | invalidDerivationAttachment (ownerNamespace ruleName : String)
  | invalidOperationalExprRef (id : Nat)
  | operationalExprTypeMismatch (left right : Nat)
  | unsupportedOperationalExpr (id : Nat)
  | unsupportedNode (node : Nat)
  deriving BEq, DecidableEq, Repr

/-! ## Selection-preserving operational expressions

The executable checker still evaluates ordinary operations into the flat facts above.  This
request-local arena is the compact boundary for unresolved dynamic selections.  Arena indices are
allocation identities only: they are never matrix, relation, or symbolic-equality evidence. -/

inductive OperationalSelectionBranches where
  | exact (branches : Array OperationalExprId)
  | schemaEnvelope
      (count : Nat)
      (representative : OperationalExprId)
      (summary : SelectedMatrixSummary)
  deriving BEq

inductive OperationalMatrixExprNode where
  | concrete (fact : OperationalMatrixFact)
  | add (left right : OperationalExprId)
  | subtract (left right : OperationalExprId)
  | multiply (left right : OperationalExprId)
  | tensor (left right : OperationalExprId)
  | concat (axis : ConcatAxis) (inputs : Array OperationalExprId)
  | transform (operation : OperationalFactorTransform) (value : OperationalExprId)
  | select
      (selection : DynamicSelectionIdentity)
      (branches : OperationalSelectionBranches)
  deriving BEq

structure OperationalMatrixExpr where
  matrixType : MatrixTypeExpr
  node : OperationalMatrixExprNode
  deriving BEq

structure OperationalExprArena where
  nodes : Array OperationalMatrixExpr := #[]
  deriving BEq

private structure OperationalExprEvaluationStats where
  evaluations : Nat := 0
  memoHits : Nat := 0
  memoMisses : Nat := 0
  deriving BEq, DecidableEq, Repr

private structure OperationalExprEvaluationState where
  memo : Array (Option Int)
  stats : OperationalExprEvaluationStats := {}
  deriving BEq

/-- Facts for one frozen scope and the append-only expression arena shared by its request.  Wire
facts remain indexed by exact `(node, port)` locations; expression IDs are valid only in `arena`. -/
structure OperationalScopeFacts where
  values : Array (Array OperationalFact) := #[]
  arena : OperationalExprArena := {}

private def OperationalExprArena.get?
    (arena : OperationalExprArena) (id : OperationalExprId) : Option OperationalMatrixExpr :=
  arena.nodes[id]?

private def OperationalExprArena.push
    (arena : OperationalExprArena)
    (expression : OperationalMatrixExpr) : OperationalExprArena × OperationalExprId :=
  ({ arena with nodes := arena.nodes.push expression }, arena.nodes.size)

private def OperationalExprArena.pushConcrete
    (arena : OperationalExprArena)
    (fact : OperationalMatrixFact) : OperationalExprArena × OperationalExprId :=
  arena.push { matrixType := fact.matrixType, node := .concrete fact }

private def OperationalExprArena.pushMatrixFact
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | .matrix fact => pure (arena.pushConcrete fact)
  | .matrixExpr root =>
      match arena.get? root with
      | some _ => pure (arena, root)
      | none => throw (.invalidOperationalExprRef root)
  | _ => throw (.unsupportedOperationalExpr arena.nodes.size)

private def OperationalExprArena.concreteFact
    (arena : OperationalExprArena)
    (id : OperationalExprId) : Except OperationalError OperationalMatrixFact := do
  match arena.get? id with
  | some { node := .concrete fact, .. } => pure fact
  | some _ => throw (.unsupportedOperationalExpr id)
  | none => throw (.invalidOperationalExprRef id)

private def OperationalExprArena.checkedType
    (arena : OperationalExprArena)
    (first : OperationalExprId)
    (remaining : Array OperationalExprId) : Except OperationalError MatrixTypeExpr := do
  let firstExpr ← match arena.get? first with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef first)
  for id in remaining do
    let expression ← match arena.get? id with
      | some expression => pure expression
      | none => throw (.invalidOperationalExprRef id)
    if expression.matrixType ≠ firstExpr.matrixType then
      throw (.operationalExprTypeMismatch first id)
  pure firstExpr.matrixType

private def OperationalExprArena.pushSelect
    (arena : OperationalExprArena)
    (selection : DynamicSelectionIdentity)
    (branches : OperationalSelectionBranches) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  match branches with
  | .exact values =>
      let first ← match values[0]? with
        | some first => pure first
        | none => throw (.invalidCount 0 0)
      let matrixType ← arena.checkedType first (values.extract 1 values.size)
      if values.all (· == first) then pure (arena, first)
      else pure (arena.push { matrixType, node := .select selection branches })
  | .schemaEnvelope count representative summary =>
      if count = 0 then throw (.invalidCount 0 0)
      let expression ← match arena.get? representative with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef representative)
      let fact ← match expression.node with
        | .concrete fact => pure fact
        | _ => throw (.unsupportedOperationalExpr representative)
      if summary.uniformSchema != some (operationalUniformSchema fact) ||
          summary.relationFree != !matrixFactHasRelation fact ||
          summary.sharedLastPublicIdentity != boundaryLastPublicIdentity? fact ||
          summary.sharedFirstRelationPublicIdentity !=
            boundaryFirstRelationPublicIdentity? fact then
        throw (.unsupportedOperationalExpr representative)
      pure (arena.push { matrixType := expression.matrixType, node := .select selection branches })

private def OperationalExprArena.pushExactSelection
    (arena : OperationalExprArena)
    (selection : DynamicSelectionIdentity)
    (branches : Array OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let mut arena := arena
  let mut ids : Array OperationalExprId := #[]
  for branch in branches do
    let (nextArena, id) ← arena.pushMatrixFact branch
    arena := nextArena
    ids := ids.push id
  arena.pushSelect selection (.exact ids)

private def OperationalSelectionBranches.staticBranch
    (branches : OperationalSelectionBranches)
    (index : Nat) : Except OperationalError OperationalExprId :=
  match branches with
  | .exact values => match values[index]? with
      | some value => pure value
      | none => throw (.invalidCount 0 index)
  | .schemaEnvelope _ representative _ => throw (.unsupportedOperationalExpr representative)

private def OperationalExprEvaluationState.empty
    (arena : OperationalExprArena) : OperationalExprEvaluationState := {
  memo := Array.replicate arena.nodes.size none
}

/-- The operational checker has an explicit transfer category for every executable IR node.
This definition and the exhaustive classifiers below are deliberately separate from the transfer
implementation: adding an IR constructor must first make this file fail to compile, rather than
silently reaching the conservative fallback. -/
inductive OperationalTransferClass where
  | input
  | scalar
  | matrix
  | structural
  deriving BEq, DecidableEq

private def classifyIntBinary : IntBinaryOp → Unit
  | .add | .subtract | .multiply | .divide | .remainder => ()

private def classifyIntCompare : IntCompareOp → Unit
  | .equal | .less | .lessEqual => ()

private def classifyRealBinary : RealBinaryOp → Unit
  | .add | .subtract | .multiply | .divide => ()

private def classifyConcatAxis : ConcatAxis → Unit
  | .rows | .columns | .diagonal => ()

private def classifyHashVariant : Mxx.HashVariant → Unit
  | .plain | .decomposed | .smallDecomposed => ()

private def classifyLoopInputMode : LoopInputMode → Unit
  | .broadcast | .zip | .zipOffset _ => ()

/-- Exhaustive compile-time inventory of the operational transfer surface. Nested operation enums
are themselves classified exhaustively, so extending (for example) `IntBinaryOp` or `ConcatAxis`
also requires an explicit operational-checker decision. -/
def operationalTransferClass : NodeKind → OperationalTransferClass
  | .input _ => .input
  | .constantInt _
  | .evaluateInt _
  | .constantReal _
  | .constantBool _
  | .boolToInt
  | .intToReal
  | .realSqrt
  | .bitExtract _
  | .extractCoefficient _ => .scalar
  | .intBinary operation =>
      let _ := classifyIntBinary operation
      .scalar
  | .intCompare operation =>
      let _ := classifyIntCompare operation
      .scalar
  | .realBinary operation =>
      let _ := classifyRealBinary operation
      .scalar
  | .zeroMatrix _
  | .identityMatrix _
  | .constantMatrix _ _
  | .unitRowMatrix _ _
  | .unitColumnMatrix _ _
  | .gadgetMatrix _ _
  | .smallGadgetMatrix _ _
  | .powerOfBaseMatrix _ _ _
  | .rotationMatrix _ _
  | .gadgetTrapdoor _ _
  | .constantCoefficient _
  | .uniformResidueSample _
  | .uniformIntervalSample _ _ _
  | .gaussianSample _ _
  | .gadgetDecompose _ _ _ _
  | .trapdoorSample _ _
  | .trapdoorPublic
  | .preimageSample _ _
  | .matrixAdd
  | .matrixSubtract
  | .matrixMultiply
  | .matrixNegate
  | .matrixScale _
  | .transpose
  | .slice _ _
  | .tensor
  | .reshape _ _
  | .thresholdDecodeBool _ _ _
  | .thresholdDecodeInt _ _ _
  | .crtRecompose _ _
  | .packPolynomialCoefficients _ _ => .matrix
  | .hashSample _ variant _ _ _ _ _ _ =>
      let _ := classifyHashVariant variant
      .matrix
  | .concat axis =>
      let _ := classifyConcatAxis axis
      .matrix
  | .select
  | .familyPack
  | .familyGetStatic _
  | .familyGetDynamic
  | .subgraphCall _ _
  | .sequentialLoop _ _ _ _ _ => .structural
  | .parallelLoop _ _ _ _ inputModes =>
      let _ := inputModes.map classifyLoopInputMode
      .structural

def absolute (value : Int) : Int := if value < 0 then -value else value

def capCentered (modulus value : Int) : Int :=
  if modulus ≤ 0 then 0 else min (modulus / 2) (absolute value)

def matrixCap (matrixType : MatrixTypeExpr) (environment : ParamEnvironment) : Option Int := do
  let modulus ← matrixType.modulus.evaluate environment
  if modulus ≤ 0 then none else some (modulus / 2)

def matrixRingDimension (matrixType : MatrixTypeExpr) (environment : ParamEnvironment) : Option Int := do
  let value ← matrixType.ringDimension.evaluate environment
  if value < 0 then none else some value

def resolveGadgetLayout
    (node : Nat)
    (layouts : List Mxx.GadgetLayoutDescriptor)
    (params : Mxx.SamplerParams) : Except OperationalError Mxx.GadgetLayoutDescriptor := do
  let candidates := layouts.filter fun descriptor => descriptor.matches params
  match candidates with
  | [descriptor] =>
      if descriptor.valid then pure descriptor else throw (.invalidGadgetLayout node)
  | [] => throw (.missingGadgetLayout node)
  | _ => throw (.ambiguousGadgetLayout node)

private def factClosedMaximum : OperationalFact → Option Int
  | .matrix fact => match fact.totalHardBound with
      | .closedInt (.constant maximum) => some maximum
      | _ => none
  | .trapdoor fact => match fact.maximum with
      | .closedInt (.constant maximum) => some maximum
      | _ => none
  | .familyUniform _ _ element _ => factClosedMaximum element
  | _ => none

structure OperationalNumericSlot where
  matrixMaximum : Option Int := none
  integerLower : Option Int := none
  integerUpper : Option Int := none
  deriving Inhabited

abbrev OperationalNumericState := Array OperationalNumericSlot

def factNumericSlot : OperationalFact → OperationalNumericSlot
  | fact@(.matrix _) | fact@(.trapdoor _) | fact@(.familyUniform _ _ _ _) =>
      { matrixMaximum := factClosedMaximum fact }
  | .integer fact => { integerLower := some fact.lower, integerUpper := some fact.upper }
  | _ => {}

private def lookupPrevious
    (states : List OperationalNumericState) : OperationalBoundPath → Option Int
  | .matrixMaximum depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.matrixMaximum)
  | .integerLower depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.integerLower)
  | .integerUpper depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.integerUpper)

private def operationalBoundPathSlot : OperationalBoundPath → Nat
  | .matrixMaximum _ slot | .integerLower _ slot | .integerUpper _ slot => slot

private def operationalBoundPathAtCurrentDepth : OperationalBoundPath → Bool
  | .matrixMaximum depth _ | .integerLower depth _ | .integerUpper depth _ => depth = 0

private def numericStateFromComponents
    (paths : List OperationalBoundPath)
    (values : List Int) : Except OperationalError OperationalNumericState := do
  if paths.length != values.length then
    throw (.unsupportedOutputArity values.length paths.length)
  if paths.any fun path => !operationalBoundPathAtCurrentDepth path then
    throw (.invalidPreviousPath (paths.find? (fun path =>
      !operationalBoundPathAtCurrentDepth path) |>.getD (.matrixMaximum 0 0)))
  let size := paths.foldl (fun current path => max current (operationalBoundPathSlot path + 1)) 0
  let mut result : OperationalNumericState := Array.replicate size {}
  for (path, value) in paths.zip values do
    let slot := operationalBoundPathSlot path
    let previous := result[slot]!
    result := result.set! slot <| match path with
      | .matrixMaximum _ _ => { previous with matrixMaximum := some value }
      | .integerLower _ _ => { previous with integerLower := some value }
      | .integerUpper _ _ => { previous with integerUpper := some value }
  pure result

private def factMaximumExpr : OperationalFact → Option OperationalBoundExpr
  | .matrix fact => some fact.totalHardBound
  | .trapdoor fact => some fact.maximum
  | .familyUniform _ _ element _ => factMaximumExpr element
  | _ => none

private def factNumericExpressions
    (slot : Nat) : OperationalFact → List (OperationalBoundPath × OperationalBoundExpr)
  | .matrix fact => [(.matrixMaximum 0 slot, fact.totalHardBound)]
  | .trapdoor fact => [(.matrixMaximum 0 slot, fact.maximum)]
  | .integer fact => [
      (.integerLower 0 slot, fact.lowerExpression),
      (.integerUpper 0 slot, fact.upperExpression)
    ]
  | .familyUniform _ _ element _ => factNumericExpressions slot element
  | _ => []

private def abstractCarriedMaximum (slot : Nat) : OperationalFact → OperationalFact
  | .matrix fact => .matrix {
      fact with totalHardBound := .previous (.matrixMaximum 0 slot) }
  | .trapdoor fact => .trapdoor { fact with maximum := .previous (.matrixMaximum 0 slot) }
  | .integer fact => .integer {
      fact with
      lowerExpression := .previous (.integerLower 0 slot)
      upperExpression := .previous (.integerUpper 0 slot)
    }
  | .familyUniform binder coordinate element count =>
      .familyUniform binder coordinate (abstractCarriedMaximum slot element) count
  | fact => fact

private def setFactMaximum (maximum : Int) : OperationalFact → OperationalFact
  | .matrix fact => .matrix {
      fact with totalHardBound := .closedInt (.constant maximum) }
  | .trapdoor fact => .trapdoor { fact with maximum := .closedInt (.constant maximum) }
  | .familyUniform binder coordinate element count =>
      .familyUniform binder coordinate (setFactMaximum maximum element) count
  | fact => fact

private def setFactMaximumExpr
    (maximum : OperationalBoundExpr) : OperationalFact → OperationalFact
  | .matrix fact => .matrix { fact with totalHardBound := maximum }
  | .trapdoor fact => .trapdoor { fact with maximum }
  | .familyUniform binder coordinate element count =>
      .familyUniform binder coordinate (setFactMaximumExpr maximum element) count
  | fact => fact

private def sameCarriedSchema : OperationalFact → OperationalFact → Bool
  | .matrix left, .matrix right =>
      left.matrixParams.modulus == right.matrixParams.modulus &&
      left.matrixParams.ringDimension == right.matrixParams.ringDimension &&
      left.matrixParams.rows == right.matrixParams.rows &&
      left.matrixParams.columns == right.matrixParams.columns &&
      left.canonicalRange == right.canonicalRange &&
      left.identity.isNone && right.identity.isNone &&
      left.relations.isEmpty && right.relations.isEmpty &&
      left.polynomial.map operationalLargeFactorCount ==
        right.polynomial.map operationalLargeFactorCount
  | .familyUniform _ leftCoordinate left leftCount,
      .familyUniform _ rightCoordinate right rightCount =>
      -- Producer/binder identities are values, not schema: each loop body necessarily creates a
      -- different family producer than the initial carried family.  Uniform-vs-nonuniform shape,
      -- count, and element schema are the invariant parts.
      leftCoordinate.isSome == rightCoordinate.isSome &&
        leftCount == rightCount && sameCarriedSchema left right
  | .integer _, .integer _ => true
  | .boolean, .boolean | .real, .real => true
  | _, _ => false

private def carriedLargeFactorCounts : OperationalFact → List Nat
  | .matrix fact => fact.polynomial.map operationalLargeFactorCount
  | .familyUniform _ _ element _ => carriedLargeFactorCounts element
  | _ => []

def intExprIsClosed : IntExpr → Bool
  | .constant _ => true
  | .parameter _ => true
  | .loopIndex _ => false
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprIsClosed left && intExprIsClosed right
  | .log2Ceil value => intExprIsClosed value

private def intExprUsesLoop (slot : Nat) : IntExpr → Bool
  | .constant _ | .parameter _ => false
  | .loopIndex candidate => candidate == slot
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprUsesLoop slot left || intExprUsesLoop slot right
  | .log2Ceil value => intExprUsesLoop slot value

private def intExprUsesParameter (name : String) : IntExpr → Bool
  | .constant _ | .loopIndex _ => false
  | .parameter candidate => candidate == name
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprUsesParameter name left || intExprUsesParameter name right
  | .log2Ceil value => intExprUsesParameter name value

def replaceLoopIndex
    (environment : ParamEnvironment) (slot : Nat) (value : Nat) : ParamEnvironment :=
  (ParamKey.loopIndex slot, ParamValue.integer (Int.ofNat value)) ::
    environment.filter fun entry => entry.1 != .loopIndex slot

def replaceParameter
    (environment : ParamEnvironment) (name : String) (value : Int) : ParamEnvironment :=
  (ParamKey.parameter name, ParamValue.integer value) ::
    environment.filter fun entry => entry.1 != .parameter name

/-- Evaluates only the numeric expression over the loop coordinates it actually references.
This never reevaluates an IR scope and allocates no lane facts. -/
private def evaluateIntOverLoops
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError (List Int) := do
  let rec visit (environment : ParamEnvironment) : List OperationalParameterDomain →
      Except OperationalError (List Int)
    | [] => match expression.evaluate environment with
        | some value => pure [value]
        | none => throw .nonClosedExpression
    | .loopIndex slot count :: tail =>
        if !intExprUsesLoop slot expression || count = 0 then visit environment tail else
          return (← (List.range count).mapM fun index =>
            visit (replaceLoopIndex environment slot index) tail).flatten
    | .parameter name sourceEnvironment sourceDomains sourceExpression :: tail =>
        if !intExprUsesParameter name expression then visit environment tail else do
          let values ← evaluateIntOverLoops sourceEnvironment sourceDomains sourceExpression
          return (← values.mapM fun value =>
            visit (replaceParameter environment name value) tail).flatten
  visit environment domains

private def evaluateIntMinimum
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail => pure (tail.foldl min first)

private def evaluateIntMaximum
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail => pure (tail.foldl max first)

private def evaluateIntMaximumAbsolute
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  let values ← evaluateIntOverLoops environment domains expression
  pure (values.foldl (fun maximum value => max maximum (absolute value)) 0)

def evaluateIntInvariant
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail =>
      if tail.all (· == first) then pure first else throw .nonClosedExpression

private def extendParameterDomains
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (bindings : List (String × IntExpr)) : Except OperationalError (List OperationalParameterDomain) := do
  let mut result := domains
  for (name, expression) in bindings do
    result := .parameter name environment domains expression :: result.filter fun domain => match domain with
      | .parameter candidate _ _ _ => candidate != name
      | .loopIndex _ _ => true
  pure result

def instantiateParameterDomains (slot index : Nat) :
    List OperationalParameterDomain → List OperationalParameterDomain
  | [] => []
  | .loopIndex candidate count :: tail =>
      if candidate = slot then instantiateParameterDomains slot index tail
      else .loopIndex candidate count :: instantiateParameterDomains slot index tail
  | .parameter name environment domains expression :: tail =>
      .parameter name (replaceLoopIndex environment slot index)
          (instantiateParameterDomains slot index domains) expression ::
        instantiateParameterDomains slot index tail

def materializeInvariantParameters
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain) : Except OperationalError ParamEnvironment := do
  let mut result := environment
  for domain in domains do
    match domain with
    | .loopIndex _ _ => pure ()
    | .parameter name sourceEnvironment sourceDomains sourceExpression =>
        let value ← evaluateIntInvariant sourceEnvironment sourceDomains sourceExpression
        result := replaceParameter result name value
  pure result

@[simp] theorem materializeInvariantParameters_nil (environment : ParamEnvironment) :
    materializeInvariantParameters environment [] = .ok environment := by
  rfl

private def instantiateBoundLoopIndex (slot index : Nat) : OperationalBoundExpr → OperationalBoundExpr
  | .closedInt value => .closedInt value
  | .contextual kind environment domains value =>
      .contextual kind (replaceLoopIndex environment slot index)
        (instantiateParameterDomains slot index domains)
        value
  | .previous path => .previous path
  | .negate value => .negate (instantiateBoundLoopIndex slot index value)
  | .add left right => .add (instantiateBoundLoopIndex slot index left)
      (instantiateBoundLoopIndex slot index right)
  | .subtract left right => .subtract (instantiateBoundLoopIndex slot index left)
      (instantiateBoundLoopIndex slot index right)
  | .multiply left right => .multiply (instantiateBoundLoopIndex slot index left)
      (instantiateBoundLoopIndex slot index right)
  | .divide left right => .divide (instantiateBoundLoopIndex slot index left)
      (instantiateBoundLoopIndex slot index right)
  | .minimum left right => .minimum (instantiateBoundLoopIndex slot index left)
      (instantiateBoundLoopIndex slot index right)
  | .maximum left right => .maximum (instantiateBoundLoopIndex slot index left)
      (instantiateBoundLoopIndex slot index right)
  | .centeredCap modulus value => .centeredCap (instantiateBoundLoopIndex slot index modulus)
      (instantiateBoundLoopIndex slot index value)
  | .matrixProduct ringDimension innerDimension left right =>
      .matrixProduct (instantiateBoundLoopIndex slot index ringDimension)
        (instantiateBoundLoopIndex slot index innerDimension)
        (instantiateBoundLoopIndex slot index left) (instantiateBoundLoopIndex slot index right)
  | .recurrence count initial transition outputSlot =>
      .recurrence count (initial.map (instantiateBoundLoopIndex slot index))
        (transition.map (instantiateBoundLoopIndex slot index)) outputSlot
  | .recurrenceState count paths initial transition output =>
      .recurrenceState count paths (initial.map (instantiateBoundLoopIndex slot index))
        (transition.map (instantiateBoundLoopIndex slot index)) output

private def shiftPreviousDepthFrom (cutoff : Nat) : OperationalBoundExpr → OperationalBoundExpr
  | .closedInt value => .closedInt value
  | .contextual kind environment domains value => .contextual kind environment domains value
  | .previous (.matrixMaximum depth slot) =>
      .previous (.matrixMaximum (if cutoff ≤ depth then depth + 1 else depth) slot)
  | .previous (.integerLower depth slot) =>
      .previous (.integerLower (if cutoff ≤ depth then depth + 1 else depth) slot)
  | .previous (.integerUpper depth slot) =>
      .previous (.integerUpper (if cutoff ≤ depth then depth + 1 else depth) slot)
  | .negate value => .negate (shiftPreviousDepthFrom cutoff value)
  | .add left right => .add (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .subtract left right => .subtract (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .multiply left right => .multiply (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .divide left right => .divide (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .minimum left right => .minimum (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .maximum left right => .maximum (shiftPreviousDepthFrom cutoff left)
      (shiftPreviousDepthFrom cutoff right)
  | .centeredCap modulus value => .centeredCap (shiftPreviousDepthFrom cutoff modulus)
      (shiftPreviousDepthFrom cutoff value)
  | .matrixProduct ringDimension innerDimension left right =>
      .matrixProduct (shiftPreviousDepthFrom cutoff ringDimension)
        (shiftPreviousDepthFrom cutoff innerDimension) (shiftPreviousDepthFrom cutoff left)
        (shiftPreviousDepthFrom cutoff right)
  | .recurrence count initial transition slot =>
      .recurrence count (initial.map (shiftPreviousDepthFrom cutoff))
        (transition.map (shiftPreviousDepthFrom (cutoff + 1))) slot
  | .recurrenceState count paths initial transition output =>
      .recurrenceState count paths (initial.map (shiftPreviousDepthFrom cutoff))
        (transition.map (shiftPreviousDepthFrom (cutoff + 1))) output

private def shiftPreviousDepth := shiftPreviousDepthFrom 0

private def OperationalBoundExpr.usesPrevious : OperationalBoundExpr → Bool
  | .closedInt _ | .contextual .. => false
  | .previous _ => true
  | .negate value => value.usesPrevious
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .minimum left right | .maximum left right | .centeredCap left right =>
      left.usesPrevious || right.usesPrevious
  | .matrixProduct ringDimension innerDimension left right =>
      ringDimension.usesPrevious || innerDimension.usesPrevious || left.usesPrevious ||
        right.usesPrevious
  | .recurrence .. | .recurrenceState .. => true

def OperationalBoundExpr.evaluateWithStates
    (environment : ParamEnvironment)
    (previousStates : List OperationalNumericState) : OperationalBoundExpr → Except OperationalError Int
  | .closedInt value => do
      if !intExprIsClosed value then throw .nonClosedExpression
      match value.evaluate environment with
      | some result => pure result
      | none => throw .nonClosedExpression
  | .contextual kind contextualEnvironment domains value =>
      match kind with
      | .minimum => evaluateIntMinimum contextualEnvironment domains value
      | .maximum => evaluateIntMaximum contextualEnvironment domains value
      | .maximumAbsolute => evaluateIntMaximumAbsolute contextualEnvironment domains value
  | .previous path =>
      match lookupPrevious previousStates path with
      | some result => pure result
      | none => throw (.invalidPreviousPath path)
  | .negate value => return -(← value.evaluateWithStates environment previousStates)
  | .add left right => return (← left.evaluateWithStates environment previousStates) +
      (← right.evaluateWithStates environment previousStates)
  | .subtract left right => return (← left.evaluateWithStates environment previousStates) -
      (← right.evaluateWithStates environment previousStates)
  | .multiply left right => return (← left.evaluateWithStates environment previousStates) *
      (← right.evaluateWithStates environment previousStates)
  | .divide left right => do
      let denominator ← right.evaluateWithStates environment previousStates
      if denominator = 0 then throw .divisionByZero
      if denominator < 0 then throw (.negativeDenominator denominator)
      return (← left.evaluateWithStates environment previousStates) / denominator
  | .minimum left right => do
      let left ← left.evaluateWithStates environment previousStates
      let right ← right.evaluateWithStates environment previousStates
      return min left right
  | .maximum left right => do
      let left ← left.evaluateWithStates environment previousStates
      let right ← right.evaluateWithStates environment previousStates
      return max left right
  | .centeredCap modulus value => do
      let modulus ← modulus.evaluateWithStates environment previousStates
      let value ← value.evaluateWithStates environment previousStates
      return capCentered modulus value
  | .matrixProduct ringDimension innerDimension left right => do
      let ringDimension ← ringDimension.evaluateWithStates environment previousStates
      let innerDimension ← innerDimension.evaluateWithStates environment previousStates
      let left ← left.evaluateWithStates environment previousStates
      let right ← right.evaluateWithStates environment previousStates
      return ringDimension * innerDimension * left * right
  | .recurrence count initial transition slot => do
      if initial.length != transition.length then
        throw (.unsupportedOutputArity transition.length initial.length)
      let initialValues ← initial.mapM (OperationalBoundExpr.evaluateWithStates environment previousStates)
      let initialState : OperationalNumericState := initialValues.map
        (fun value => { matrixMaximum := some value }) |>.toArray
      let rec iterate : Nat → OperationalNumericState → Except OperationalError OperationalNumericState
        | 0, state => pure state
        | remaining + 1, state => do
            let values ← transition.mapM
              (OperationalBoundExpr.evaluateWithStates environment (state :: previousStates))
            iterate remaining (values.map (fun value => { matrixMaximum := some value }) |>.toArray)
      let finalState ← iterate count initialState
      match finalState[slot]? >>= (·.matrixMaximum) with
      | some value => pure value
      | none => throw (.invalidPreviousPath (.matrixMaximum 0 slot))
  | .recurrenceState count paths initial transition output => do
      if paths.length != initial.length || initial.length != transition.length then
        throw (.unsupportedOutputArity transition.length initial.length)
      let initialValues ← initial.mapM
        (OperationalBoundExpr.evaluateWithStates environment previousStates)
      let initialState ← numericStateFromComponents paths initialValues
      let rec iterateState : Nat → OperationalNumericState →
          Except OperationalError OperationalNumericState
        | 0, state => pure state
        | remaining + 1, state => do
            let values ← transition.mapM
              (OperationalBoundExpr.evaluateWithStates environment (state :: previousStates))
            iterateState remaining (← numericStateFromComponents paths values)
      let finalState ← iterateState count initialState
      match lookupPrevious [finalState] output with
      | some value => pure value
      | none => throw (.invalidPreviousPath output)

@[simp] theorem OperationalBoundExpr.evaluateWithStates_closedConstant
    (environment : ParamEnvironment)
    (previousStates : List OperationalNumericState)
    (value : Int) :
    OperationalBoundExpr.evaluateWithStates environment previousStates
      (.closedInt (.constant value)) = .ok value := by
  simp [OperationalBoundExpr.evaluateWithStates, intExprIsClosed, IntExpr.evaluate]
  rfl

@[simp] theorem OperationalBoundExpr.evaluateWithStates_contextualMaximum_nil
    (environment : ParamEnvironment)
    (previousStates : List OperationalNumericState)
    (value : IntExpr)
    (result : Int)
    (evaluates : value.evaluate environment = some result) :
    OperationalBoundExpr.evaluateWithStates environment previousStates
      (.contextual .maximum environment [] value) = .ok result := by
  simp [OperationalBoundExpr.evaluateWithStates, evaluateIntMaximum, evaluateIntOverLoops,
    evaluateIntOverLoops.visit, evaluates]
  rfl

def OperationalBoundExpr.evaluate
    (environment : ParamEnvironment)
    (previousState : OperationalState)
    (expression : OperationalBoundExpr) : Except OperationalError Int :=
  expression.evaluateWithStates environment [previousState.map factNumericSlot]

private def evaluateOperationalExprBoundWithFuel
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) : Nat →
    Except OperationalError (Int × OperationalExprEvaluationState)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => match state.memo[id]? with
  | none => throw (.invalidOperationalExprRef id)
  | some (some value) =>
      pure (value, { state with stats := {
        state.stats with memoHits := state.stats.memoHits + 1
      } })
  | some none => do
      let expression ← match arena.get? id with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef id)
      let state := { state with stats := {
        state.stats with
        evaluations := state.stats.evaluations + 1
        memoMisses := state.stats.memoMisses + 1
      } }
      let (value, state) ← match expression.node with
        | .concrete fact => do
            let value ← match fact.totalHardBound with
              | .closedInt (.constant value) => pure value
              | expression => expression.evaluateWithStates environment []
            pure (value, state)
        | .select _ branches =>
            let branchIds ← match branches with
              | .exact values => pure values
              | .schemaEnvelope count representative _ =>
                  if count = 0 then throw (.invalidCount 0 0)
                  pure #[representative]
            let first ← match branchIds[0]? with
              | some first => pure first
              | none => throw (.invalidCount 0 0)
            let (firstBound, state) ←
              evaluateOperationalExprBoundWithFuel arena environment first state fuel
            let mut maximum := firstBound
            let mut state := state
            for branch in branchIds.extract 1 branchIds.size do
              let (bound, nextState) ←
                evaluateOperationalExprBoundWithFuel arena environment branch state fuel
              maximum := max maximum bound
              state := nextState
            pure (maximum, state)
        | .add .. | .subtract .. | .multiply .. | .tensor .. | .concat .. | .transform .. =>
            throw (.unsupportedOperationalExpr id)
      let memo := state.memo.set! id (some value)
      pure (value, { state with memo })

private def evaluateOperationalExprBound
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :
    Except OperationalError (Int × OperationalExprEvaluationState) :=
  evaluateOperationalExprBoundWithFuel arena environment id state (arena.nodes.size + 1)

private def setFactRecurrenceState
    (count : Nat)
    (paths : List OperationalBoundPath)
    (initial transition : List OperationalBoundExpr)
    (slot : Nat)
    (environment : ParamEnvironment) : OperationalFact → Except OperationalError OperationalFact
  | .matrix fact =>
      let maximum := OperationalBoundExpr.recurrenceState
        count paths initial transition (.matrixMaximum 0 slot)
      pure (.matrix { fact with totalHardBound := maximum })
  | .trapdoor fact =>
      let maximum := OperationalBoundExpr.recurrenceState
        count paths initial transition (.matrixMaximum 0 slot)
      pure (.trapdoor { fact with maximum })
  | .integer fact => do
      let lowerExpression :=
        .recurrenceState count paths initial transition (.integerLower 0 slot)
      let upperExpression :=
        .recurrenceState count paths initial transition (.integerUpper 0 slot)
      let lower ← lowerExpression.evaluateWithStates environment []
      let upper ← upperExpression.evaluateWithStates environment []
      if lower > upper then throw (.invalidBound slot lower)
      pure (.integer { fact with lower, upper, lowerExpression, upperExpression })
  | .familyUniform binder coordinate element familyCount =>
      return .familyUniform binder coordinate
        (← setFactRecurrenceState count paths initial transition slot environment element)
        familyCount
  | fact => pure fact

def evaluateTransition
    (environment : ParamEnvironment)
    (previousState : OperationalState)
    (transition : Array OperationalBoundExpr) : Except OperationalError OperationalState := do
  if transition.size != previousState.size then
    throw (.unsupportedOutputArity transition.size previousState.size)
  let values ← transition.toList.mapM (OperationalBoundExpr.evaluate environment previousState)
  let next := values.zip previousState.toList |>.map fun (value, previous) =>
    setFactMaximum value previous
  pure next.toArray

def repeatTransition
    (count : Nat)
    (environment : ParamEnvironment)
    (transition : Array OperationalBoundExpr)
    (state : OperationalState) : Except OperationalError OperationalState :=
  match count with
  | 0 => pure state
  | count + 1 => do
      let next ← evaluateTransition environment state transition
      repeatTransition count environment transition next

def fallbackMatrixFact
    (node port : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment) : Except OperationalError OperationalFact := do
  let cap ← match matrixCap matrixType environment with
    | some cap => pure cap
    | none => throw (.invalidMatrixParameters node)
  let params ← match matrixType.evaluate environment (.constant cap) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters node)
  let fact : OperationalMatrixFact := {
    subject := { node, port }
    origin := .value temporaryScope { node, port }
    matrixType
    matrixParams := params
    totalHardBound := .closedInt (.constant cap)
  }
  pure (.matrix (fact.initializePrimitivePolynomial .large))

def defaultFact
    (node : Nat)
    (port : Nat)
    (wireType : WireTypeExpr)
    (environment : ParamEnvironment) : Except OperationalError OperationalFact :=
  match wireType with
  | .matrix matrixType => fallbackMatrixFact node port matrixType environment
  | .trapdoor matrixType _ _ _ _ =>
      match matrixCap matrixType environment with
      | some cap => do
          let params ← match matrixType.evaluate environment (.constant cap) with
            | some params => pure params
            | none => throw (.invalidMatrixParameters node)
          pure (.trapdoor {
            subject := { node, port }
            matrixType
            matrixParams := params
            maximum := .closedInt (.constant cap)
            publicIdentity := .sampledTrapdoor temporaryScope { node, port := 0 }
          })
      | none => throw (.invalidMatrixParameters node)
  | .integer | .constantInt => pure (.integer {
      subject := { node, port }
      origin := .local temporaryScope { node, port }
      lower := 0
      upper := 0
      lowerExpression := .closedInt (.constant 0)
      upperExpression := .closedInt (.constant 0)
    })
  | .boolean | .constantBool => pure .boolean
  | .real | .constantReal => pure .real
  | .bytes length =>
      match length.evaluate environment with
      | some value => pure (.bytes {
          subject := { node, port }
          origin := .local temporaryScope { node, port }
          length := value
        })
      | none => throw (.invalidCount node 0)
  | .typedBlob typeName _ => pure (.typedBlob typeName)
  | .preimage matrixType => fallbackMatrixFact node port matrixType environment
  | .indexedFamily element count => do
      let element ← defaultFact node port element environment
      match count.evaluate environment with
      | some value => pure (.familyUniform
          { owner := .root (.standalone 0), producerNode := node, binderSlot := port }
          none element value)
      | none => throw (.invalidCount node 0)

def lookupFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalFact :=
  match facts.values[wire.node]?.bind fun outputs => outputs[wire.port]? with
  | some fact => pure fact
  | none => throw (.missingOperand node wire)

def integerFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalIntegerFact := do
  match ← lookupFact node facts wire with
  | .integer fact => pure fact
  | _ => throw (.operandNotInteger node wire)

private def requireBooleanFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError Unit := do
  match ← lookupFact node facts wire with
  | .boolean => pure ()
  | _ => throw (.operandNotBoolean node wire)

private def requireRealFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError Unit := do
  match ← lookupFact node facts wire with
  | .real => pure ()
  | _ => throw (.operandNotReal node wire)

private def integerFact
    (node port : Nat)
    (lower upper : Int) : Except OperationalError OperationalFact := do
  if lower > upper then throw (.invalidBound node lower)
  pure (.integer {
    subject := { node, port }
    origin := .local temporaryScope { node, port }
    lower
    upper
    lowerExpression := .closedInt (.constant lower)
    upperExpression := .closedInt (.constant upper)
  })

private def integerFactWithExpressions
    (node port : Nat)
    (lower upper : Int)
    (lowerExpression upperExpression : OperationalBoundExpr) :
    Except OperationalError OperationalFact := do
  if lower > upper then throw (.invalidBound node lower)
  pure (.integer {
    subject := { node, port }
    origin := .local temporaryScope { node, port }
    lower
    upper
    lowerExpression
    upperExpression
  })

private structure OperationalIntegerInterval where
  lower : Int
  upper : Int
  lowerExpression : OperationalBoundExpr
  upperExpression : OperationalBoundExpr

private def integerBinaryInterval
    (node : Nat)
    (operation : IntBinaryOp)
    (left right : OperationalIntegerFact) : Except OperationalError OperationalIntegerInterval := do
  match operation with
  | .add => pure {
      lower := left.lower + right.lower
      upper := left.upper + right.upper
      lowerExpression := .add left.lowerExpression right.lowerExpression
      upperExpression := .add left.upperExpression right.upperExpression
    }
  | .subtract => pure {
      lower := left.lower - right.upper
      upper := left.upper - right.lower
      lowerExpression := .subtract left.lowerExpression right.upperExpression
      upperExpression := .subtract left.upperExpression right.lowerExpression
    }
  | .multiply =>
      let values := [
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper
      ]
      match values with
      | [] => throw (.invalidBound node 0)
      | first :: tail =>
          let expressions := [
            OperationalBoundExpr.multiply left.lowerExpression right.lowerExpression,
            OperationalBoundExpr.multiply left.lowerExpression right.upperExpression,
            OperationalBoundExpr.multiply left.upperExpression right.lowerExpression,
            OperationalBoundExpr.multiply left.upperExpression right.upperExpression
          ]
          let firstExpression := expressions.headD (.closedInt (.constant first))
          pure {
            lower := tail.foldl min first
            upper := tail.foldl max first
            lowerExpression := expressions.drop 1 |>.foldl OperationalBoundExpr.minimum firstExpression
            upperExpression := expressions.drop 1 |>.foldl OperationalBoundExpr.maximum firstExpression
          }
  | .divide =>
      if left.lower < 0 then throw (.invalidBound node left.lower)
      if right.lower ≤ 0 then throw .divisionByZero
      pure {
        lower := left.lower / right.upper
        upper := left.upper / right.lower
        lowerExpression := .divide left.lowerExpression right.upperExpression
        upperExpression := .divide left.upperExpression right.lowerExpression
      }
  | .remainder =>
      if left.lower < 0 then throw (.invalidBound node left.lower)
      if right.lower ≤ 0 then throw .divisionByZero
      pure {
        lower := 0
        upper := right.upper - 1
        lowerExpression := .closedInt (.constant 0)
        upperExpression := .subtract right.upperExpression (.closedInt (.constant 1))
      }

def matrixMaximum
    (node : Nat)
    (wire : WireRef)
    (facts : OperationalScopeFacts) : Except OperationalError Int := do
  match ← lookupFact node facts wire with
  | .matrix fact => fact.totalHardBound.evaluate [] #[]
  | .trapdoor fact => fact.maximum.evaluate [] #[]
  | _ => throw (.operandNotMatrix node wire)

def matrixMaximumExpr
    (node : Nat)
    (wire : WireRef)
    (facts : OperationalScopeFacts) : Except OperationalError OperationalBoundExpr := do
  match ← lookupFact node facts wire with
  | .matrix fact => pure fact.totalHardBound
  | .trapdoor fact => pure fact.maximum
  | _ => throw (.operandNotMatrix node wire)

def maximumArgumentExprs
    (node : Nat)
    (arguments : List WireRef)
    (facts : OperationalScopeFacts) : Except OperationalError OperationalBoundExpr := do
  let values ← arguments.mapM (matrixMaximumExpr node · facts)
  pure <| values.foldl OperationalBoundExpr.maximum (.closedInt (.constant 0))

def maximumArguments
    (node : Nat)
    (arguments : List WireRef)
    (facts : OperationalScopeFacts) : Except OperationalError Int := do
  let values ← arguments.mapM (matrixMaximum node · facts)
  pure <| values.foldl max 0

def cappedMatrixFact
    (nodeIndex : Nat)
    (outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : Int) : Except OperationalError OperationalFact := do
  let cap ← match matrixCap matrixType environment with
    | some value => pure value
    | none => throw (.invalidMatrixParameters nodeIndex)
  if bound < 0 then throw (.invalidBound nodeIndex bound)
  let maximum := min cap bound
  let params ← match matrixType.evaluate environment (.constant maximum) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters nodeIndex)
  let fact : OperationalMatrixFact := {
    subject := { node := nodeIndex, port := outputPort }
    origin := .value temporaryScope { node := nodeIndex, port := outputPort }
    matrixType
    matrixParams := params
    totalHardBound := .closedInt (.constant maximum)
  }
  pure (.matrix (fact.initializePrimitivePolynomial .bounded))

def cappedMatrixFactExpr
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : OperationalBoundExpr) : Except OperationalError OperationalFact := do
  let cap ← match matrixCap matrixType environment with
    | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
  let parameterBound ← if bound.usesPrevious then pure cap else do
    let maximum ← bound.evaluate environment #[]
    if maximum < 0 then throw (.invalidBound nodeIndex maximum)
    pure (min cap maximum)
  let params ← match matrixType.evaluate environment (.constant parameterBound) with
    | some params => pure params | none => throw (.invalidMatrixParameters nodeIndex)
  let fact : OperationalMatrixFact := {
    subject := { node := nodeIndex, port := outputPort }
    origin := .value temporaryScope { node := nodeIndex, port := outputPort }
    matrixType
    matrixParams := params
    totalHardBound := .minimum (.closedInt (.constant cap)) bound
  }
  pure (.matrix (fact.initializePrimitivePolynomial .bounded))

def classifiedMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : Int)
    (large : Bool)
    (canonicalRange : CanonicalRange := .unknown)
    (metadata : OperationalMatrixMetadata := {}) : Except OperationalError OperationalFact := do
  let fact ← cappedMatrixFact nodeIndex outputPort matrixType environment bound
  match fact with
  | .matrix fact =>
      let role := if large then OperationalFactorRole.large else .bounded
      pure (.matrix (({ fact with
        totalHardBound := .closedInt (.constant bound), canonicalRange, metadata
      }).initializePrimitivePolynomial role))
  | _ => throw (.malformedRelation nodeIndex)

def classifiedMatrixFactExpr
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : OperationalBoundExpr)
    (large : Bool)
    (canonicalRange : CanonicalRange := .unknown)
    (metadata : OperationalMatrixMetadata := {}) : Except OperationalError OperationalFact := do
  let fact ← cappedMatrixFactExpr nodeIndex outputPort matrixType environment bound
  match fact with
  | .matrix fact =>
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let totalHardBound := .minimum (.closedInt (.constant cap)) bound
      let role := if large then OperationalFactorRole.large else .bounded
      pure (.matrix (({ fact with totalHardBound, canonicalRange, metadata
        }).initializePrimitivePolynomial role))
  | _ => throw (.malformedRelation nodeIndex)

def matrixTargetSummary (fact : OperationalMatrixFact) : RelationTargetSummary := {
  origin := fact.origin
  matrixType := fact.matrixType
  matrixParams := fact.matrixParams
  totalHardBound := fact.totalHardBound
  canonicalRange := fact.canonicalRange
  polynomial := relationSnapshotPolynomial fact.polynomial
}

def matrixFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalMatrixFact := do
  match ← lookupFact node facts wire with
  | .matrix fact => pure fact
  | _ => throw (.operandNotMatrix node wire)

private def operationalProductFromFactors
    (factors : List OperationalFactorKey) : Except OperationalFlatError OperationalProductKey := do
  let first ← match factors.head? with
    | some factor => pure factor
    | none => throw .malformedProduct
  let rec visit
      (previousType : MatrixTypeExpr)
      (remaining : List OperationalFactorKey)
      (modes : List OperationalProductMode) :
      Except OperationalFlatError (List OperationalProductMode × MatrixTypeExpr) := do
    match remaining with
    | [] => pure (modes, previousType)
    | factor :: tail =>
        let (mode, outputType) ← inferOperationalProductMode previousType factor.inputType
        visit outputType tail (modes ++ [mode])
  let (modes, outputType) ← visit first.outputType (factors.drop 1) []
  pure { factors, modes, outputType }

private def factorPublicIdentity? (factor : OperationalFactorKey) : Option PublicMatrixIdentity :=
  match factor.leaf with
  | .primitive (.publicMatrix identity) => some identity
  | _ => none

private def factorPrimitiveOrigin? (factor : OperationalFactorKey) : Option MatrixOriginIdentity :=
  match factor.leaf with
  | .primitive (.matrix origin) => some origin
  | _ => none

private def matchingFactorRelation?
    (left right : OperationalFactorKey) : Option OperationalMatrixRelation := do
  if !left.transforms.isEmpty || !right.transforms.isEmpty then none else pure ()
  let publicIdentity ← factorPublicIdentity? left
  let producer ← factorPrimitiveOrigin? right
  let identityMatches (candidate : PublicMatrixIdentity) :=
    candidate == publicIdentity || match candidate, publicIdentity with
      | .selected _ candidateSelection candidateSource,
          .selected _ publicSelection publicSource =>
          candidateSelection == publicSelection &&
            publicIdentityTemplateEqual candidateSource publicSource
      | _, _ => false
  right.relations.find? fun relation =>
    match relation with
      | .decomposition value => identityMatches value.publicIdentity &&
          value.producer == producer &&
            value.status == ReconstructionStatus.available
      | .preimage value => identityMatches value.publicIdentity && value.producer == producer

private def rewriteOperationalTermRelation?
    (node : Nat)
    (term : OperationalTerm) : Except OperationalError (Option OperationalPolynomial) := do
  let rec visit
      (accumulated : List OperationalFactorKey) :
      List OperationalFactorKey → Except OperationalError (Option OperationalPolynomial)
    | left :: right :: tail =>
        match matchingFactorRelation? left right with
        | none => visit (accumulated ++ [left]) (right :: tail)
        | some relation => do
            let target := match relation with
              | .decomposition value => value.inputSummary
              | .preimage value => value.targetSummary
            let targetPolynomial := operationalPolynomialFromSnapshot target.polynomial
            if targetPolynomial.isEmpty then
              throw (.malformedRelation node)
            let rewritten ← targetPolynomial.mapM fun targetTerm => do
              let product ← operationalProductFromFactors
                (accumulated ++ targetTerm.product.factors ++ tail) |>.mapError
                  (fun _ => OperationalError.invalidMatrixParameters node)
              pure {
                coefficient := term.coefficient * targetTerm.coefficient
                product
              }
            pure (some rewritten)
    | _ => pure none
  visit [] term.product.factors

private def rewriteOperationalRelations
    (node : Nat)
    (polynomial : OperationalPolynomial) : Except OperationalError OperationalPolynomial :=
  let rec iterate : Nat → OperationalPolynomial → Except OperationalError OperationalPolynomial
    | 0, _ => throw (.invalidMatrixParameters node)
    | fuel + 1, current => do
        let rewrites ← current.mapM (rewriteOperationalTermRelation? node)
        if rewrites.all Option.isNone then pure current
        else
          let next := (current.zip rewrites).flatMap fun (term, rewrite) =>
            rewrite.getD [term]
          iterate fuel (normalizeOperationalTerms next)
  iterate 64 polynomial

private def sameConcreteMatrixShape (left right : Mxx.SamplerParams) : Bool :=
  left.modulus == right.modulus &&
    left.ringDimension == right.ringDimension &&
    left.rows == right.rows &&
    left.columns == right.columns

private def equivalentRetypeOperationalFactor
    (outputType : MatrixTypeExpr)
    (standalone : Bool)
    (factor : OperationalFactorKey) : OperationalFactorKey :=
  let updateSummary (summary : OperationalBoundedFactorSummary) := {
    summary with matrixType := outputType
  }
  let leaf := match factor.leaf with
    | .boundedSummary origin summary => .boundedSummary origin (updateSummary summary)
    | .exactTransform tokens _ => .exactTransform tokens outputType
    | primitive => primitive
  {
    factor with
    leaf
    inputType := if standalone then outputType else factor.inputType
    outputType
    boundedSummary := factor.boundedSummary.map updateSummary
  }

/-- Canonicalize a matrix-type expression after proving that its evaluated shape is unchanged.
This changes no value and records no transform, so relation-owner identities remain bare. -/
private def equivalentRetypeOperationalPolynomial
    (outputType : MatrixTypeExpr)
    (input : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  input.mapM fun term => do
    match term.product.factors.reverse with
    | [] => throw .malformedProduct
    | last :: reversePrefix =>
        let replacement := equivalentRetypeOperationalFactor outputType reversePrefix.isEmpty last
        let factors := (replacement :: reversePrefix).reverse
        pure { term with product := { term.product with factors, outputType } }

private def retypeMatrixFact
    (node : Nat)
    (expected : MatrixTypeExpr)
    (fact : OperationalMatrixFact)
    (environment : ParamEnvironment) : Except OperationalError OperationalMatrixFact := do
  if fact.matrixType == expected then return fact
  let expectedParams ← match expected.evaluate environment
      (.constant fact.matrixParams.maxCoefficientBound) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters node)
  if !sameConcreteMatrixShape fact.matrixParams expectedParams then
    throw (.outputTypeMismatch node)
  let polynomial ← equivalentRetypeOperationalPolynomial expected fact.polynomial
    |>.mapError fun error => OperationalError.flat node error
  pure { fact with matrixType := expected, matrixParams := expectedParams, polynomial }

private def requireMatrixType
    (node : Nat)
    (wire : WireRef)
    (expected : MatrixTypeExpr)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment) : Except OperationalError OperationalMatrixFact := do
  retypeMatrixFact node expected (← matrixFactAt node facts wire) environment

def valueOriginAt
    (scope : ScopeTemplateKey)
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalValueOrigin := do
  match ← lookupFact node facts wire with
  | .integer fact => pure fact.origin
  | .bytes fact => pure fact.origin
  | .matrix { origin := .value originScope originWire, .. } =>
      pure (.local originScope originWire)
  | .matrix { origin := .protocolInput input, .. } => pure (.protocolInput input)
  | _ => pure (.local scope wire)

def operationalPolynomialNoiseSummary
    (polynomial : OperationalPolynomial) :
    Except OperationalFlatError (Option OperationalBoundedFactorSummary) := do
  let noise := sortOperationalTerms
    (normalizeOperationalTerms (polynomial.filter operationalTermIsNoise))
  if noise.isEmpty then return none
  let summaries ← noise.mapM boundedNoiseTermSummary
  let firstTerm ← match noise.head? with
    | some term => pure term
    | none => throw .malformedProduct
  let hardBound := (noise.zip summaries).foldl (fun current pair =>
    .add current (.multiply
      (.closedInt (.constant (operationalAbsoluteCoefficient pair.1.coefficient)))
      pair.2.hardBound)) (.closedInt (.constant 0))
  let tokens := [.sumStart] ++ ((noise.zip summaries).flatMap fun (term, summary) =>
    boundedNoiseTermTokens term summary) ++ [.summaryBound hardBound, .sumEnd]
  pure (some {
    matrixType := firstTerm.product.outputType
    hardBound
    metadata := {
      isConstantPolynomial := summaries.all (·.metadata.isConstantPolynomial)
      knownZeroRows := none
    }
    provenance := tokens
  })

def OperationalMatrixFact.noiseHardBound
    (fact : OperationalMatrixFact) : Except OperationalFlatError OperationalBoundExpr := do
  match ← operationalPolynomialNoiseSummary fact.polynomial with
  | some summary => pure summary.hardBound
  | none => pure (.closedInt (.constant 0))

def OperationalMatrixFact.evaluateNoiseHardBound
    (fact : OperationalMatrixFact)
    (environment : ParamEnvironment)
    (states : List OperationalNumericState := []) : Except OperationalError Int := do
  let expression ← fact.noiseHardBound |>.mapError fun _ =>
    OperationalError.invalidMatrixParameters fact.subject.node
  expression.evaluateWithStates environment states

private def flatErrorAt (node : Nat) : OperationalFlatError → OperationalError
  | error => .flat node error

private def polynomialMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (polynomial : OperationalPolynomial)
    (canonicalRange : CanonicalRange := .unknown) : Except OperationalError OperationalFact := do
  let polynomial ← normalizeOperationalPolynomial polynomial |>.mapError (flatErrorAt nodeIndex)
  let cap ← match matrixCap matrixType environment with
    | some value => pure value
    | none => throw (.invalidMatrixParameters nodeIndex)
  let noiseSummary ← operationalPolynomialNoiseSummary polynomial |>.mapError (flatErrorAt nodeIndex)
  let totalHardBound :=
    if polynomial.any operationalTermIsSignal then
      .closedInt (.constant cap)
    else match noiseSummary with
      | some summary => .minimum (.closedInt (.constant cap)) summary.hardBound
      | none => .closedInt (.constant 0)
  let parameterBound :=
    if totalHardBound.usesPrevious then cap
    else match totalHardBound.evaluate environment #[] with
      | .ok value => min cap value
      | .error _ => cap
  let params ← match matrixType.evaluate environment (.constant parameterBound) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters nodeIndex)
  let metadata := match noiseSummary with
    | some summary => summary.metadata
    | none => {}
  pure (.matrix {
    subject := { node := nodeIndex, port := outputPort }
    origin := .value temporaryScope { node := nodeIndex, port := outputPort }
    matrixType
    matrixParams := params
    totalHardBound
    polynomial
    metadata
    canonicalRange
  })

private def multiplyConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial
    |>.mapError (flatErrorAt nodeIndex)
  let rewritten ← rewriteOperationalRelations nodeIndex raw
  let polynomial ← match rule with
    | .matrixMultiplyRelation declaredRight => do
        if declaredRight != rightWire then throw (.missingRelation nodeIndex declaredRight)
        if rewritten == raw then throw (.missingRelation nodeIndex rightWire)
        pure rewritten
    | _ => pure rewritten
  match ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial with
  | .matrix output => pure output
  | _ => throw (.operandNotMatrix nodeIndex rightWire)

private def addConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (subtract : Bool)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let left ← retypeMatrixFact nodeIndex matrixType left environment
  let right ← retypeMatrixFact nodeIndex matrixType right environment
  let polynomial := if subtract then
    subtractOperationalPolynomials left.polynomial right.polynomial
  else
    addOperationalPolynomials left.polynomial right.polynomial
  match ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial with
  | .matrix output => pure output
  | _ => throw (.operandNotMatrix nodeIndex { node := nodeIndex, port := outputPort })

private def addOperationalExprIds
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (subtract : Bool)
    (environment : ParamEnvironment) :
    OperationalExprArena → OperationalExprId → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, left, _, 0 => throw (.unsupportedOperationalExpr left)
  | arena, left, right, fuel + 1 => do
      let leftExpr ← match arena.get? left with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef left)
      let rightExpr ← match arena.get? right with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef right)
      match leftExpr.node, rightExpr.node with
      | .concrete leftFact, .concrete rightFact =>
          let output ← addConcreteMatrixFacts nodeIndex outputPort matrixType subtract environment
            leftFact rightFact
          pure (arena.pushConcrete output)
      | .select leftSelection (.exact leftBranches),
          .select rightSelection (.exact rightBranches) =>
          if leftSelection == rightSelection then
            if leftBranches.size != rightBranches.size then
              throw (.operationalExprTypeMismatch left right)
            let mut arena := arena
            let mut outputs : Array OperationalExprId := #[]
            for branch in [:leftBranches.size] do
              let (nextArena, output) ← addOperationalExprIds nodeIndex outputPort matrixType
                subtract environment arena leftBranches[branch]! rightBranches[branch]! fuel
              arena := nextArena
              outputs := outputs.push output
            arena.pushSelect leftSelection (.exact outputs)
          else
            let mut arena := arena
            let mut outputs : Array OperationalExprId := #[]
            for branch in leftBranches do
              let (nextArena, output) ← addOperationalExprIds nodeIndex outputPort matrixType
                subtract environment arena branch right fuel
              arena := nextArena
              outputs := outputs.push output
            arena.pushSelect leftSelection (.exact outputs)
      | .select selection (.exact branches), _ =>
          let mut arena := arena
          let mut outputs : Array OperationalExprId := #[]
          for branch in branches do
            let (nextArena, output) ← addOperationalExprIds nodeIndex outputPort matrixType
              subtract environment arena branch right fuel
            arena := nextArena
            outputs := outputs.push output
          arena.pushSelect selection (.exact outputs)
      | _, .select selection (.exact branches) =>
          let mut arena := arena
          let mut outputs : Array OperationalExprId := #[]
          for branch in branches do
            let (nextArena, output) ← addOperationalExprIds nodeIndex outputPort matrixType
              subtract environment arena left branch fuel
            arena := nextArena
            outputs := outputs.push output
          arena.pushSelect selection (.exact outputs)
      | .select leftSelection (.schemaEnvelope leftCount leftRepresentative leftSummary),
          .select rightSelection (.schemaEnvelope rightCount rightRepresentative rightSummary) =>
          if leftSelection != rightSelection then
            let operation := if subtract then OperationalMatrixExprNode.subtract left right
              else .add left right
            pure (arena.push { matrixType, node := operation })
          else if leftCount != rightCount then
            throw (.operationalExprTypeMismatch left right)
          else
            let (arena, output) ← addOperationalExprIds nodeIndex outputPort matrixType subtract
              environment arena leftRepresentative rightRepresentative fuel
            let outputFact ← arena.concreteFact output
            let _ ← match rightSummary.uniformSchema with
              | some schema => pure schema
              | none => throw (.unsupportedOperationalExpr rightRepresentative)
            let outputSummary ← match recomputeSelectedMatrixSummary leftSummary outputFact with
              | some value => pure value
              | none => throw (.unsupportedOperationalExpr leftRepresentative)
            arena.pushSelect leftSelection (.schemaEnvelope leftCount output outputSummary)
      | .select selection (.schemaEnvelope count representative summary), _ =>
          match rightExpr.node with
          | .select .. =>
              let operation := if subtract then OperationalMatrixExprNode.subtract left right
                else .add left right
              pure (arena.push { matrixType, node := operation })
          | _ =>
              let (arena, output) ← addOperationalExprIds nodeIndex outputPort matrixType subtract
                environment arena representative right fuel
              let outputFact ← arena.concreteFact output
              let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
                | some value => pure value
                | none => throw (.unsupportedOperationalExpr representative)
              arena.pushSelect selection (.schemaEnvelope count output outputSummary)
      | _, .select selection (.schemaEnvelope count representative summary) =>
          match leftExpr.node with
          | .select .. =>
              let operation := if subtract then OperationalMatrixExprNode.subtract left right
                else .add left right
              pure (arena.push { matrixType, node := operation })
          | _ =>
              let (arena, output) ← addOperationalExprIds nodeIndex outputPort matrixType subtract
                environment arena left representative fuel
              let outputFact ← arena.concreteFact output
              let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
                | some value => pure value
                | none => throw (.unsupportedOperationalExpr representative)
              arena.pushSelect selection (.schemaEnvelope count output outputSummary)
      | _, _ =>
          if leftExpr.matrixType != rightExpr.matrixType || leftExpr.matrixType != matrixType then
            throw (.operationalExprTypeMismatch left right)
          let operation := if subtract then OperationalMatrixExprNode.subtract left right
            else .add left right
          pure (arena.push { matrixType, node := operation })

private def addOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (subtract : Bool)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (left right : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, leftId) ← arena.pushMatrixFact left
  let (arena, rightId) ← arena.pushMatrixFact right
  let (arena, result) ← addOperationalExprIds nodeIndex outputPort matrixType subtract environment
    arena leftId rightId (arena.nodes.size + 1)
  pure (arena, .matrixExpr result)

private def multiplyOperationalExprIds
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment) :
    OperationalExprArena → OperationalExprId → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, left, _, 0 => throw (.unsupportedOperationalExpr left)
  | arena, left, right, fuel + 1 => do
      let leftExpr ← match arena.get? left with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef left)
      let rightExpr ← match arena.get? right with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef right)
      match leftExpr.node, rightExpr.node with
      | .concrete leftFact, .concrete rightFact =>
          let output ← multiplyConcreteMatrixFacts nodeIndex outputPort matrixType rule rightWire
            environment leftFact rightFact
          pure (arena.pushConcrete output)
      | .select leftSelection (.exact leftBranches),
          .select rightSelection (.exact rightBranches) =>
          if leftSelection == rightSelection then
            if leftBranches.size != rightBranches.size then
              throw (.operationalExprTypeMismatch left right)
            let mut arena := arena
            let mut outputs : Array OperationalExprId := #[]
            for branch in [:leftBranches.size] do
              let leftBranch := leftBranches[branch]!
              let rightBranch := rightBranches[branch]!
              let (nextArena, output) ← multiplyOperationalExprIds nodeIndex outputPort matrixType
                rule rightWire environment arena leftBranch rightBranch fuel
              arena := nextArena
              outputs := outputs.push output
            arena.pushSelect leftSelection (.exact outputs)
          else
            let mut arena := arena
            let mut outputs : Array OperationalExprId := #[]
            for branch in leftBranches do
              let (nextArena, output) ← multiplyOperationalExprIds nodeIndex outputPort matrixType
                rule rightWire environment arena branch right fuel
              arena := nextArena
              outputs := outputs.push output
            arena.pushSelect leftSelection (.exact outputs)
      | .select selection (.exact branches), _ =>
          let mut arena := arena
          let mut outputs : Array OperationalExprId := #[]
          for branch in branches do
            let (nextArena, output) ← multiplyOperationalExprIds nodeIndex outputPort matrixType
              rule rightWire environment arena branch right fuel
            arena := nextArena
            outputs := outputs.push output
          arena.pushSelect selection (.exact outputs)
      | _, .select selection (.exact branches) =>
          let mut arena := arena
          let mut outputs : Array OperationalExprId := #[]
          for branch in branches do
            let (nextArena, output) ← multiplyOperationalExprIds nodeIndex outputPort matrixType
              rule rightWire environment arena left branch fuel
            arena := nextArena
            outputs := outputs.push output
          arena.pushSelect selection (.exact outputs)
      | .select leftSelection (.schemaEnvelope leftCount leftRepresentative leftSummary),
          .select rightSelection (.schemaEnvelope rightCount rightRepresentative rightSummary) =>
          if leftSelection != rightSelection then
            pure (arena.push { matrixType, node := .multiply left right })
          else if leftCount != rightCount then
            throw (.operationalExprTypeMismatch left right)
          else
            let _ ← match rightSummary.uniformSchema with
              | some schema => pure schema
              | none => throw (.unsupportedOperationalExpr rightRepresentative)
            let (arena, output) ← multiplyOperationalExprIds nodeIndex outputPort matrixType rule
              rightWire environment arena leftRepresentative rightRepresentative fuel
            let outputFact ← arena.concreteFact output
            let outputSummary ← match recomputeSelectedMatrixSummary leftSummary outputFact with
              | some value => pure value
              | none => throw (.unsupportedOperationalExpr leftRepresentative)
            arena.pushSelect leftSelection (.schemaEnvelope leftCount output outputSummary)
      | .select selection (.schemaEnvelope count representative summary), _ =>
          match rightExpr.node with
          | .select .. => pure (arena.push { matrixType, node := .multiply left right })
          | _ =>
              let (arena, output) ← multiplyOperationalExprIds nodeIndex outputPort matrixType rule
                rightWire environment arena representative right fuel
              let outputFact ← arena.concreteFact output
              let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
                | some value => pure value
                | none => throw (.unsupportedOperationalExpr representative)
              arena.pushSelect selection (.schemaEnvelope count output outputSummary)
      | _, .select selection (.schemaEnvelope count representative summary) =>
          match leftExpr.node with
          | .select .. => pure (arena.push { matrixType, node := .multiply left right })
          | _ =>
              let (arena, output) ← multiplyOperationalExprIds nodeIndex outputPort matrixType rule
                rightWire environment arena left representative fuel
              let outputFact ← arena.concreteFact output
              let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
                | some value => pure value
                | none => throw (.unsupportedOperationalExpr representative)
              arena.pushSelect selection (.schemaEnvelope count output outputSummary)
      | _, _ =>
          if leftExpr.matrixType.columns != rightExpr.matrixType.rows then
            throw (.operationalExprTypeMismatch left right)
          pure (arena.push { matrixType, node := .multiply left right })

private def multiplyOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (left right : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, leftId) ← arena.pushMatrixFact left
  let (arena, rightId) ← arena.pushMatrixFact right
  let (arena, result) ← multiplyOperationalExprIds nodeIndex outputPort matrixType rule rightWire
    environment arena leftId rightId (arena.nodes.size + 1)
  pure (arena, .matrixExpr result)

private def operationalProductTokens
    (term : OperationalTerm) : List OperationalCompressionToken :=
  [.productStart, .termStart term.coefficient] ++
    term.product.factors.flatMap (fun factor => match factor.leaf with
      | .primitive identity => [.primitive identity] ++
          factor.transforms.map OperationalCompressionToken.transform
      | .boundedSummary origin _ => origin.tokens
      | .exactTransform tokens _ => tokens) ++
    term.product.modes.map OperationalCompressionToken.productMode ++
    [.intermediateType term.product.outputType, .termEnd, .productEnd]

private def tensorOperationalPolynomials
    (outputType : MatrixTypeExpr)
    (left right : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let rows ← left.mapM fun leftTerm => right.mapM fun rightTerm => do
    let tokens := [.groupStart] ++ operationalProductTokens leftTerm ++ [.groupEnd,
      .groupStart] ++ operationalProductTokens rightTerm ++ [.groupEnd,
      .intermediateType outputType]
    let role := if operationalTermIsSignal leftTerm || operationalTermIsSignal rightTerm then
      OperationalFactorRole.large else .bounded
    let summary ← if role == OperationalFactorRole.bounded then do
      let leftSummary ← boundedNoiseTermSummary leftTerm
      let rightSummary ← boundedNoiseTermSummary rightTerm
      let ringFactor := if leftSummary.metadata.isConstantPolynomial ||
          rightSummary.metadata.isConstantPolynomial then .closedInt (.constant 1)
        else .closedInt outputType.ringDimension
      pure (some {
        matrixType := outputType
        hardBound := .multiply ringFactor
          (.multiply leftSummary.hardBound rightSummary.hardBound)
        metadata := {
          isConstantPolynomial := leftSummary.metadata.isConstantPolynomial &&
            rightSummary.metadata.isConstantPolynomial
          knownZeroRows := none
        }
        provenance := tokens
      })
    else pure none
    let leaf := match summary with
      | some bounded =>
          let origin : OperationalCompressionOrigin := { kind := .boundedRun, tokens }
          OperationalFactorLeaf.boundedSummary origin bounded
      | none => .exactTransform tokens outputType
    let factor : OperationalFactorKey := {
      leaf
      inputType := outputType
      outputType
      role
      boundedSummary := summary
    }
    pure {
      coefficient := leftTerm.coefficient * rightTerm.coefficient
      product := { factors := [factor], modes := [], outputType }
    }
  pure (normalizeOperationalTerms rows.flatten)

private def tensorConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← tensorOperationalPolynomials matrixType left.polynomial right.polynomial
    |>.mapError (flatErrorAt nodeIndex)
  match ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial with
  | .matrix output => pure output
  | _ => throw (.operandNotMatrix nodeIndex right.subject)

private def tensorOperationalExprIds
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment) :
    OperationalExprArena → OperationalExprId → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, left, _, 0 => throw (.unsupportedOperationalExpr left)
  | arena, left, right, fuel + 1 => do
      let leftExpr ← match arena.get? left with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef left)
      let rightExpr ← match arena.get? right with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef right)
      match leftExpr.node, rightExpr.node with
      | .concrete leftFact, .concrete rightFact =>
          let output ← tensorConcreteMatrixFacts nodeIndex outputPort matrixType environment
            leftFact rightFact
          pure (arena.pushConcrete output)
      | .select leftSelection (.exact leftBranches),
          .select rightSelection (.exact rightBranches) =>
          if leftSelection != rightSelection then
            pure (arena.push { matrixType, node := .tensor left right })
          else if leftBranches.size != rightBranches.size then
            throw (.operationalExprTypeMismatch left right)
          else
            let mut arena := arena
            let mut outputs : Array OperationalExprId := #[]
            for branch in [:leftBranches.size] do
              let (nextArena, output) ← tensorOperationalExprIds nodeIndex outputPort matrixType
                environment arena leftBranches[branch]! rightBranches[branch]! fuel
              arena := nextArena
              outputs := outputs.push output
            arena.pushSelect leftSelection (.exact outputs)
      | .select selection (.exact branches), _ =>
          match rightExpr.node with
          | .select .. => pure (arena.push { matrixType, node := .tensor left right })
          | _ =>
              let mut arena := arena
              let mut outputs : Array OperationalExprId := #[]
              for branch in branches do
                let (nextArena, output) ← tensorOperationalExprIds nodeIndex outputPort matrixType
                  environment arena branch right fuel
                arena := nextArena
                outputs := outputs.push output
              arena.pushSelect selection (.exact outputs)
      | _, .select selection (.exact branches) =>
          match leftExpr.node with
          | .select .. => pure (arena.push { matrixType, node := .tensor left right })
          | _ =>
              let mut arena := arena
              let mut outputs : Array OperationalExprId := #[]
              for branch in branches do
                let (nextArena, output) ← tensorOperationalExprIds nodeIndex outputPort matrixType
                  environment arena left branch fuel
                arena := nextArena
                outputs := outputs.push output
              arena.pushSelect selection (.exact outputs)
      | .select leftSelection (.schemaEnvelope leftCount leftRepresentative leftSummary),
          .select rightSelection (.schemaEnvelope rightCount rightRepresentative rightSummary) =>
          if leftSelection != rightSelection then
            pure (arena.push { matrixType, node := .tensor left right })
          else if leftCount != rightCount then
            throw (.operationalExprTypeMismatch left right)
          else
            let _ ← match rightSummary.uniformSchema with
              | some schema => pure schema
              | none => throw (.unsupportedOperationalExpr rightRepresentative)
            let (arena, output) ← tensorOperationalExprIds nodeIndex outputPort matrixType
              environment arena leftRepresentative rightRepresentative fuel
            let outputFact ← arena.concreteFact output
            let outputSummary ← match recomputeSelectedMatrixSummary leftSummary outputFact with
              | some value => pure value
              | none => throw (.unsupportedOperationalExpr leftRepresentative)
            arena.pushSelect leftSelection (.schemaEnvelope leftCount output outputSummary)
      | .select selection (.schemaEnvelope count representative summary), _ =>
          match rightExpr.node with
          | .select .. => pure (arena.push { matrixType, node := .tensor left right })
          | _ =>
              let (arena, output) ← tensorOperationalExprIds nodeIndex outputPort matrixType
                environment arena representative right fuel
              let outputFact ← arena.concreteFact output
              let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
                | some value => pure value
                | none => throw (.unsupportedOperationalExpr representative)
              arena.pushSelect selection (.schemaEnvelope count output outputSummary)
      | _, .select selection (.schemaEnvelope count representative summary) =>
          match leftExpr.node with
          | .select .. => pure (arena.push { matrixType, node := .tensor left right })
          | _ =>
              let (arena, output) ← tensorOperationalExprIds nodeIndex outputPort matrixType
                environment arena left representative fuel
              let outputFact ← arena.concreteFact output
              let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
                | some value => pure value
                | none => throw (.unsupportedOperationalExpr representative)
              arena.pushSelect selection (.schemaEnvelope count output outputSummary)
      | _, _ => pure (arena.push { matrixType, node := .tensor left right })

private def tensorOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (left right : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, leftId) ← arena.pushMatrixFact left
  let (arena, rightId) ← arena.pushMatrixFact right
  let (arena, result) ← tensorOperationalExprIds nodeIndex outputPort matrixType environment
    arena leftId rightId (arena.nodes.size + 1)
  pure (arena, .matrixExpr result)

private def mapOperationalExprWithFuel
    (mapFact : OperationalMatrixFact → OperationalMatrixFact)
    (mapSelection : DynamicSelectionIdentity → DynamicSelectionIdentity) :
    OperationalExprArena → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, root, 0 => throw (.unsupportedOperationalExpr root)
  | arena, root, fuel + 1 => do
      let expression ← match arena.get? root with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef root)
      let pushUnary
          (constructor : OperationalExprId → OperationalMatrixExprNode)
          (value : OperationalExprId) := do
        let (arena, mapped) ←
          mapOperationalExprWithFuel mapFact mapSelection arena value fuel
        pure (arena.push { expression with node := constructor mapped })
      let pushBinary
          (constructor : OperationalExprId → OperationalExprId → OperationalMatrixExprNode)
          (left right : OperationalExprId) := do
        let (arena, mappedLeft) ←
          mapOperationalExprWithFuel mapFact mapSelection arena left fuel
        let (arena, mappedRight) ←
          mapOperationalExprWithFuel mapFact mapSelection arena right fuel
        pure (arena.push { expression with node := constructor mappedLeft mappedRight })
      match expression.node with
      | .concrete fact => pure (arena.pushConcrete (mapFact fact))
      | .add left right => pushBinary OperationalMatrixExprNode.add left right
      | .subtract left right => pushBinary OperationalMatrixExprNode.subtract left right
      | .multiply left right => pushBinary OperationalMatrixExprNode.multiply left right
      | .tensor left right => pushBinary OperationalMatrixExprNode.tensor left right
      | .concat axis inputs =>
          let mut arena := arena
          let mut mappedInputs : Array OperationalExprId := #[]
          for input in inputs do
            let (nextArena, mapped) ←
              mapOperationalExprWithFuel mapFact mapSelection arena input fuel
            arena := nextArena
            mappedInputs := mappedInputs.push mapped
          pure (arena.push { expression with node := .concat axis mappedInputs })
      | .transform operation value =>
          pushUnary (OperationalMatrixExprNode.transform operation) value
      | .select selection (.exact branches) =>
          let mut arena := arena
          let mut mappedBranches : Array OperationalExprId := #[]
          for branch in branches do
            let (nextArena, mapped) ←
              mapOperationalExprWithFuel mapFact mapSelection arena branch fuel
            arena := nextArena
            mappedBranches := mappedBranches.push mapped
          arena.pushSelect (mapSelection selection) (.exact mappedBranches)
      | .select selection (.schemaEnvelope count representative summary) =>
          let (arena, mapped) ←
            mapOperationalExprWithFuel mapFact mapSelection arena representative fuel
          let mappedFact ← arena.concreteFact mapped
          let mappedSummary ← match recomputeSelectedMatrixSummary summary mappedFact with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr representative)
          arena.pushSelect (mapSelection selection)
            (.schemaEnvelope count mapped mappedSummary)

private def mapOperationalExpr
    (arena : OperationalExprArena)
    (root : OperationalExprId)
    (mapFact : OperationalMatrixFact → OperationalMatrixFact)
    (mapSelection : DynamicSelectionIdentity → DynamicSelectionIdentity := id) :
    Except OperationalError (OperationalExprArena × OperationalExprId) :=
  mapOperationalExprWithFuel mapFact mapSelection arena root (arena.nodes.size + 1)

private def exactOneIndicatorFactor
    (scope : ScopeTemplateKey)
    (node : Nat)
    (selection : OperationalValueOrigin)
    (branch : Nat)
    (matrixType : MatrixTypeExpr) : OperationalFactorKey :=
  let scalarType : MatrixTypeExpr := {
    modulus := matrixType.modulus
    ringDimension := matrixType.ringDimension
    rows := .constant 1
    columns := .constant 1
  }
  let binder : FamilyTemplateBinder := { owner := scope, producerNode := node, binderSlot := 0 }
  let metadata : OperationalMatrixMetadata := { isConstantPolynomial := true }
  let summary : OperationalBoundedFactorSummary := {
    matrixType := scalarType
    hardBound := .closedInt (.constant 1)
    metadata
    provenance := [.primitive (.selectionIndicator binder { index := selection } branch)]
  }
  {
    leaf := .primitive (.selectionIndicator binder { index := selection } branch)
    inputType := scalarType
    outputType := scalarType
    role := .bounded
    boundedSummary := some summary
    protections := [.exactOneIndicator]
  }

private def parameterScalarPolynomial
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (value : IntExpr)
    (matrixType : MatrixTypeExpr) : OperationalPolynomial :=
  let scalarType : MatrixTypeExpr := {
    modulus := matrixType.modulus
    ringDimension := matrixType.ringDimension
    rows := .constant 1
    columns := .constant 1
  }
  let identity := OperationalPrimitiveIdentity.parameterScalar environment domains value
  let metadata : OperationalMatrixMetadata := { isConstantPolynomial := true }
  let summary : OperationalBoundedFactorSummary := {
    matrixType := scalarType
    hardBound := .contextual .maximumAbsolute environment domains value
    metadata
    provenance := [.primitive identity]
  }
  [{
    coefficient := 1
    product := {
      factors := [{
        leaf := .primitive identity
        inputType := scalarType
        outputType := scalarType
        role := .bounded
        boundedSummary := some summary
      }]
      modes := []
      outputType := scalarType
    }
  }]

private def prependOperationalFactor
    (factor : OperationalFactorKey)
    (term : OperationalTerm) : Except OperationalFlatError OperationalTerm := do
  let product ← operationalProductFromFactors (factor :: term.product.factors)
  pure { term with product }

private def maximumOperationalBounds : List OperationalBoundExpr → OperationalBoundExpr
  | [] => .closedInt (.constant 0)
  | head :: tail => tail.foldl OperationalBoundExpr.maximum head

private def discardBranchLocalRelations (term : OperationalTerm) : OperationalTerm := {
  term with product := {
    term.product with factors := term.product.factors.map fun factor => {
      factor with
      protections := factor.protections.filter fun protection => match protection with
        | .relationOwner | .decompositionOwner => false
        | _ => true
      relations := []
    }
  }
}

/-- Preserve exact-one branch structure for signal terms and use max, not a triangle sum, for the
bounded noise selected from mutually exclusive branches. -/
private def selectOperationalPolynomials
    (scope : ScopeTemplateKey)
    (node : Nat)
    (selection : OperationalValueOrigin)
    (matrixType : MatrixTypeExpr)
    (branches : List OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  if branches.isEmpty then throw .malformedProduct
  let indexed := branches.zipIdx
  let signalRows ← indexed.mapM fun (branchTerms, branch) => do
    let indicator := exactOneIndicatorFactor scope node selection branch matrixType
    (branchTerms.filter operationalTermIsSignal).mapM (prependOperationalFactor indicator)
  let branchNoise ← indexed.mapM fun (branchTerms, _) =>
    compressBoundedNoiseSum
      ((branchTerms.filter operationalTermIsNoise).map discardBranchLocalRelations)
  let noiseSummaries ← branchNoise.mapM fun terms => match terms with
    | [] => pure none
    | [term] => match term.product.factors with
      | [factor] => return some (← factorBoundedSummary factor)
      | _ => throw .malformedProduct
    | _ => throw .malformedProduct
  let presentSummaries := noiseSummaries.filterMap id
  let selectedNoise ← if presentSummaries.isEmpty then pure [] else do
    let bound := maximumOperationalBounds (presentSummaries.map (·.hardBound))
    let metadata : OperationalMatrixMetadata := {
      isConstantPolynomial := presentSummaries.all (·.metadata.isConstantPolynomial)
      knownZeroRows := none
    }
    let tokens := [.sumStart] ++ presentSummaries.flatMap (·.provenance) ++
      [.summaryBound bound, .sumEnd]
    let summary : OperationalBoundedFactorSummary := {
      matrixType
      hardBound := bound
      metadata
      provenance := tokens
    }
    let origin : OperationalCompressionOrigin := { kind := .boundedNoiseSum, tokens }
    let factor : OperationalFactorKey := {
      leaf := .boundedSummary origin summary
      inputType := matrixType
      outputType := matrixType
      role := .bounded
      boundedSummary := some summary
    }
    pure [{ coefficient := 1, product := { factors := [factor], modes := [], outputType := matrixType } }]
  pure (normalizeOperationalTerms (signalRows.flatten ++ selectedNoise))

private def transposeOperationalMatrixType (type : MatrixTypeExpr) : MatrixTypeExpr := {
  type with rows := type.columns, columns := type.rows
}

private def transformOperationalFactor
    (transform : OperationalFactorTransform)
    (outputType : MatrixTypeExpr)
    (factor : OperationalFactorKey) : OperationalFactorKey :=
  let transformSummary (summary : OperationalBoundedFactorSummary) := {
    summary with
    matrixType := outputType
    metadata := { summary.metadata with knownZeroRows := none }
    provenance := summary.provenance ++ [.transform transform, .intermediateType outputType]
  }
  let leaf := match factor.leaf with
    | .boundedSummary origin summary =>
        let tokens := origin.tokens ++ [.transform transform, .intermediateType outputType]
        .boundedSummary { origin with tokens } (transformSummary summary)
    | .exactTransform tokens _ =>
        .exactTransform (tokens ++ [.transform transform, .intermediateType outputType]) outputType
    | primitive => primitive
  {
    factor with
    leaf
    transforms := factor.transforms ++ [transform]
    inputType := outputType
    outputType
    boundedSummary := factor.boundedSummary.map transformSummary
    protections := factor.protections.filter fun protection => match protection with
      | .relationOwner | .decompositionOwner => false
      | _ => true
    relations := []
  }

private def replaceOperationalFactorAt
    (index : Nat)
    (replacement : OperationalFactorKey)
    (factors : List OperationalFactorKey) : List OperationalFactorKey :=
  let rec visit : Nat → List OperationalFactorKey → List OperationalFactorKey
    | _, [] => []
    | 0, _ :: tail => replacement :: tail
    | remaining + 1, head :: tail => head :: visit remaining tail
  visit index factors

private def rowBoundaryIndex (product : OperationalProductKey) : Nat :=
  let rec visit : Nat → List OperationalProductMode → Nat
    | index, .leftPolynomialScalarBroadcast :: tail => visit (index + 1) tail
    | index, _ => index
  visit 0 product.modes

private def columnBoundaryIndex (product : OperationalProductKey) : Nat :=
  let rec skipRightScalars : Nat → List OperationalProductMode → Nat
    | index, [] => index
    | index, .rightPolynomialScalarBroadcast :: tail => skipRightScalars (index - 1) tail
    | index, _ => index
  skipRightScalars (product.factors.length - 1) product.modes.reverse

private def transformOperationalBoundary
    (axis : ConcatAxis)
    (part : Nat)
    (outputType : MatrixTypeExpr)
    (term : OperationalTerm) : Except OperationalFlatError OperationalTerm := do
  let applyAt (index : Nat) (transform : OperationalFactorTransform)
      (changeType : MatrixTypeExpr → MatrixTypeExpr) :
      Except OperationalFlatError OperationalProductKey := do
    let factor ← match term.product.factors[index]? with
      | some factor => pure factor
      | none => throw OperationalFlatError.malformedProduct
    let replacement := transformOperationalFactor transform (changeType factor.outputType) factor
    operationalProductFromFactors
      (replaceOperationalFactorAt index replacement term.product.factors)
  let product ← match axis with
    | .rows => applyAt (rowBoundaryIndex term.product) (.rowEmbed .rows part) (fun matrixType =>
        { matrixType with rows := outputType.rows })
    | .columns => applyAt (columnBoundaryIndex term.product) (.columnEmbed .columns part) (fun matrixType =>
        { matrixType with columns := outputType.columns })
    | .diagonal => do
        let rowProduct ← applyAt (rowBoundaryIndex term.product) (.rowEmbed .diagonal part)
          (fun matrixType => { matrixType with rows := outputType.rows })
        let index := columnBoundaryIndex rowProduct
        let factor ← match rowProduct.factors[index]? with
          | some factor => pure factor
          | none => throw .malformedProduct
        let replacement := transformOperationalFactor (.columnEmbed .diagonal part)
          { factor.outputType with columns := outputType.columns } factor
        operationalProductFromFactors (replaceOperationalFactorAt index replacement rowProduct.factors)
  pure { term with product }

private def concatOperationalPolynomials
    (axis : ConcatAxis)
    (outputType : MatrixTypeExpr)
    (inputs : List OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let rows ← inputs.zipIdx.mapM fun (terms, part) =>
    terms.mapM (transformOperationalBoundary axis part outputType)
  pure (normalizeOperationalTerms rows.flatten)

private def concatCanonicalRange (inputs : Array OperationalMatrixFact) : CanonicalRange :=
  if inputs.all (fun input => match input.canonicalRange with
      | .below _ => true
      | .unknown => false) then
    .below (inputs.foldl (fun result input => match input.canonicalRange with
      | .below value => max result value
      | .unknown => result) 0)
  else .unknown

private def concatConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (inputs : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← concatOperationalPolynomials axis matrixType
    (inputs.toList.map (·.polynomial)) |>.mapError (flatErrorAt nodeIndex)
  match ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
      (concatCanonicalRange inputs) with
  | .matrix output => pure output
  | _ => throw (.operandNotMatrix nodeIndex { node := nodeIndex, port := outputPort })

private def concatOperationalExprIds
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment) :
    OperationalExprArena → Array OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, roots, 0 => throw (.unsupportedOperationalExpr (roots[0]?.getD 0))
  | arena, roots, fuel + 1 => do
      if roots.isEmpty then throw (.invalidCount nodeIndex 0)
      let expressions ← roots.mapM fun root => match arena.get? root with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef root)
      if expressions.all fun expression => match expression.node with
          | .concrete _ => true
          | _ => false then
        let inputs ← expressions.mapM fun expression => match expression.node with
          | .concrete fact => pure fact
          | _ => throw (.unsupportedOperationalExpr 0)
        let output ← concatConcreteMatrixFacts nodeIndex outputPort axis matrixType environment inputs
        return arena.pushConcrete output
      let selected? := expressions.zipIdx.findSome? fun (expression, position) =>
        match expression.node with
        | .select selection branches => some (position, selection, branches)
        | _ => none
      let (position, selection, branches) ← match selected? with
        | some selected => pure selected
        | none => return arena.push { matrixType, node := .concat axis roots }
      let hasDifferentSelection := expressions.any fun expression => match expression.node with
        | .select candidate _ => candidate != selection
        | _ => false
      if hasDifferentSelection then
        return arena.push { matrixType, node := .concat axis roots }
      match branches with
      | .exact selectedBranches =>
          let aligned := expressions.all fun expression => match expression.node with
            | .select _ (.exact candidates) => candidates.size == selectedBranches.size
            | .select _ (.schemaEnvelope ..) => false
            | _ => true
          if !aligned then throw (.operationalExprTypeMismatch roots[position]! roots[position]!)
          let mut arena := arena
          let mut outputs : Array OperationalExprId := #[]
          for branch in [:selectedBranches.size] do
            let branchRoots := expressions.zipIdx.map fun (expression, inputIndex) =>
              match expression.node with
              | .select _ (.exact candidates) => candidates[branch]!
              | _ => roots[inputIndex]!
            let (nextArena, output) ← concatOperationalExprIds nodeIndex outputPort axis matrixType
              environment arena branchRoots fuel
            arena := nextArena
            outputs := outputs.push output
          arena.pushSelect selection (.exact outputs)
      | .schemaEnvelope count representative summary =>
          let aligned := expressions.all fun expression => match expression.node with
            | .select _ (.schemaEnvelope candidateCount _ candidateSummary) =>
                candidateCount == count && candidateSummary.uniformSchema.isSome
            | .select _ (.exact _) => false
            | _ => true
          if !aligned then throw (.operationalExprTypeMismatch roots[position]! roots[position]!)
          let representativeRoots := expressions.zipIdx.map fun (expression, inputIndex) =>
            match expression.node with
            | .select _ (.schemaEnvelope _ candidate _) => candidate
            | _ => roots[inputIndex]!
          let (arena, output) ← concatOperationalExprIds nodeIndex outputPort axis matrixType
            environment arena representativeRoots fuel
          let outputFact ← arena.concreteFact output
          let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr representative)
          arena.pushSelect selection (.schemaEnvelope count output outputSummary)

private def concatOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (inputs : Array OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let mut arena := arena
  let mut roots : Array OperationalExprId := #[]
  for input in inputs do
    let (nextArena, root) ← arena.pushMatrixFact input
    arena := nextArena
    roots := roots.push root
  let (finalArena, result) ← concatOperationalExprIds nodeIndex outputPort axis matrixType environment
    arena roots (arena.nodes.size + 1)
  pure (finalArena, .matrixExpr result)

private def transposeOperationalPolynomial
    (terms : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let terms ← terms.mapM fun term => do
    let factors := term.product.factors.reverse.map fun factor =>
      transformOperationalFactor .transpose (transposeOperationalMatrixType factor.outputType) factor
    let product ← operationalProductFromFactors factors
    pure { term with product }
  pure (normalizeOperationalTerms terms)

private def sliceOperationalPolynomial
    (rows columns : Option (IntExpr × IntExpr))
    (outputType : MatrixTypeExpr)
    (terms : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let terms ← terms.mapM fun term => do
    let term ← match rows with
      | none => pure term
      | some (start, stop) => do
          let index := rowBoundaryIndex term.product
          let factor ← match term.product.factors[index]? with
            | some factor => pure factor
            | none => throw .malformedProduct
          let replacement := transformOperationalFactor (.rowSlice start stop)
            { factor.outputType with rows := outputType.rows } factor
          let product ← operationalProductFromFactors
            (replaceOperationalFactorAt index replacement term.product.factors)
          pure { term with product }
    match columns with
    | none => pure term
    | some (start, stop) => do
        let index := columnBoundaryIndex term.product
        let factor ← match term.product.factors[index]? with
          | some factor => pure factor
          | none => throw .malformedProduct
        let replacement := transformOperationalFactor (.columnSlice start stop)
          { factor.outputType with columns := outputType.columns } factor
        let product ← operationalProductFromFactors
          (replaceOperationalFactorAt index replacement term.product.factors)
        pure { term with product }
  pure (normalizeOperationalTerms terms)

private def boundedStructuralTransformPolynomial
    (transform : OperationalFactorTransform)
    (outputType : MatrixTypeExpr)
    (input : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  if input.any operationalTermIsSignal then throw .cannotPreserveNoiseSeparation
  let compressed ← compressBoundedNoiseSum input
  compressed.mapM fun term => do
    match term.product.factors with
    | [factor] =>
        let transformed := transformOperationalFactor transform outputType factor
        pure { term with product := {
          factors := [transformed]
          modes := []
          outputType
        }}
    | _ => throw .malformedProduct

private def transformConcreteMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (operation : OperationalFactorTransform)
    (environment : ParamEnvironment)
    (input : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← match operation with
    | .negate =>
        let input ← retypeMatrixFact nodeIndex matrixType input environment
        pure (scaleOperationalPolynomial (-1) input.polynomial)
    | .transpose =>
        transposeOperationalPolynomial input.polynomial |>.mapError (flatErrorAt nodeIndex)
    | .rowSlice start stop =>
        sliceOperationalPolynomial (some (start, stop)) none matrixType input.polynomial
          |>.mapError (flatErrorAt nodeIndex)
    | .columnSlice start stop =>
        sliceOperationalPolynomial none (some (start, stop)) matrixType input.polynomial
          |>.mapError (flatErrorAt nodeIndex)
    | .reshape rows columns =>
        let outputParams ← match matrixType.evaluate environment
            (.constant input.matrixParams.maxCoefficientBound) with
          | some params => pure params
          | none => throw (.invalidMatrixParameters nodeIndex)
        if sameConcreteMatrixShape input.matrixParams outputParams then
          equivalentRetypeOperationalPolynomial matrixType input.polynomial
            |>.mapError (flatErrorAt nodeIndex)
        else
          boundedStructuralTransformPolynomial (.reshape rows columns) matrixType input.polynomial
            |>.mapError (flatErrorAt nodeIndex)
    | .constantCoefficient index =>
        boundedStructuralTransformPolynomial (.constantCoefficient index) matrixType input.polynomial
          |>.mapError (flatErrorAt nodeIndex)
    | .rowEmbed axis part | .columnEmbed axis part =>
        input.polynomial.mapM (transformOperationalBoundary axis part matrixType)
          |>.mapError (flatErrorAt nodeIndex)
  match ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
      input.canonicalRange with
  | .matrix output => pure output
  | _ => throw (.operandNotMatrix nodeIndex input.subject)

private def transformOperationalExprId
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (operation : OperationalFactorTransform)
    (environment : ParamEnvironment) :
    OperationalExprArena → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, root, 0 => throw (.unsupportedOperationalExpr root)
  | arena, root, fuel + 1 => do
      let expression ← match arena.get? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      match expression.node with
      | .concrete input =>
          let output ← transformConcreteMatrixFact nodeIndex outputPort matrixType operation
            environment input
          pure (arena.pushConcrete output)
      | .select selection (.exact branches) =>
          let mut arena := arena
          let mut outputs : Array OperationalExprId := #[]
          for branch in branches do
            let (nextArena, output) ← transformOperationalExprId nodeIndex outputPort matrixType
              operation environment arena branch fuel
            arena := nextArena
            outputs := outputs.push output
          arena.pushSelect selection (.exact outputs)
      | .select selection (.schemaEnvelope count representative summary) =>
          let (arena, output) ← transformOperationalExprId nodeIndex outputPort matrixType
            operation environment arena representative fuel
          let outputFact ← arena.concreteFact output
          let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr representative)
          arena.pushSelect selection (.schemaEnvelope count output outputSummary)
      | _ =>
          pure (arena.push { matrixType, node := .transform operation root })

private def transformOperationalExprFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (operation : OperationalFactorTransform)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (input : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, root) ← arena.pushMatrixFact input
  let (arena, result) ← transformOperationalExprId nodeIndex outputPort matrixType operation
    environment arena root (arena.nodes.size + 1)
  pure (arena, .matrixExpr result)

private def scaleConcreteMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (scalar : IntExpr)
    (scalarValues : List Int)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (input : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let input ← retypeMatrixFact nodeIndex matrixType input environment
  let first ← match scalarValues with
    | first :: _ => pure first
    | [] => throw (.invalidMatrixParameters nodeIndex)
  let polynomial ←
    if scalarValues.all (· == first) then
      pure (scaleOperationalPolynomial first input.polynomial)
    else
      multiplyOperationalPolynomials
        (parameterScalarPolynomial environment loopDomains scalar matrixType)
        input.polynomial |>.mapError (flatErrorAt nodeIndex)
  match ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
      input.canonicalRange with
  | .matrix output => pure output
  | _ => throw (.operandNotMatrix nodeIndex input.subject)

private def scaleOperationalExprId
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (scalar : IntExpr)
    (scalarValues : List Int)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain) :
    OperationalExprArena → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, root, 0 => throw (.unsupportedOperationalExpr root)
  | arena, root, fuel + 1 => do
      if !scalarValues.isEmpty && scalarValues.all (· == 1) then return (arena, root)
      let expression ← match arena.get? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      match expression.node with
      | .concrete input =>
          let output ← scaleConcreteMatrixFact nodeIndex outputPort matrixType scalar scalarValues
            environment loopDomains input
          pure (arena.pushConcrete output)
      | .select selection (.exact branches) =>
          let mut arena := arena
          let mut outputs : Array OperationalExprId := #[]
          for branch in branches do
            let (nextArena, output) ← scaleOperationalExprId nodeIndex outputPort matrixType scalar
              scalarValues environment loopDomains arena branch fuel
            arena := nextArena
            outputs := outputs.push output
          arena.pushSelect selection (.exact outputs)
      | .select selection (.schemaEnvelope count representative summary) =>
          let (arena, output) ← scaleOperationalExprId nodeIndex outputPort matrixType scalar
            scalarValues environment loopDomains arena representative fuel
          let outputFact ← arena.concreteFact output
          let outputSummary ← match recomputeSelectedMatrixSummary summary outputFact with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr representative)
          arena.pushSelect selection (.schemaEnvelope count output outputSummary)
      | _ => throw (.unsupportedOperationalExpr root)

private def scaleOperationalExprFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (scalar : IntExpr)
    (scalarValues : List Int)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (arena : OperationalExprArena)
    (input : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, root) ← arena.pushMatrixFact input
  let (arena, result) ← scaleOperationalExprId nodeIndex outputPort matrixType scalar scalarValues
    environment loopDomains arena root (arena.nodes.size + 1)
  pure (arena, .matrixExpr result)

/-- Group the already-derived exact signal part of a BGG encoding while retaining its bounded
noise as a separate top-level term.  The complete pre-grouping signal polynomial is embedded in a
flat token sequence, so this cannot create a false cancellation or hide bounded noise.  The paired
public-key/plaintext origins identify the one executable BGG value selected at runtime. -/
private def groupExactSignal
    (identityTokens : List OperationalCompressionToken)
    (vector : OperationalMatrixFact) :
    Except OperationalFlatError OperationalMatrixFact := do
  let signal := sortOperationalTerms (vector.polynomial.filter operationalTermIsSignal)
  let noise := vector.polynomial.filter operationalTermIsNoise
  if signal.isEmpty then
    return { vector with polynomial := (← compressBoundedNoiseSum noise) }
  let tokens := [.groupStart] ++ identityTokens ++ [.sumStart] ++
    signal.flatMap operationalProductTokens ++
    [.sumEnd, .intermediateType vector.matrixType, .groupEnd]
  let factor : OperationalFactorKey := {
    leaf := .exactTransform tokens vector.matrixType
    inputType := vector.matrixType
    outputType := vector.matrixType
    role := .large
  }
  let groupedSignal : OperationalTerm := {
    coefficient := 1
    product := { factors := [factor], modes := [], outputType := vector.matrixType }
  }
  let compressedNoise ← compressBoundedNoiseSum noise
  pure { vector with polynomial := groupedSignal :: compressedNoise }

private def groupBggEncodingSignal
    (vector publicKey plaintext : OperationalMatrixFact) :
    Except OperationalFlatError OperationalMatrixFact :=
  groupExactSignal
    [.primitive (.matrix publicKey.origin), .primitive (.matrix plaintext.origin)] vector

private def groupBggEncodingFact : OperationalFact → OperationalFact → OperationalFact →
    Except OperationalError OperationalFact
  | .matrix vector, .matrix publicKey, .matrix plaintext =>
      return .matrix (← groupBggEncodingSignal vector publicKey plaintext |>.mapError (.flat 0))
  | .familyUniform vectorBinder vectorCoordinate vector vectorCount,
      .familyUniform _ _ publicKey publicCount,
      .familyUniform _ _ plaintext plaintextCount => do
      if vectorCount != publicCount || vectorCount != plaintextCount then
        throw (.invalidDerivationAttachment "mxx-bgg" "encoding-family-pairing")
      return .familyUniform vectorBinder vectorCoordinate
        (← groupBggEncodingFact vector publicKey plaintext) vectorCount
  | _, _, _ => throw (.invalidDerivationAttachment "mxx-bgg" "encoding-family-pairing")

private def groupPublicKeySignalFact : OperationalFact → Except OperationalError OperationalFact
  | .matrix fact =>
      return .matrix (← groupExactSignal [] fact |>.mapError (.flat 0))
  | .familyUniform binder coordinate element count =>
      return .familyUniform binder coordinate (← groupPublicKeySignalFact element) count
  | _ => throw (.invalidDerivationAttachment "mxx-bgg" "public-key-signal-grouping")

/-- Promote the output of a separately validated exact Boolean carrier selection to one Large
signal factor. The validator below proves that this value is exactly `select(bit, zero, carrier)`
with a deterministic constant carrier, so this grouping cannot hide sampler noise. -/
private def groupProtocolBooleanSignalFact : OperationalFact → Except OperationalError OperationalFact
  | .matrix fact =>
      let tokens := [
        .groupStart,
        .primitive (.matrix fact.origin),
        .intermediateType fact.matrixType,
        .groupEnd
      ]
      let factor : OperationalFactorKey := {
        leaf := .exactTransform tokens fact.matrixType
        inputType := fact.matrixType
        outputType := fact.matrixType
        role := .large
      }
      let term : OperationalTerm := {
        coefficient := 1
        product := { factors := [factor], modes := [], outputType := fact.matrixType }
      }
      pure (.matrix { fact with polynomial := [term], metadata := {} })
  | _ =>
      throw (.invalidDerivationAttachment "mxx-correctness"
        "protocol-boolean-signal-grouping")

private def derivationAttachmentRole
    (attachment : DerivationAttachment)
    (role : String) : Except OperationalError WireRef :=
  match attachment.roles.filter (fun candidate => candidate.1 == role) with
  | [(_, wire)] => pure wire
  | _ => throw (.missingDerivationAttachmentRole attachment.ownerNamespace attachment.ruleName role)

private def validateDerivationAttachment
    (scope : Scope) (attachment : DerivationAttachment) : Except OperationalError Unit := do
  let isBggEncoding := attachment.ownerNamespace == "mxx-bgg" &&
    attachment.ruleName == "encoding-family-pairing"
  let isBggPublicKey := attachment.ownerNamespace == "mxx-bgg" &&
    attachment.ruleName == "public-key-signal-grouping"
  let isProtocolBoolean := attachment.ownerNamespace == "mxx-correctness" &&
    attachment.ruleName == "protocol-boolean-signal-grouping"
  if !(isBggEncoding || isBggPublicKey || isProtocolBoolean) then
    throw (.unknownDerivationAttachment attachment.ownerNamespace attachment.ruleName)
  let required := if isBggEncoding then ["vector", "public-key", "plaintext"]
    else if isProtocolBoolean then ["value", "selector", "zero", "carrier"]
    else ["value"]
  for role in required do
    let wire ← derivationAttachmentRole attachment role
    let outputCount ← match scope.nodes[wire.node]? with
      | some node => pure node.outputCount
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    if wire.port >= outputCount then
      throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
  if attachment.roles.length != required.length then
    throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
  if isProtocolBoolean then
    let valueWire ← derivationAttachmentRole attachment "value"
    let selectorWire ← derivationAttachmentRole attachment "selector"
    let zeroWire ← derivationAttachmentRole attachment "zero"
    let carrierWire ← derivationAttachmentRole attachment "carrier"
    let valueNode ← match scope.nodes[valueWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let indexWire ← match valueNode.kind, valueNode.arguments with
      | .select, [index, zero, carrier] =>
          if valueWire.port != 0 || zero != zeroWire || carrier != carrierWire then
            throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
          pure index
      | _, _ => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let indexNode ← match scope.nodes[indexWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    match indexNode.kind, indexNode.arguments with
    | .boolToInt, [source] =>
        if indexWire.port != 0 || source != selectorWire then
          throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    | _, _ => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let selectorNode ← match scope.nodes[selectorWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    if selectorWire.port != 0 || selectorNode.outputTypes != [.boolean] then
      throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    match selectorNode.kind with
    | .input _ => pure ()
    | _ => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let zeroNode ← match scope.nodes[zeroWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    let carrierNode ← match scope.nodes[carrierWire.node]? with
      | some value => pure value
      | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    match zeroNode.kind, carrierNode.kind with
    | .zeroMatrix zeroType, .constantMatrix carrierType coefficients =>
        if zeroWire.port != 0 || carrierWire.port != 0 || zeroType != carrierType ||
            coefficients.isEmpty || coefficients.all (· == .constant 0) ||
            valueNode.outputTypes != [.matrix zeroType] then
          throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
    | _, _ => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)

private def replaceOperationalFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef)
    (fact : OperationalFact) : Except OperationalError OperationalScopeFacts := do
  let outputs ← match facts.values[wire.node]? with
    | some outputs => pure outputs
    | none => throw (.missingOperand node wire)
  if wire.port >= outputs.size then throw (.missingOperand node wire)
  pure { facts with values := facts.values.set! wire.node (outputs.set! wire.port fact) }

private def applyDerivationAttachment
    (node : Nat)
    (attachment : DerivationAttachment)
    (facts : OperationalScopeFacts) : Except OperationalError OperationalScopeFacts := do
  if attachment.ownerNamespace == "mxx-bgg" &&
      attachment.ruleName == "encoding-family-pairing" then
    let vectorWire ← derivationAttachmentRole attachment "vector"
    let publicKeyWire ← derivationAttachmentRole attachment "public-key"
    let plaintextWire ← derivationAttachmentRole attachment "plaintext"
    let vector ← lookupFact node facts vectorWire
    let publicKey ← lookupFact node facts publicKeyWire
    let plaintext ← lookupFact node facts plaintextWire
    let grouped ← groupBggEncodingFact vector publicKey plaintext
    replaceOperationalFact node facts vectorWire grouped
  else if attachment.ownerNamespace == "mxx-correctness" &&
      attachment.ruleName == "protocol-boolean-signal-grouping" then
    let valueWire ← derivationAttachmentRole attachment "value"
    let value ← lookupFact node facts valueWire
    let grouped ← groupProtocolBooleanSignalFact value
    replaceOperationalFact node facts valueWire grouped
  else
    let valueWire ← derivationAttachmentRole attachment "value"
    let value ← lookupFact node facts valueWire
    let grouped ← groupPublicKeySignalFact value
    replaceOperationalFact node facts valueWire grouped

private def applyPreparedDerivationAttachments
    (node : Nat)
    (attachments : Array DerivationAttachment)
    (facts : OperationalScopeFacts) : Except OperationalError OperationalScopeFacts :=
  attachments.foldlM (init := facts) fun current attachment =>
    applyDerivationAttachment node attachment current

def availableRelation
    (node : Nat)
    (wire : WireRef)
    (fact : OperationalMatrixFact) : Except OperationalError OperationalMatrixRelation := do
  match fact.relations with
  | [] => throw (.missingRelation node wire)
  | [relation] =>
      let available := match relation with
        | .decomposition relation => relation.status == ReconstructionStatus.available
        | .preimage _ => true
      if available then pure relation else throw (OperationalError.unavailableRelation node wire)
  | _ => throw (.ambiguousRelation node wire)

def relationPublicIdentity : OperationalMatrixRelation → PublicMatrixIdentity
  | .decomposition relation => relation.publicIdentity
  | .preimage relation => relation.publicIdentity

def relationTarget : OperationalMatrixRelation → RelationTargetSummary
  | .decomposition relation => relation.inputSummary
  | .preimage relation => relation.targetSummary

private def publicIdentityIsLarge : PublicMatrixIdentity → Bool
  | .sampledTrapdoor .. | .gadget .. => true
  | .selected _ _ source => publicIdentityIsLarge source
  | .loopInstance _ _ source => publicIdentityIsLarge source

private def publicIdentityMaximum
    (residueCap : Int) : PublicMatrixIdentity → Int
  | .sampledTrapdoor .. => residueCap
  | .gadget .. => residueCap
  | .selected _ _ source => publicIdentityMaximum residueCap source
  | .loopInstance _ _ source => publicIdentityMaximum residueCap source

def rebindSubject (subject : WireRef) : OperationalFact → Except OperationalError OperationalFact
  | .matrix fact =>
      if fact.relations.all fun relation => match relation with
          | .decomposition relation => relation.producer == fact.origin
          | .preimage relation => relation.producer == fact.origin then
        pure (.matrix { fact with subject })
      else throw (.malformedRelation subject.node)
  | .trapdoor fact => pure (.trapdoor { fact with subject })
  | .integer fact => pure (.integer { fact with subject })
  | .bytes fact => pure (.bytes { fact with subject })
  | .familyUniform binder coordinate element count =>
      return .familyUniform binder coordinate (← rebindSubject subject element) count
  | .familyPacked elements count _ => do
      let elements ← elements.mapM (rebindSubject subject)
      return packedOperationalFamily elements count
  | .selectedMatrices family =>
      let selection := family.selection
      let branches := family.branches
      if branches.all fun branch => branch.relations.all fun relation => match relation with
          | .decomposition relation => relation.producer == branch.origin
          | .preimage relation => relation.producer == branch.origin then
        pure (.selectedMatrices (family.map selection fun branch => { branch with subject }))
      else throw (.malformedRelation subject.node)
  | fact => pure fact

private def namespaceFreshOrigin
    (scope : ScopeTemplateKey)
    (wire : WireRef) : MatrixOriginIdentity → MatrixOriginIdentity
  | .value originScope originWire =>
      if originScope == temporaryScope && originWire == wire then .value scope originWire
      else .value originScope originWire
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | origin@(.deterministicHash _) => origin
  | .loopInstance slot index source =>
      .loopInstance slot index (namespaceFreshOrigin scope wire source)
  | .selected binder selection source =>
      .selected binder selection (namespaceFreshOrigin scope wire source)

private def namespaceFreshPublicIdentity
    (scope : ScopeTemplateKey)
    (wire : WireRef) : PublicMatrixIdentity → PublicMatrixIdentity
  | .sampledTrapdoor originScope originWire =>
      if originScope == temporaryScope && originWire.node == wire.node then
        .sampledTrapdoor scope originWire
      else .sampledTrapdoor originScope originWire
  | identity@(.gadget ..) => identity
  | .selected binder selection source =>
      .selected binder selection (namespaceFreshPublicIdentity scope wire source)
  | .loopInstance slot index source =>
      .loopInstance slot index (namespaceFreshPublicIdentity scope wire source)

private def mapOperationalPrimitiveIdentity
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin) :
    OperationalPrimitiveIdentity → OperationalPrimitiveIdentity
  | .matrix identity => .matrix (mapOrigin identity)
  | .publicMatrix identity => .publicMatrix (mapPublic identity)
  | .value identity => .value (mapValue identity)
  | .parameterScalar environment domains value => .parameterScalar environment domains value
  | .identityMatrix type => .identityMatrix type
  | .selectionIndicator binder selection branch =>
      .selectionIndicator binder { index := mapValue selection.index } branch
  | .indexedArtifact input index => .indexedArtifact input index
  | .recurrenceResult scope node path => .recurrenceResult scope node path
  | .carriedInput path => .carriedInput path

private def mapOperationalCompressionToken
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr) :
    OperationalCompressionToken → OperationalCompressionToken
  | .primitive identity =>
      .primitive (mapOperationalPrimitiveIdentity mapOrigin mapPublic mapValue identity)
  | .summaryBound bound => .summaryBound (mapBound bound)
  | token => token

private def mapOperationalBoundedSummary
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (summary : OperationalBoundedFactorSummary) : OperationalBoundedFactorSummary := {
  summary with
  hardBound := mapBound summary.hardBound
  provenance := summary.provenance.map
    (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
}

private def mapOperationalPolynomial
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (mapRelation : OperationalMatrixRelation → OperationalMatrixRelation)
    (polynomial : OperationalPolynomial) : OperationalPolynomial :=
  polynomial.map fun term => { term with product := {
    term.product with factors := term.product.factors.map fun factor =>
      let boundedSummary := factor.boundedSummary.map
        (mapOperationalBoundedSummary mapOrigin mapPublic mapValue mapBound)
      let leaf := match factor.leaf with
        | .primitive identity => .primitive
            (mapOperationalPrimitiveIdentity mapOrigin mapPublic mapValue identity)
        | .boundedSummary origin summary =>
            let tokens := origin.tokens.map
              (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
            .boundedSummary { origin with tokens }
              (mapOperationalBoundedSummary mapOrigin mapPublic mapValue mapBound summary)
        | .exactTransform tokens type =>
            let tokens := tokens.map
              (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
            .exactTransform tokens type
      { factor with
        leaf
        boundedSummary
        relations := factor.relations.map mapRelation
      }
  }}

private def mapRelationSnapshotPolynomial
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr)
    (polynomial : RelationSnapshotPolynomial) : RelationSnapshotPolynomial :=
  polynomial.map fun term => { term with product := {
    term.product with factors := term.product.factors.map fun factor =>
      let boundedSummary := factor.boundedSummary.map
        (mapOperationalBoundedSummary mapOrigin mapPublic mapValue mapBound)
      let leaf := match factor.leaf with
        | .primitive identity => .primitive
            (mapOperationalPrimitiveIdentity mapOrigin mapPublic mapValue identity)
        | .boundedSummary origin summary =>
            let tokens := origin.tokens.map
              (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
            .boundedSummary { origin with tokens }
              (mapOperationalBoundedSummary mapOrigin mapPublic mapValue mapBound summary)
        | .exactTransform tokens type =>
            let tokens := tokens.map
              (mapOperationalCompressionToken mapOrigin mapPublic mapValue mapBound)
            .exactTransform tokens type
      { factor with leaf, boundedSummary }
  }}

private def namespaceFreshSummary
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (summary : RelationTargetSummary) : RelationTargetSummary := {
  summary with
  origin := namespaceFreshOrigin scope wire summary.origin
  polynomial := mapRelationSnapshotPolynomial
    (namespaceFreshOrigin scope wire)
    (namespaceFreshPublicIdentity scope wire)
    id id summary.polynomial
}

private def namespaceFreshRelation
    (scope : ScopeTemplateKey)
    (wire : WireRef) : OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with
      producer := namespaceFreshOrigin scope wire relation.producer
      inputOrigin := namespaceFreshOrigin scope wire relation.inputOrigin
      inputSummary := namespaceFreshSummary scope wire relation.inputSummary
      publicIdentity := namespaceFreshPublicIdentity scope wire relation.publicIdentity
    }
  | .preimage relation => .preimage {
      relation with
      producer := namespaceFreshOrigin scope wire relation.producer
      targetOrigin := namespaceFreshOrigin scope wire relation.targetOrigin
      targetSummary := namespaceFreshSummary scope wire relation.targetSummary
      publicIdentity := namespaceFreshPublicIdentity scope wire relation.publicIdentity
    }

private def shiftTargetPreviousDepth
    (target : RelationTargetSummary) : RelationTargetSummary := {
  target with
  totalHardBound := shiftPreviousDepth target.totalHardBound
  polynomial := mapRelationSnapshotPolynomial id id id shiftPreviousDepth target.polynomial
}

private def shiftRelationPreviousDepth :
    OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with inputSummary := shiftTargetPreviousDepth relation.inputSummary }
  | .preimage relation => .preimage {
      relation with targetSummary := shiftTargetPreviousDepth relation.targetSummary }

/-- Insert a new innermost recurrence state. References already present in invariant facts then
refer to the enclosing state, while the new carried placeholders continue to use depth zero. -/
private partial def shiftFactPreviousDepth : OperationalFact → OperationalFact
  | .matrix fact => .matrix {
      fact with
      totalHardBound := shiftPreviousDepth fact.totalHardBound
      relations := fact.relations.map shiftRelationPreviousDepth
      polynomial := mapOperationalPolynomial id id id shiftPreviousDepth
        shiftRelationPreviousDepth fact.polynomial
    }
  | .trapdoor fact => .trapdoor { fact with maximum := shiftPreviousDepth fact.maximum }
  | .integer fact => .integer {
      fact with
      lowerExpression := shiftPreviousDepth fact.lowerExpression
      upperExpression := shiftPreviousDepth fact.upperExpression
    }
  | .familyUniform binder coordinate element count =>
      .familyUniform binder coordinate (shiftFactPreviousDepth element) count
  | .familyPacked elements count _ =>
      packedOperationalFamily (elements.map shiftFactPreviousDepth) count
  | .selectedMatrices family =>
      let selection := family.selection
      let shiftBranch (branch : OperationalMatrixFact) :=
        match shiftFactPreviousDepth (.matrix branch) with
        | .matrix shifted => shifted
        | _ => branch
      .selectedMatrices (family.map selection shiftBranch)
  | fact => fact

private def namespaceFreshValueOrigin
    (scope : ScopeTemplateKey)
    (wire : WireRef) : OperationalValueOrigin → OperationalValueOrigin
  | .local originScope originWire =>
      if originScope == temporaryScope && originWire == wire then .local scope originWire
      else .local originScope originWire
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | .loopInstance slot index source =>
      .loopInstance slot index (namespaceFreshValueOrigin scope wire source)
  | .selected binder index source =>
      .selected binder (namespaceFreshValueOrigin scope wire index)
        (namespaceFreshValueOrigin scope wire source)

/-- Namespace only identities created by this exact output.  Caller origins transported through
an input are deliberately left unchanged. -/
partial def namespaceFreshOutput
    (scope : ScopeTemplateKey)
    (wire : WireRef) : OperationalFact → OperationalFact
  | .matrix fact => .matrix {
      fact with
      origin := namespaceFreshOrigin scope wire fact.origin
      identity := fact.identity.map (namespaceFreshPublicIdentity scope wire)
      relations := fact.relations.map (namespaceFreshRelation scope wire)
      polynomial := mapOperationalPolynomial
        (namespaceFreshOrigin scope wire)
        (namespaceFreshPublicIdentity scope wire)
        (namespaceFreshValueOrigin scope wire)
        id
        (namespaceFreshRelation scope wire)
        fact.polynomial
    }
  | .trapdoor fact => .trapdoor {
      fact with publicIdentity := namespaceFreshPublicIdentity scope wire fact.publicIdentity
    }
  | .integer fact => .integer {
      fact with
      origin := match fact.origin with
        | .local originScope originWire =>
            if originScope == temporaryScope && originWire == wire then .local scope wire
            else fact.origin
        | .protocolInput _ => fact.origin
        | .protocolFamilyElement _ _ => fact.origin
        | .loopInstance _ _ _ => fact.origin
        | .selected _ _ _ => fact.origin
    }
  | .bytes fact => .bytes {
      fact with
      origin := match fact.origin with
        | .local originScope originWire =>
            if originScope == temporaryScope && originWire == wire then .local scope wire
            else fact.origin
        | .protocolInput _ => fact.origin
        | .protocolFamilyElement _ _ => fact.origin
        | .loopInstance _ _ _ => fact.origin
        | .selected _ _ _ => fact.origin
    }
  | .selectedMatrices family =>
      let selection := family.selection
      let namespaceBranch (branch : OperationalMatrixFact) :=
        match namespaceFreshOutput scope wire (.matrix branch) with
        | .matrix namespaced => namespaced
        | _ => branch
      .selectedMatrices (family.map (namespaceFreshValueOrigin scope wire selection)
        namespaceBranch)
  | fact => fact

partial def factHasRelation : OperationalFact → Bool
  | .matrix fact => !fact.relations.isEmpty || fact.polynomial.any fun term =>
      term.product.factors.any fun factor => !factor.relations.isEmpty
  | .familyUniform _ _ element _ => factHasRelation element
  | .familyPacked elements _ _ => elements.any factHasRelation
  | .selectedMatrices family =>
      family.branches.any fun branch => factHasRelation (.matrix branch)
  | _ => false

def packedFacts : List OperationalFact → OperationalFact
  | elements => packedOperationalFamily elements.toArray

def unpackPackedFacts : OperationalFact → Option (Array OperationalFact)
  | .familyPacked elements _ _ => some elements
  | _ => none

private def booleanFamilyCount (fact : OperationalFact) : Option Int :=
  match fact with
  | .familyUniform _ _ .boolean count => some count
  | packed => do
      let elements ← unpackPackedFacts packed
      if elements.all (· == .boolean) then some (Int.ofNat elements.size) else none

private def instantiateHashIdentityLoopIndex
    (slot index : Nat) (identity : DeterministicHashIdentity) : DeterministicHashIdentity :=
  { identity with
    parameterEnvironment := replaceLoopIndex identity.parameterEnvironment slot index
    parameterDomains := instantiateParameterDomains slot index identity.parameterDomains
  }

private def instantiateOriginLoopIndex
    (slot index : Nat) : MatrixOriginIdentity → MatrixOriginIdentity
  | .value scope wire => .loopInstance slot index (.value scope wire)
  | .protocolInput input => .protocolInput input
  | .protocolFamilyElement input familyIndex => .protocolFamilyElement input familyIndex
  | .deterministicHash identity =>
      .deterministicHash (instantiateHashIdentityLoopIndex slot index identity)
  | .loopInstance existingSlot existingIndex source =>
      .loopInstance existingSlot existingIndex (instantiateOriginLoopIndex slot index source)
  | .selected binder selection source =>
      .selected binder selection (instantiateOriginLoopIndex slot index source)

private def instantiateValueOriginLoopIndex
    (slot index : Nat) : OperationalValueOrigin → OperationalValueOrigin
  | .local scope wire => .loopInstance slot index (.local scope wire)
  | .protocolInput input => .protocolInput input
  | .protocolFamilyElement input familyIndex => .protocolFamilyElement input familyIndex
  | .loopInstance existingSlot existingIndex source =>
      .loopInstance existingSlot existingIndex
        (instantiateValueOriginLoopIndex slot index source)
  | .selected binder selection source =>
      .selected binder (instantiateValueOriginLoopIndex slot index selection)
        (instantiateValueOriginLoopIndex slot index source)

private def instantiatePublicIdentityLoopIndex
    (slot index : Nat) : PublicMatrixIdentity → PublicMatrixIdentity
  | identity@(.gadget ..) => identity
  | .sampledTrapdoor scope wire =>
      .loopInstance slot index (.sampledTrapdoor scope wire)
  | .selected binder selection source =>
      .selected binder selection (instantiatePublicIdentityLoopIndex slot index source)
  | .loopInstance existingSlot existingIndex source =>
      .loopInstance existingSlot existingIndex
        (instantiatePublicIdentityLoopIndex slot index source)

private def instantiateTargetLoopIndex
    (slot index : Nat) (target : RelationTargetSummary) : RelationTargetSummary :=
  { target with
    origin := instantiateOriginLoopIndex slot index target.origin
    totalHardBound := instantiateBoundLoopIndex slot index target.totalHardBound
    polynomial := mapRelationSnapshotPolynomial
      (instantiateOriginLoopIndex slot index)
      (instantiatePublicIdentityLoopIndex slot index)
      (instantiateValueOriginLoopIndex slot index)
      (instantiateBoundLoopIndex slot index)
      target.polynomial
  }

private def instantiateRelationLoopIndex
    (slot index : Nat) : OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with
      producer := instantiateOriginLoopIndex slot index relation.producer
      publicIdentity := instantiatePublicIdentityLoopIndex slot index relation.publicIdentity
      inputOrigin := instantiateOriginLoopIndex slot index relation.inputOrigin
      inputSummary := instantiateTargetLoopIndex slot index relation.inputSummary
    }
  | .preimage relation => .preimage {
      relation with
      producer := instantiateOriginLoopIndex slot index relation.producer
      publicIdentity := instantiatePublicIdentityLoopIndex slot index relation.publicIdentity
      targetOrigin := instantiateOriginLoopIndex slot index relation.targetOrigin
      targetSummary := instantiateTargetLoopIndex slot index relation.targetSummary
    }

private partial def instantiateFactLoopIndex (slot index : Nat) : OperationalFact → OperationalFact
  | .matrix fact => .matrix {
      fact with
      origin := instantiateOriginLoopIndex slot index fact.origin
      totalHardBound := instantiateBoundLoopIndex slot index fact.totalHardBound
      identity := fact.identity.map (instantiatePublicIdentityLoopIndex slot index)
      relations := fact.relations.map (instantiateRelationLoopIndex slot index)
      polynomial := mapOperationalPolynomial
        (instantiateOriginLoopIndex slot index)
        (instantiatePublicIdentityLoopIndex slot index)
        (instantiateValueOriginLoopIndex slot index)
        (instantiateBoundLoopIndex slot index)
        (instantiateRelationLoopIndex slot index)
        fact.polynomial
    }
  | .trapdoor fact => .trapdoor {
      fact with
      maximum := instantiateBoundLoopIndex slot index fact.maximum
      publicIdentity := instantiatePublicIdentityLoopIndex slot index fact.publicIdentity }
  | .integer fact => .integer {
      fact with
      origin := instantiateValueOriginLoopIndex slot index fact.origin
      lowerExpression := instantiateBoundLoopIndex slot index fact.lowerExpression
      upperExpression := instantiateBoundLoopIndex slot index fact.upperExpression
    }
  | .bytes fact => .bytes {
      fact with origin := instantiateValueOriginLoopIndex slot index fact.origin }
  | .familyUniform binder coordinate element count =>
      .familyUniform binder coordinate (instantiateFactLoopIndex slot index element) count
  | .familyPacked elements count _ =>
      packedOperationalFamily (elements.map (instantiateFactLoopIndex slot index)) count
  | .selectedMatrices family =>
      let selection := family.selection
      let instantiateBranch (branch : OperationalMatrixFact) :=
        match instantiateFactLoopIndex slot index (.matrix branch) with
        | .matrix instantiated => instantiated
        | _ => branch
      .selectedMatrices (family.map (instantiateValueOriginLoopIndex slot index selection)
        instantiateBranch)
  | fact => fact

private def selectProtocolValueOrigin
    (index : Nat) : OperationalValueOrigin → OperationalValueOrigin
  | .protocolInput input => .protocolFamilyElement input index
  | .loopInstance slot lane source =>
      .loopInstance slot lane (selectProtocolValueOrigin index source)
  | .selected binder selection source =>
      .selected binder (selectProtocolValueOrigin index selection)
        (selectProtocolValueOrigin index source)
  | origin => origin

private def selectProtocolHashIdentity
    (index : Nat) (identity : DeterministicHashIdentity) : DeterministicHashIdentity :=
  { identity with
    keyOrigin := selectProtocolValueOrigin index identity.keyOrigin
    trailingIntegerOrigins := identity.trailingIntegerOrigins.map (selectProtocolValueOrigin index)
  }

private def selectProtocolMatrixOrigin
    (index : Nat) : MatrixOriginIdentity → MatrixOriginIdentity
  | .protocolInput input => .protocolFamilyElement input index
  | .deterministicHash identity =>
      .deterministicHash (selectProtocolHashIdentity index identity)
  | .loopInstance slot lane source =>
      .loopInstance slot lane (selectProtocolMatrixOrigin index source)
  | .selected binder selection source =>
      .selected binder selection (selectProtocolMatrixOrigin index source)
  | origin => origin

private def selectProtocolTarget
    (index : Nat) (target : RelationTargetSummary) : RelationTargetSummary :=
  { target with
    origin := selectProtocolMatrixOrigin index target.origin
    polynomial := mapRelationSnapshotPolynomial
      (selectProtocolMatrixOrigin index) id (selectProtocolValueOrigin index) id
      target.polynomial
  }

private def selectProtocolRelation
    (index : Nat) : OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with
      producer := selectProtocolMatrixOrigin index relation.producer
      inputOrigin := selectProtocolMatrixOrigin index relation.inputOrigin
      inputSummary := selectProtocolTarget index relation.inputSummary
    }
  | .preimage relation => .preimage {
      relation with
      producer := selectProtocolMatrixOrigin index relation.producer
      targetOrigin := selectProtocolMatrixOrigin index relation.targetOrigin
      targetSummary := selectProtocolTarget index relation.targetSummary
    }

private partial def selectProtocolFamilyElement (index : Nat) : OperationalFact → OperationalFact
  | .matrix fact => .matrix {
      fact with
      origin := selectProtocolMatrixOrigin index fact.origin
      relations := fact.relations.map (selectProtocolRelation index)
      polynomial := mapOperationalPolynomial
        (selectProtocolMatrixOrigin index)
        id
        (selectProtocolValueOrigin index)
        id
        (selectProtocolRelation index)
        fact.polynomial
    }
  | .integer fact => .integer {
      fact with origin := selectProtocolValueOrigin index fact.origin }
  | .bytes fact => .bytes {
      fact with origin := selectProtocolValueOrigin index fact.origin }
  | .familyUniform binder coordinate element count =>
      .familyUniform binder coordinate (selectProtocolFamilyElement index element) count
  | .familyPacked elements count _ =>
      packedOperationalFamily (elements.map (selectProtocolFamilyElement index)) count
  | .selectedMatrices family =>
      let selection := family.selection
      let selectBranch (branch : OperationalMatrixFact) :=
        match selectProtocolFamilyElement index (.matrix branch) with
        | .matrix selected => selected
        | _ => branch
      .selectedMatrices (family.map (selectProtocolValueOrigin index selection) selectBranch)
  | fact => fact

private def joinCanonicalRanges : List CanonicalRange → CanonicalRange
  | [] => .unknown
  | ranges =>
      if ranges.all (fun range => match range with | .below _ => true | .unknown => false) then
        .below (ranges.foldl (fun result range => match range with
          | .below value => max result value
          | .unknown => result) 0)
      else .unknown

/-- A dynamic family lookup may denote any element.  It therefore joins numeric information and
drops all producer-specific identities and modular relations. -/
def joinDynamicFacts
    (node : Nat)
    (subject : WireRef)
    (facts : List OperationalFact)
    (selection : Option OperationalValueOrigin := none) : Except OperationalError OperationalFact := do
  match facts with
  | [] => throw (.invalidCount node 0)
  | .matrix first :: tail =>
      let matrices ← tail.mapM fun fact => match fact with
        | .matrix value => pure value
        | _ => throw (.loopInputModeMismatch node 0)
      if !(matrices.all fun value =>
          value.matrixParams.modulus == first.matrixParams.modulus &&
          value.matrixParams.ringDimension == first.matrixParams.ringDimension &&
          value.matrixParams.rows == first.matrixParams.rows &&
          value.matrixParams.columns == first.matrixParams.columns) then
        throw (.loopInputModeMismatch node 0)
      let all := first :: matrices
      let selection ← match selection with
        | some value => pure value
        | none => throw (.loopInputModeMismatch node 0)
      let polynomial ← selectOperationalPolynomials temporaryScope node selection first.matrixType
        (all.map (·.polynomial)) |>.mapError (flatErrorAt node)
      let totalHardBound := maximumOperationalBounds (all.map (·.totalHardBound))
      pure (.matrix {
        first with
        subject
        origin := .value temporaryScope subject
        totalHardBound
        polynomial
        canonicalRange := joinCanonicalRanges (all.map (·.canonicalRange))
        identity := none
        relations := []
      })
  | .integer first :: tail =>
      let intervals ← tail.mapM fun fact => match fact with
        | .integer value => pure value
        | _ => throw (.loopInputModeMismatch node 0)
      pure (.integer {
        subject
        origin := .local temporaryScope subject
        lower := intervals.foldl (fun value interval => min value interval.lower) first.lower
        upper := intervals.foldl (fun value interval => max value interval.upper) first.upper
        lowerExpression := intervals.foldl
          (fun value interval => .minimum value interval.lowerExpression) first.lowerExpression
        upperExpression := intervals.foldl
          (fun value interval => .maximum value interval.upperExpression) first.upperExpression
      })
  | first :: tail =>
      if tail.all (· == first) then rebindSubject subject first
      else throw (.loopInputModeMismatch node 0)

/-- Select one element of a uniform family by the exact executable index wire.  The wrapper is
structural: two selections compare equal only when the family binder and index-wire instance are
identical.  Arithmetic equivalence of two index computations is never inferred. -/
private def dynamicSelectionScope : OperationalValueOrigin → ScopeTemplateKey
  | .local scope _ => scope
  | .protocolInput _ | .protocolFamilyElement _ _ => temporaryScope
  | .loopInstance _ _ source => dynamicSelectionScope source
  | .selected _ index _ => dynamicSelectionScope index

private def selectDynamicValueOrigin
    (binder : FamilyTemplateBinder)
    (selection : OperationalValueOrigin)
    (source : OperationalValueOrigin) : OperationalValueOrigin :=
  .selected binder selection source

private def selectDynamicMatrixOrigin
    (binder : FamilyTemplateBinder)
    (selection : OperationalValueOrigin)
    (source : MatrixOriginIdentity) : MatrixOriginIdentity :=
  .selected binder { index := selection } source

private def selectDynamicTarget
    (binder : FamilyTemplateBinder)
    (selection : OperationalValueOrigin)
    (target : RelationTargetSummary) : RelationTargetSummary := {
  target with
  origin := selectDynamicMatrixOrigin binder selection target.origin
  polynomial := mapRelationSnapshotPolynomial
    (selectDynamicMatrixOrigin binder selection)
    (fun identity => .selected binder { index := selection } identity)
    (selectDynamicValueOrigin binder selection) id target.polynomial
}

private def selectDynamicRelation
    (binder : FamilyTemplateBinder)
    (selection : OperationalValueOrigin) :
    OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with
      producer := selectDynamicMatrixOrigin binder selection relation.producer
      publicIdentity := .selected binder { index := selection } relation.publicIdentity
      inputOrigin := selectDynamicMatrixOrigin binder selection relation.inputOrigin
      inputSummary := selectDynamicTarget binder selection relation.inputSummary
    }
  | .preimage relation => .preimage {
      relation with
      producer := selectDynamicMatrixOrigin binder selection relation.producer
      publicIdentity := .selected binder { index := selection } relation.publicIdentity
      targetOrigin := selectDynamicMatrixOrigin binder selection relation.targetOrigin
      targetSummary := selectDynamicTarget binder selection relation.targetSummary
    }

private def selectDynamicMatrixFact
    (binder : FamilyTemplateBinder)
    (selection : OperationalValueOrigin)
    (subject : WireRef)
    (fact : OperationalMatrixFact) : OperationalMatrixFact := {
  fact with
  subject
  origin := selectDynamicMatrixOrigin binder selection fact.origin
  identity := fact.identity.map fun identity =>
    .selected binder { index := selection } identity
  relations := fact.relations.map (selectDynamicRelation binder selection)
  polynomial := mapOperationalPolynomial
    (selectDynamicMatrixOrigin binder selection)
    (fun identity => .selected binder { index := selection } identity)
    (selectDynamicValueOrigin binder selection)
    id
    (selectDynamicRelation binder selection)
    fact.polynomial
}

def selectDynamicUniformFact
    (binder : FamilyTemplateBinder)
    (selection : OperationalValueOrigin)
    (subject : WireRef) : OperationalFact → Except OperationalError OperationalFact
  | .matrix fact => pure (.matrix (selectDynamicMatrixFact binder selection subject fact))
  | .trapdoor fact => pure (.trapdoor {
      fact with
      subject
      publicIdentity := .selected binder { index := selection } fact.publicIdentity
    })
  | .integer fact =>
      let selected := { fact with
        origin := selectDynamicValueOrigin binder selection fact.origin }
      pure (.integer { selected with subject })
  | .bytes fact =>
      let selected := { fact with
        origin := selectDynamicValueOrigin binder selection fact.origin }
      pure (.bytes { selected with subject })
  | fact => rebindSubject subject fact

/-- Remove only producer identities from a relation-free branch.  The remaining value is the
complete operational schema used by every later bound rule: ordered products, transforms, roles,
matrix types, bound expressions, and metadata all remain exact. -/
private def relationFreeUniformSchema?
    (fact : OperationalMatrixFact) : Option UniformMatrixSchema := do
  if factHasRelation (.matrix fact) then none else
    some (operationalUniformSchema fact)

private def uniformBoundaryRelationPair
    (left right : OperationalMatrixFact) : Bool :=
  match left.polynomial, right.polynomial with
  | [{ product := { factors := leftFactors, .. }, .. }],
      [{ product := { factors := rightFactors, .. }, .. }] =>
      match leftFactors.getLast?, rightFactors.head? with
      | some leftFactor, some rightFactor =>
          (matchingFactorRelation? leftFactor rightFactor).isSome
      | _, _ => false
  | _, _ => false

private def uniformBoundaryRelationPairs
    (left right : SelectedMatrixFamily) : Bool :=
  left.count == right.count &&
    left.summary.uniformSchema.isSome && right.summary.uniformSchema.isSome &&
    left.summary.sharedLastPublicIdentity.isSome &&
    left.summary.sharedLastPublicIdentity == right.summary.sharedFirstRelationPublicIdentity

private def uniformBoundaryRelationsWithSharedLeft
    (left : OperationalMatrixFact)
    (right : SelectedMatrixFamily) : Bool :=
  let identity := boundaryLastPublicIdentity? left
  right.summary.uniformSchema.isSome && identity.isSome &&
    identity == right.summary.sharedFirstRelationPublicIdentity

/-- Collapse a selected family only after every branch-local relation has been consumed and every
complete branch has the same operational schema modulo producer identity.  The representative is
wrapped in the executable selection identity, so it cannot spuriously match an ordinary branch's
identity in a later relation rewrite. -/
private def compressUniformSelectedMatrices
    (node : Nat)
    (subject : WireRef)
    (selection : OperationalValueOrigin)
    (branches : Array OperationalMatrixFact)
    (count : Nat := branches.size)
    (representsLoopLanes : Bool := false) : Except OperationalError OperationalFact := do
  let first ← match branches[0]? with
    | some first => pure first
    | none => throw (.invalidCount node 0)
  let family := if count == branches.size then
      { selectedMatrixFamily selection branches with representsLoopLanes }
    else selectedMatrixEnvelope selection count first (representsLoopLanes := representsLoopLanes)
  match family.summary.uniformSchema with
  | some _ =>
      if family.summary.relationFree then
        let binder : FamilyTemplateBinder := {
          owner := dynamicSelectionScope selection
          producerNode := node
          binderSlot := 0
        }
        selectDynamicUniformFact binder selection subject (.matrix first)
      else
        pure (.selectedMatrices family)
  | none => pure (.selectedMatrices family)

/-- Build the one-iteration template consumed by a parallel-loop body. Packed executable
families are joined through the existing exact-one selection representation: signal branches
retain indicators and bounded branches use their maximum bound. Offset families deliberately use
a distinct selection origin, which forgets cross-family correlation rather than inventing it. -/
def loopTemplateArgumentFact
    (node argument count : Nat)
    (mode : LoopInputMode)
    (fact : OperationalFact) : Except OperationalError OperationalFact := do
  match mode with
  | .broadcast => pure fact
  | .zip | .zipOffset _ =>
      match fact with
      | .familyUniform _ _ element familyCount =>
          let offset := match mode with | .zipOffset value => value | _ => 0
          if count + offset > familyCount.toNat then
            throw (.loopInputModeMismatch node argument)
          else pure element
      | .familyPacked elements familyCount matrixSummary =>
          let offset := match mode with
            | .zip => 0
            | .zipOffset value => value
            | .broadcast => 0
          if familyCount < count + offset then
            throw (.loopInputModeMismatch node argument)
          let baseSelection : OperationalValueOrigin :=
            .local temporaryScope { node, port := 0 }
          let selection := match mode with
            | .zip => baseSelection
            | .zipOffset value => .loopInstance argument value baseSelection
            | .broadcast => baseSelection
          match matrixSummary with
          | some summary =>
              let representativeIndex := if elements.size == familyCount then offset else 0
              let representative ← match elements[representativeIndex]? with
                | some (.matrix matrix) => pure matrix
                | _ => throw (.loopInputModeMismatch node argument)
              if summary.relationFree then
                compressUniformSelectedMatrices node { node, port := argument } selection
                  #[representative] count true
              else pure (.selectedMatrices
                (selectedMatrixEnvelope selection count representative summary true))
          | none =>
              if elements.size < count + offset then
                throw (.loopInputModeMismatch node argument)
              let branches := ((elements.extract offset (offset + count)).toList)
              if branches.any factHasRelation then do
                let matrices ← branches.toArray.mapM fun branch =>
                  match branch with
                  | .matrix matrix => pure matrix
                  | _ => throw (.loopInputModeMismatch node argument)
                pure (.selectedMatrices
                  { selectedMatrixFamily selection matrices with representsLoopLanes := true })
              else
                joinDynamicFacts node { node, port := argument } branches (some selection)
      | _ => throw (.loopInputModeMismatch node argument)

private def liftSelectedUnaryMatrix
    (node : Nat)
    (wire : WireRef)
    (input : OperationalFact)
    (operation : OperationalMatrixFact → Except OperationalError OperationalFact) :
    Except OperationalError OperationalFact := do
  match input with
  | .matrix matrix => operation matrix
  | .selectedMatrices family => do
      let selection := family.selection
      let branches := family.branches
      let outputs ← branches.mapM operation
      let matrices ← outputs.mapM fun output => match output with
        | .matrix matrix => pure matrix
        | _ => throw (.operandNotMatrix node wire)
      compressUniformSelectedMatrices node wire selection matrices family.count
        family.representsLoopLanes
  | _ => throw (.operandNotMatrix node wire)

private structure AlignedMatrixInputs where
  selection : Option OperationalValueOrigin
  rows : Array (List OperationalMatrixFact)

private def alignSelectedMatrixInputs
    (node : Nat)
    (inputs : List (WireRef × OperationalFact)) :
    Except OperationalError AlignedMatrixInputs := do
  let selected := inputs.filterMap fun (_, input) => match input with
    | .selectedMatrices family => some (family.selection, family.branches.size)
    | _ => none
  match selected with
  | [] =>
      let row ← inputs.mapM fun (wire, input) => match input with
        | .matrix matrix => pure matrix
        | _ => throw (.operandNotMatrix node wire)
      pure { selection := none, rows := #[row] }
  | (selection, count) :: tail => do
      if tail.any fun candidate => candidate.1 != selection || candidate.2 != count then
        throw (.selectedFamilyOperationUnsupported node)
      let mut rows := #[]
      for branch in [:count] do
        let row ← inputs.mapM fun (wire, input) => match input with
          | .matrix matrix => pure matrix
          | .selectedMatrices family => do
              if family.selection != selection then
                throw (.selectedFamilyOperationUnsupported node)
              match family.branches[branch]? with
              | some matrix => pure matrix
              | none => throw (.selectedFamilyOperationUnsupported node)
          | _ => throw (.operandNotMatrix node wire)
        rows := rows.push row
      pure { selection := some selection, rows }

private def finishAlignedMatrixOutputs
    (node : Nat)
    (wire : WireRef)
    (selection : Option OperationalValueOrigin)
    (outputs : Array OperationalFact) : Except OperationalError OperationalFact := do
  match selection with
  | none => match outputs[0]? with
      | some output => pure output
      | none => throw (.selectedFamilyOperationUnsupported node)
  | some selection =>
      let branches ← outputs.mapM fun output => match output with
        | .matrix matrix => pure matrix
        | _ => throw (.operandNotMatrix node wire)
      compressUniformSelectedMatrices node wire selection branches

def genericNodeFact
    (scopeKey : ScopeTemplateKey)
    (nodeIndex : Nat)
    (node : Node)
    (rule : DerivationRule)
    (outputPort : Nat)
    (outputType : WireTypeExpr)
    (facts : OperationalScopeFacts)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalFact := do
  let matrixType? := match outputType with
    | .matrix matrixType | .preimage matrixType => some matrixType
    | _ => none
  let embeddedMatrixType? := match node.kind with
    | .zeroMatrix matrixType
    | .identityMatrix matrixType
    | .constantMatrix matrixType _
    | .unitRowMatrix matrixType _
    | .unitColumnMatrix matrixType _
    | .gadgetMatrix matrixType _
    | .smallGadgetMatrix matrixType _
    | .powerOfBaseMatrix matrixType _ _
    | .rotationMatrix matrixType _
    | .uniformResidueSample matrixType
    | .uniformIntervalSample matrixType _ _
    | .gaussianSample matrixType _
    | .preimageSample matrixType _
    | .packPolynomialCoefficients matrixType _ => some matrixType
    | .trapdoorSample matrixType _ =>
        if outputPort == 0 then some matrixType else none
    | .hashSample matrixType .plain _ _ _ _ _ _ => some matrixType
    | .hashSample _ .decomposed _ _ _ _ _ _
    | .hashSample _ .smallDecomposed _ _ _ _ _ _ => none
    | _ => none
  match embeddedMatrixType?, matrixType? with
  | some embedded, some output =>
      if embedded != output then throw (.outputTypeMismatch nodeIndex)
  | some _, none => throw (.outputTypeMismatch nodeIndex)
  | none, _ => pure ()
  let outputIsInteger := match outputType with
    | .integer | .constantInt => true
    | _ => false
  let outputIsBoolean := match outputType with
    | .boolean | .constantBool => true
    | _ => false
  match node.kind with
  | .constantInt _ | .evaluateInt _ | .boolToInt | .intBinary _ | .extractCoefficient _ =>
      if !outputIsInteger then throw (.outputTypeMismatch nodeIndex)
  | .constantBool _ | .intCompare _ | .bitExtract _ | .thresholdDecodeBool _ _ _ =>
      if !outputIsBoolean then throw (.outputTypeMismatch nodeIndex)
  | _ => pure ()
  match matrixType? with
  | some matrixType =>
      let _ ← evaluateIntInvariant environment loopDomains matrixType.modulus
      let _ ← evaluateIntInvariant environment loopDomains matrixType.ringDimension
      let _ ← evaluateIntInvariant environment loopDomains matrixType.rows
      let _ ← evaluateIntInvariant environment loopDomains matrixType.columns
      pure ()
  | none => pure ()
  match node.kind, matrixType? with
  | .input _, _ =>
      defaultFact nodeIndex outputPort outputType environment
  | .constantInt value, none =>
      if node.arguments.isEmpty then integerFact nodeIndex outputPort value value
      else throw (.unsupportedOutputArity nodeIndex node.arguments.length)
  | .evaluateInt value, none =>
      if !node.arguments.isEmpty then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      integerFactWithExpressions nodeIndex outputPort
        (← evaluateIntMinimum environment loopDomains value)
        (← evaluateIntMaximum environment loopDomains value)
        (.contextual .minimum environment loopDomains value)
        (.contextual .maximum environment loopDomains value)
  | .boolToInt, none =>
      let inputWire ← match node.arguments with
        | [wire] => pure wire
        | _ => throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      requireBooleanFact nodeIndex facts inputWire
      integerFact nodeIndex outputPort 0 1
  | .intBinary operation, none =>
      if node.arguments.length != 2 then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let leftWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let rightWire ← match node.arguments[1]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex leftWire)
      let left ← integerFactAt nodeIndex facts leftWire
      let right ← integerFactAt nodeIndex facts rightWire
      let interval ← integerBinaryInterval nodeIndex operation left right
      integerFactWithExpressions nodeIndex outputPort interval.lower interval.upper
        interval.lowerExpression interval.upperExpression
  | .constantReal _, none =>
      if node.arguments.isEmpty then pure .real
      else throw (.unsupportedOutputArity nodeIndex node.arguments.length)
  | .intToReal, none =>
      let inputWire ← match node.arguments with
        | [wire] => pure wire
        | _ => throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let _ ← integerFactAt nodeIndex facts inputWire
      pure .real
  | .realBinary _, none =>
      let (leftWire, rightWire) ← match node.arguments with
        | [left, right] => pure (left, right)
        | _ => throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      requireRealFact nodeIndex facts leftWire
      requireRealFact nodeIndex facts rightWire
      pure .real
  | .realSqrt, none =>
      let inputWire ← match node.arguments with
        | [wire] => pure wire
        | _ => throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      requireRealFact nodeIndex facts inputWire
      pure .real
  | .constantBool _, none =>
      if node.arguments.isEmpty then pure .boolean
      else throw (.unsupportedOutputArity nodeIndex node.arguments.length)
  | .intCompare _, none =>
      let (leftWire, rightWire) ← match node.arguments with
        | [left, right] => pure (left, right)
        | _ => throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let _ ← integerFactAt nodeIndex facts leftWire
      let _ ← integerFactAt nodeIndex facts rightWire
      pure .boolean
  | .bitExtract bit, none =>
      let inputWire ← match node.arguments with
        | [wire] => pure wire
        | _ => throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let _ ← integerFactAt nodeIndex facts inputWire
      let minimum ← evaluateIntMinimum environment loopDomains bit
      if minimum < 0 then throw (.invalidCount nodeIndex minimum)
      pure .boolean
  | .extractCoefficient position, none =>
      let matrixWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let matrix ← match ← lookupFact nodeIndex facts matrixWire with
        | .matrix matrix => pure matrix
        | .selectedMatrices .. => throw (.selectedFamilyOperationUnsupported nodeIndex)
        | _ => throw (.operandNotMatrix nodeIndex matrixWire)
      if node.arguments.length != 1 then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let minimum ← evaluateIntMinimum environment loopDomains position
      let maximum ← evaluateIntMaximum environment loopDomains position
      if minimum < 0 || maximum >= Int.ofNat matrix.matrixParams.ringDimension then
        throw (.invalidCount nodeIndex maximum)
      let exclusiveUpper ← match matrix.canonicalRange with
        | .below upper => pure (Int.ofNat upper)
        | .unknown => pure matrix.matrixParams.modulus
      if exclusiveUpper <= 0 then
        throw (.invalidMatrixParameters nodeIndex)
      integerFact nodeIndex outputPort 0 (exclusiveUpper - 1)
  | .thresholdDecodeBool ciphertextModulus plaintextModulus length, none |
      .thresholdDecodeInt ciphertextModulus plaintextModulus length, none =>
      let inputWire ← match node.arguments with
        | [wire] => pure wire
        | _ => throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let input ← match ← lookupFact nodeIndex facts inputWire with
        | .matrix matrix => pure matrix
        | .selectedMatrices .. => throw (.selectedFamilyOperationUnsupported nodeIndex)
        | _ => throw (.operandNotMatrix nodeIndex inputWire)
      let ciphertext ← evaluateIntInvariant environment loopDomains ciphertextModulus
      let plaintext ← evaluateIntInvariant environment loopDomains plaintextModulus
      let count ← evaluateIntInvariant environment loopDomains length
      if input.matrixParams.rows != 1 || input.matrixParams.columns != 1 ||
          ciphertext != input.matrixParams.modulus || plaintext <= 1 || count <= 0 ||
          count > Int.ofNat input.matrixParams.ringDimension || node.outputCount != count.toNat then
        throw (.invalidMatrixParameters nodeIndex)
      match node.kind with
      | .thresholdDecodeBool .. => pure .boolean
      | .thresholdDecodeInt .. => integerFact nodeIndex outputPort 0 (plaintext - 1)
      | _ => throw (.unsupportedNode nodeIndex)
  | .select, none =>
      let indexWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let index ← integerFactAt nodeIndex facts indexWire
      if node.arguments.length < 2 then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let branchCount := node.arguments.length - 1
      -- Magnitude analysis is conditional on successful executable evaluation. Reject only an
      -- interval that cannot select any branch; a partially overlapping interval denotes one of
      -- the valid branches whenever the runtime Select succeeds.
      if index.upper < 0 || index.lower >= Int.ofNat branchCount then
        throw (.invalidCount nodeIndex index.upper)
      let branches ← (node.arguments.drop 1).mapM (lookupFact nodeIndex facts)
      let packedBranches? := branches.mapM fun branch => match branch with
        | .familyPacked elements count (some _) => match elements[0]? with
            | some element => match element with
                | OperationalFact.matrix representative => some (count, representative)
                | _ => none
            | none => none
        | _ => none
      match packedBranches? with
      | some packedBranches =>
          let first ← match packedBranches.head? with
            | some first => pure first
            | none => throw (OperationalError.invalidCount nodeIndex 0)
          if packedBranches.any fun branch => branch.1 != first.1 then
            throw (.loopInputModeMismatch nodeIndex 0)
          let alternatives := packedBranches.map (·.2) |>.toArray
          let selected := selectedMatrixFamily index.origin alternatives
          pure (.familyPacked #[.selectedMatrices selected] first.1 none)
      | none =>
          joinDynamicFacts nodeIndex { node := nodeIndex, port := outputPort } branches
            (some index.origin)
  | .zeroMatrix _, some matrixType =>
      polynomialMatrixFact nodeIndex outputPort matrixType environment [] (.below 1)
  | .identityMatrix _, some matrixType =>
      classifiedMatrixFact nodeIndex outputPort matrixType environment 1 false (.below 2)
        { isConstantPolynomial := true }
  | .constantMatrix _ coefficients, some matrixType =>
      let values ← coefficients.mapM (evaluateIntInvariant environment loopDomains)
      let ringDimension ← match matrixType.ringDimension.evaluate environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      if ringDimension <= 0 then throw (.invalidMatrixParameters nodeIndex)
      let modulus ← match matrixType.modulus.evaluate environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let canonicalMaximum := values.foldl (fun maximum value =>
        max maximum ((if modulus > 0 then value % modulus else value).toNat)) 0
      classifiedMatrixFact nodeIndex outputPort matrixType environment
        (values.foldl (fun maximum value => max maximum (absolute value)) 0) false
        (.below (canonicalMaximum + 1)) {
          isConstantPolynomial := values.zipIdx.all fun (value, index) =>
            index % ringDimension.toNat = 0 || value = 0
        }
  | .uniformResidueSample _, some matrixType =>
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
  | .uniformIntervalSample _ minimum maximum, some matrixType =>
      let lower ← evaluateIntMinimum environment loopDomains minimum
      let upper ← evaluateIntMaximum environment loopDomains maximum
      let bound := OperationalBoundExpr.maximum
        (.contextual .maximumAbsolute environment loopDomains minimum)
        (.contextual .maximumAbsolute environment loopDomains maximum)
      classifiedMatrixFactExpr nodeIndex outputPort matrixType environment bound false
        (if lower >= 0 then .below (upper.toNat + 1) else .unknown)
  | .gaussianSample _ maximum, some matrixType =>
      cappedMatrixFactExpr nodeIndex outputPort matrixType environment
        (.contextual .maximum environment loopDomains maximum)
  | .preimageSample _ maximum, some matrixType =>
      let bound := OperationalBoundExpr.contextual .maximum environment loopDomains maximum
      let publicWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let trapdoorWire ← match node.arguments[1]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex publicWire)
      let targetWire ← match node.arguments[2]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex publicWire)
      let publicFact ← matrixFactAt nodeIndex facts publicWire
      let trapdoor ← match ← lookupFact nodeIndex facts trapdoorWire with
        | .trapdoor fact => pure fact
        | _ => throw (.missingPublicIdentity nodeIndex trapdoorWire)
      let targetFact ← lookupFact nodeIndex facts targetWire
      let publicIdentity ← match publicFact.identity with
        | some identity => pure identity
        | none => throw (.missingPublicIdentity nodeIndex publicWire)
      if publicIdentity != trapdoor.publicIdentity then
        throw (.publicIdentityMismatch nodeIndex)
      let result ← cappedMatrixFactExpr nodeIndex outputPort matrixType environment bound
      match result with
      | .matrix result =>
          let attachRelation
              (branch : Option Nat)
              (target : OperationalMatrixFact) : OperationalMatrixFact :=
            let result := match branch with
              | some index => { result with
                  origin := .loopInstance nodeIndex index result.origin }
              | none => result
            let relation : PreimageRelation := {
              producer := result.origin
              publicIdentity
              targetOrigin := target.origin
              targetSummary := matrixTargetSummary target
            }
            ({ result with relations := [.preimage relation] }).refreshPrimitivePolynomial
          match targetFact with
          | .matrix target => pure (.matrix (attachRelation none target))
          | .selectedMatrices family =>
              match family.summary.uniformSchema, family.branches[0]? with
              | some _, some target =>
                  let representative := attachRelation (some 0) target
                  let summary : SelectedMatrixSummary := {
                    uniformSchema := some (operationalUniformSchema representative)
                    relationFree := false
                    sharedLastPublicIdentity := none
                    sharedFirstRelationPublicIdentity := some publicIdentity
                  }
                  pure (.selectedMatrices (selectedMatrixEnvelope family.selection family.count
                    representative summary family.representsLoopLanes))
              | _, _ =>
                  pure (.selectedMatrices {
                    selectedMatrixFamily family.selection
                      (family.branches.mapIdx fun index target => attachRelation (some index) target)
                    with representsLoopLanes := family.representsLoopLanes
                  })
          | _ => throw (.operandNotMatrix nodeIndex targetWire)
      | _ => throw (.malformedRelation nodeIndex)
  | .hashSample _ variant tagPrefix tagExpressions tagDecimalExpressions tagU64LeExpressions
      base digitCount, some matrixType =>
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let keyWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let keyOrigin ← valueOriginAt scopeKey nodeIndex facts keyWire
      let trailingIntegerOrigins ← (node.arguments.drop 1).mapM
        (valueOriginAt scopeKey nodeIndex facts)
      let hashIdentity (targetType : MatrixTypeExpr) : DeterministicHashIdentity := {
        keyOrigin
        matrixType := targetType
        parameterEnvironment := environment
        parameterDomains := loopDomains
        tagPrefix
        tagExpressions
        tagDecimalExpressions
        tagU64LeExpressions
        trailingIntegerOrigins
      }
      match variant with
      | .plain =>
          match ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true with
          | .matrix result => pure (.matrix {
              result with origin := .deterministicHash (hashIdentity matrixType)
            })
          | _ => throw (.malformedRelation nodeIndex)
      | .decomposed | .smallDecomposed =>
          let base ← match base with
            | some expression => evaluateIntInvariant environment loopDomains expression
            | none => throw (.gadgetLayoutMismatch nodeIndex)
          let digitCount ← match digitCount with
            | some expression => evaluateIntInvariant environment loopDomains expression
            | none => throw (.gadgetLayoutMismatch nodeIndex)
          if base <= 1 || digitCount <= 0 then throw (.gadgetLayoutMismatch nodeIndex)
          let outputParams ← match matrixType.evaluate environment (.constant 0) with
            | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
          if outputParams.rows % digitCount.toNat != 0 then
            throw (.gadgetLayoutMismatch nodeIndex)
          let descriptor ← resolveGadgetLayout nodeIndex layouts outputParams
          let small := variant == Mxx.HashVariant.smallDecomposed
          let expectedCount := if small then descriptor.smallDigitCount else
            descriptor.regularDigitCount
          if descriptor.base != base || expectedCount != digitCount.toNat then
            throw (.gadgetLayoutMismatch nodeIndex)
          let targetRows := outputParams.rows / digitCount.toNat
          let targetType : MatrixTypeExpr := {
            modulus := .constant outputParams.modulus
            ringDimension := .constant (Int.ofNat outputParams.ringDimension)
            rows := .constant (Int.ofNat targetRows)
            columns := .constant (Int.ofNat outputParams.columns)
          }
          let targetParams : Mxx.SamplerParams := {
            maxCoefficientBound := cap.natAbs
            modulus := outputParams.modulus
            ringDimension := outputParams.ringDimension
            rows := targetRows
            columns := outputParams.columns
          }
          let targetOrigin := MatrixOriginIdentity.deterministicHash (hashIdentity targetType)
          let targetSummary : RelationTargetSummary := {
            origin := targetOrigin
            matrixType := targetType
            matrixParams := targetParams
            totalHardBound := .closedInt (.constant cap)
            canonicalRange := .unknown
            polynomial := relationSnapshotPolynomial (primitiveOperationalPolynomial targetOrigin
              targetType (.closedInt (.constant cap)) .large none [] {})
          }
          let publicIdentity := PublicMatrixIdentity.gadget descriptor.paramsId
            outputParams targetRows base small digitCount.toNat
          let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment
            (Int.ofNat (Mxx.gadgetDecompositionBound base small)) false
            (if small then .below base.natAbs else .unknown)
          match result with
          | .matrix result =>
              let relation : DecompositionRelation := {
                producer := result.origin
                publicIdentity
                inputOrigin := targetOrigin
                inputSummary := targetSummary
                base
                small
                digitCount := digitCount.toNat
                status := if small then .smallRangeMissing descriptor.smallestCrtModulus else
                  .available
              }
              pure (.matrix ({ result with relations := [.decomposition relation] }).refreshPrimitivePolynomial)
          | _ => throw (.malformedRelation nodeIndex)
  | .gadgetDecompose declaredType base small digitCount, some matrixType =>
      let bound ← evaluateIntInvariant environment loopDomains base
      let count ← evaluateIntInvariant environment loopDomains digitCount
      if bound <= 1 || count <= 0 then throw (.gadgetLayoutMismatch nodeIndex)
      let params ← match declaredType.evaluate environment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let descriptor ← resolveGadgetLayout nodeIndex layouts params
      let expectedCount := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
      if count.toNat != expectedCount || bound != descriptor.base then
        throw (.gadgetLayoutMismatch nodeIndex)
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let inputFact ← lookupFact nodeIndex facts inputWire
      let (selection, inputs) ← match inputFact with
        | .matrix input => pure (none, #[input])
        | .selectedMatrices family => pure (some family.selection, family.branches)
        | _ => throw (.operandNotMatrix nodeIndex inputWire)
      let input ← match inputs[0]? with
        | some input => pure input
        | none => throw (.selectedFamilyOperationUnsupported nodeIndex)
      if inputs.any fun candidate => candidate.matrixParams != input.matrixParams then
        throw (.selectedFamilyOperationUnsupported nodeIndex)
      let publicIdentity := PublicMatrixIdentity.gadget descriptor.paramsId params
        input.matrixParams.rows bound small count.toNat
      let result ← cappedMatrixFact nodeIndex outputPort matrixType environment
        (Int.ofNat (Mxx.gadgetDecompositionBound bound small))
      match result with
      | .matrix result =>
          let attachRelation (branch : Nat) (input : OperationalMatrixFact) :=
            let result := match selection with
              | some _ => { result with origin := .loopInstance nodeIndex branch result.origin }
              | none => result
            let status := if !small then ReconstructionStatus.available else
              match input.canonicalRange with
              | .below upper => if upper <= descriptor.smallestCrtModulus then
                  .available else .smallRangeMissing descriptor.smallestCrtModulus
              | .unknown => .smallRangeMissing descriptor.smallestCrtModulus
            let relation : DecompositionRelation := {
              producer := result.origin
              publicIdentity
              inputOrigin := input.origin
              inputSummary := matrixTargetSummary input
              base := bound
              small
              digitCount := count.toNat
              status
            }
            ({ result with
              canonicalRange := if small then .below bound.natAbs else .unknown
              relations := [.decomposition relation]
            }).refreshPrimitivePolynomial
          let outputs := inputs.mapIdx attachRelation
          match selection with
          | none => match outputs[0]? with
              | some output => pure (.matrix output)
              | none => throw (.selectedFamilyOperationUnsupported nodeIndex)
          | some selection =>
              pure (.selectedMatrices (selectedMatrixFamily selection outputs))
      | _ => throw (.malformedRelation nodeIndex)
  | .matrixAdd, some matrixType | .matrixSubtract, some matrixType =>
      if node.arguments.length != 2 then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let leftWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let rightWire ← match node.arguments[1]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex leftWire)
      let leftFact ← lookupFact nodeIndex facts leftWire
      let rightFact ← lookupFact nodeIndex facts rightWire
      let combinePair
          (left right : OperationalMatrixFact) : Except OperationalError OperationalFact := do
        let left ← retypeMatrixFact nodeIndex matrixType left environment
        let right ← retypeMatrixFact nodeIndex matrixType right environment
        let polynomial := match node.kind with
          | .matrixAdd => addOperationalPolynomials left.polynomial right.polynomial
          | .matrixSubtract => subtractOperationalPolynomials left.polynomial right.polynomial
          | _ => []
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
      let finishSelected
          (selection : OperationalValueOrigin)
          (count : Nat)
          (representsLoopLanes : Bool)
          (outputs : Array OperationalFact) : Except OperationalError OperationalFact := do
        let branches ← outputs.mapM fun output => match output with
          | .matrix branch => pure branch
          | _ => throw (.operandNotMatrix nodeIndex leftWire)
        compressUniformSelectedMatrices nodeIndex { node := nodeIndex, port := outputPort }
          selection branches count representsLoopLanes
      match leftFact, rightFact with
      | .matrix left, .matrix right => combinePair left right
      | .selectedMatrices family, .matrix right => do
          let outputs ← family.branches.mapM fun left => combinePair left right
          finishSelected family.selection family.count family.representsLoopLanes outputs
      | .matrix left, .selectedMatrices family => do
          let outputs ← family.branches.mapM fun right => combinePair left right
          finishSelected family.selection family.count family.representsLoopLanes outputs
      | .selectedMatrices leftFamily, .selectedMatrices rightFamily => do
          if leftFamily.selection != rightFamily.selection ||
              leftFamily.branches.size != rightFamily.branches.size then do
            if leftFamily.selection == rightFamily.selection ||
                leftFamily.representsLoopLanes != rightFamily.representsLoopLanes ||
                (leftFamily.isEnvelope && leftFamily.summary.uniformSchema.isNone) ||
                (rightFamily.isEnvelope && rightFamily.summary.uniformSchema.isNone) then
              throw (.selectedFamilyOperationUnsupported nodeIndex)
            match leftFamily.summary.uniformSchema, leftFamily.branches[0]?,
                rightFamily.branches[0]? with
            | some _, some leftRepresentative, some rightRepresentative =>
                match absorbedSelectionBinder? rightRepresentative leftFamily.selection with
                | some binder =>
                    if rightFamily.branches.all fun branch =>
                        absorbedSelectionBinder? branch leftFamily.selection == some binder then
                      let selectedLeft ← match ← selectDynamicUniformFact binder
                          leftFamily.selection { node := nodeIndex, port := outputPort }
                          (.matrix leftRepresentative) with
                        | .matrix selected => pure selected
                        | _ => throw (.operandNotMatrix nodeIndex leftWire)
                      let outputs ← rightFamily.branches.mapM fun right =>
                        combinePair selectedLeft right
                      let branches ← outputs.mapM fun output => match output with
                        | .matrix branch => pure branch
                        | _ => throw (.operandNotMatrix nodeIndex leftWire)
                      let summary := selectedMatrixSummary branches
                      if rightFamily.isEnvelope && summary.uniformSchema.isNone then
                        throw (.selectedFamilyOperationUnsupported nodeIndex)
                      let representative ← match branches[0]? with
                        | some branch => pure branch
                        | none => throw (.selectedFamilyOperationUnsupported nodeIndex)
                      if rightFamily.isEnvelope then
                        return .selectedMatrices (selectedMatrixEnvelope rightFamily.selection
                          rightFamily.count representative summary
                          rightFamily.representsLoopLanes)
                      else
                        return .selectedMatrices {
                          selectedMatrixFamily rightFamily.selection branches with
                          representsLoopLanes := rightFamily.representsLoopLanes
                        }
                | none => pure ()
            | _, _, _ => pure ()
            if leftFamily.isEnvelope || rightFamily.isEnvelope then
              throw (.selectedFamilyOperationUnsupported nodeIndex)
            let left ← match ← joinDynamicFacts nodeIndex
                { node := nodeIndex, port := outputPort }
                (leftFamily.branches.map OperationalFact.matrix).toList
                (some leftFamily.selection) with
              | .matrix matrix => pure matrix
              | _ => throw (.operandNotMatrix nodeIndex leftWire)
            let right ← match ← joinDynamicFacts nodeIndex
                { node := nodeIndex, port := outputPort }
                (rightFamily.branches.map OperationalFact.matrix).toList
                (some rightFamily.selection) with
              | .matrix matrix => pure matrix
              | _ => throw (.operandNotMatrix nodeIndex rightWire)
            combinePair left right
          else
            let mut outputs : Array OperationalFact := #[]
            for branch in [:leftFamily.branches.size] do
              match leftFamily.branches[branch]?, rightFamily.branches[branch]? with
              | some left, some right => outputs := outputs.push (← combinePair left right)
              | _, _ => throw (.selectedFamilyOperationUnsupported nodeIndex)
            finishSelected leftFamily.selection leftFamily.count
              leftFamily.representsLoopLanes outputs
      | _, _ => throw (.operandNotMatrix nodeIndex leftWire)
  | .concat axis, some matrixType =>
      let inputs ← node.arguments.mapM fun wire =>
        return (wire, ← lookupFact nodeIndex facts wire)
      let aligned ← alignSelectedMatrixInputs nodeIndex inputs
      let outputs ← aligned.rows.mapM fun row => do
        let polynomial ← concatOperationalPolynomials axis matrixType (row.map (·.polynomial))
          |>.mapError (flatErrorAt nodeIndex)
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
          (joinCanonicalRanges (row.map (·.canonicalRange)))
      finishAlignedMatrixOutputs nodeIndex (node.arguments.headD { node := 0, port := 0 })
        aligned.selection outputs
  | .select, some matrixType =>
      let indexWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let index ← integerFactAt nodeIndex facts indexWire
      if node.arguments.length < 2 then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let branchCount := node.arguments.length - 1
      -- Match FamilyGetDynamic's conditional-on-success interpretation above.
      if index.upper < 0 || index.lower >= Int.ofNat branchCount then
        throw (.invalidCount nodeIndex index.upper)
      let branches ← (node.arguments.drop 1).mapM fun wire => do
        match ← lookupFact nodeIndex facts wire with
        | .matrix branch => retypeMatrixFact nodeIndex matrixType branch environment
        | .selectedMatrices .. => throw (.selectedFamilyOperationUnsupported nodeIndex)
        | _ => throw (.operandNotMatrix nodeIndex wire)
      let polynomial ← selectOperationalPolynomials scopeKey nodeIndex index.origin matrixType
        (branches.map (·.polynomial)) |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
        (joinCanonicalRanges (branches.map (·.canonicalRange)))
  | .transpose, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      liftSelectedUnaryMatrix nodeIndex inputWire (← lookupFact nodeIndex facts inputWire) fun input => do
        let polynomial ← transposeOperationalPolynomial input.polynomial
          |>.mapError (flatErrorAt nodeIndex)
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial input.canonicalRange
  | .slice rows columns, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      liftSelectedUnaryMatrix nodeIndex inputWire (← lookupFact nodeIndex facts inputWire) fun input => do
        let polynomial ← sliceOperationalPolynomial rows columns matrixType input.polynomial
          |>.mapError (flatErrorAt nodeIndex)
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial input.canonicalRange
  | .reshape rows columns, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      liftSelectedUnaryMatrix nodeIndex inputWire (← lookupFact nodeIndex facts inputWire) fun input => do
        let outputParams ← match matrixType.evaluate environment
            (.constant input.matrixParams.maxCoefficientBound) with
          | some params => pure params
          | none => throw (.invalidMatrixParameters nodeIndex)
        let polynomial ←
          if sameConcreteMatrixShape input.matrixParams outputParams then
            equivalentRetypeOperationalPolynomial matrixType input.polynomial
              |>.mapError (flatErrorAt nodeIndex)
          else
            boundedStructuralTransformPolynomial (.reshape rows columns) matrixType input.polynomial
              |>.mapError (flatErrorAt nodeIndex)
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial input.canonicalRange
  | .constantCoefficient index, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let minimum ← evaluateIntMinimum environment loopDomains index
      let maximum ← evaluateIntMaximum environment loopDomains index
      liftSelectedUnaryMatrix nodeIndex inputWire (← lookupFact nodeIndex facts inputWire) fun input => do
        if node.arguments.length != 1 || input.matrixParams.rows != 1 ||
            input.matrixParams.columns != 1 then
          throw (.invalidMatrixParameters nodeIndex)
        if minimum < 0 || maximum >= Int.ofNat input.matrixParams.ringDimension then
          throw (.invalidCount nodeIndex maximum)
        let polynomial ← boundedStructuralTransformPolynomial (.constantCoefficient index)
          matrixType input.polynomial |>.mapError (flatErrorAt nodeIndex)
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial input.canonicalRange
  | .matrixNegate, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let negate (input : OperationalMatrixFact) := do
        let input ← retypeMatrixFact nodeIndex matrixType input environment
        polynomialMatrixFact nodeIndex outputPort matrixType environment
          (scaleOperationalPolynomial (-1) input.polynomial) input.canonicalRange
      match ← lookupFact nodeIndex facts inputWire with
      | .matrix input => negate input
      | .selectedMatrices family => do
          let outputs ← family.branches.mapM negate
          let branches ← outputs.mapM fun output => match output with
            | .matrix branch => pure branch
            | _ => throw (.operandNotMatrix nodeIndex inputWire)
          compressUniformSelectedMatrices nodeIndex { node := nodeIndex, port := outputPort }
            family.selection branches family.count family.representsLoopLanes
      | _ => throw (.operandNotMatrix nodeIndex inputWire)
  | .matrixScale scalar, some matrixType =>
      let scalarValues ← evaluateIntOverLoops environment loopDomains scalar
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let scale (input : OperationalMatrixFact) := do
        let input ← retypeMatrixFact nodeIndex matrixType input environment
        match scalarValues with
        | [] => throw (.invalidMatrixParameters nodeIndex)
        | first :: tail =>
            if first == 1 && tail.all (· == 1) then
              pure (.matrix { input with subject := { node := nodeIndex, port := outputPort } })
            else
              let polynomial ←
                if tail.all (· == first) then
                  pure (scaleOperationalPolynomial first input.polynomial)
                else
                  multiplyOperationalPolynomials
                    (parameterScalarPolynomial environment loopDomains scalar matrixType)
                    input.polynomial |>.mapError (flatErrorAt nodeIndex)
              polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
                input.canonicalRange
      match ← lookupFact nodeIndex facts inputWire with
      | .matrix input => scale input
      | .selectedMatrices family => do
          let outputs ← family.branches.mapM scale
          let branches ← outputs.mapM fun output => match output with
            | .matrix branch => pure branch
            | _ => throw (.operandNotMatrix nodeIndex inputWire)
          compressUniformSelectedMatrices nodeIndex { node := nodeIndex, port := outputPort }
            family.selection branches family.count family.representsLoopLanes
      | _ => throw (.operandNotMatrix nodeIndex inputWire)
  | .matrixMultiply, some matrixType =>
      let leftWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let rightWire ← match node.arguments[1]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex leftWire)
      let leftFact ← lookupFact nodeIndex facts leftWire
      let rightFact ← lookupFact nodeIndex facts rightWire
      let multiplyPair
          (left right : OperationalMatrixFact) : Except OperationalError OperationalFact := do
        let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial
          |>.mapError (flatErrorAt nodeIndex)
        let rewritten ← rewriteOperationalRelations nodeIndex raw
        let polynomial ← match rule with
          | .matrixMultiplyRelation declaredRight => do
              if declaredRight != rightWire then throw (.missingRelation nodeIndex declaredRight)
              if rewritten == raw then throw (.missingRelation nodeIndex rightWire)
              pure rewritten
          | _ => pure rewritten
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
      let finishSelected
          (selection : OperationalValueOrigin)
          (count : Nat)
          (representsLoopLanes : Bool)
          (outputs : Array OperationalFact) : Except OperationalError OperationalFact := do
        let branches ← outputs.mapM fun output => match output with
          | .matrix branch => pure branch
          | _ => throw (.operandNotMatrix nodeIndex leftWire)
        compressUniformSelectedMatrices nodeIndex { node := nodeIndex, port := outputPort }
          selection branches count representsLoopLanes
      match leftFact, rightFact with
      | .matrix left, .matrix right => multiplyPair left right
      | .selectedMatrices family, .matrix right => do
          let outputs ← family.branches.mapM fun left => multiplyPair left right
          finishSelected family.selection family.count family.representsLoopLanes outputs
      | .matrix left, .selectedMatrices family => do
          if uniformBoundaryRelationsWithSharedLeft left family then
            match family.branches[0]? with
            | some right => do
                let output ← multiplyPair left right
                finishSelected family.selection family.count family.representsLoopLanes #[output]
            | none => throw (.selectedFamilyOperationUnsupported nodeIndex)
          else
            if family.isEnvelope && (family.summary.uniformSchema.isSome &&
                (family.summary.relationFree ||
                  family.summary.sharedFirstRelationPublicIdentity.isSome)) then
              match family.branches[0]? with
              | some right => do
                  let output ← multiplyPair left right
                  finishSelected family.selection family.count family.representsLoopLanes #[output]
              | none => throw (.selectedFamilyOperationUnsupported nodeIndex)
            else if family.isEnvelope then do
              throw (.selectedFamilyOperationUnsupported nodeIndex)
            else
              let outputs ← family.branches.mapM fun right => multiplyPair left right
              finishSelected family.selection family.count family.representsLoopLanes outputs
      | .selectedMatrices leftFamily, .selectedMatrices rightFamily => do
          if leftFamily.selection != rightFamily.selection ||
              leftFamily.count != rightFamily.count ||
              leftFamily.representsLoopLanes != rightFamily.representsLoopLanes then do
            if leftFamily.selection == rightFamily.selection ||
                leftFamily.representsLoopLanes != rightFamily.representsLoopLanes ||
                (leftFamily.isEnvelope && leftFamily.summary.uniformSchema.isNone) ||
                (rightFamily.isEnvelope && rightFamily.summary.uniformSchema.isNone) then
              throw (.selectedFamilyOperationUnsupported nodeIndex)
            let binder : FamilyTemplateBinder := {
              owner := dynamicSelectionScope rightFamily.selection
              producerNode := nodeIndex
              binderSlot := 0
            }
            let mut selectedBranches : Array OperationalMatrixFact := #[]
            for left in leftFamily.branches do
              let mut outputs : Array OperationalFact := #[]
              for right in rightFamily.branches do
                outputs := outputs.push (← multiplyPair left right)
              let branches ← outputs.mapM fun output => match output with
                | .matrix branch => pure branch
                | _ => throw (.operandNotMatrix nodeIndex leftWire)
              let summary := selectedMatrixSummary branches
              if summary.uniformSchema.isNone then
                throw (.selectedFamilyOperationUnsupported nodeIndex)
              let representative ← match branches[0]? with
                | some branch => pure branch
                | none => throw (.selectedFamilyOperationUnsupported nodeIndex)
              match ← selectDynamicUniformFact binder rightFamily.selection
                  { node := nodeIndex, port := outputPort } (.matrix representative) with
              | .matrix selected => selectedBranches := selectedBranches.push selected
              | _ => throw (.operandNotMatrix nodeIndex leftWire)
            let summary := selectedMatrixSummary selectedBranches
            if leftFamily.isEnvelope && summary.uniformSchema.isNone then
              throw (.selectedFamilyOperationUnsupported nodeIndex)
            let representative ← match selectedBranches[0]? with
              | some branch => pure branch
              | none => throw (.selectedFamilyOperationUnsupported nodeIndex)
            if leftFamily.isEnvelope then
              pure (.selectedMatrices (selectedMatrixEnvelope leftFamily.selection
                leftFamily.count representative summary leftFamily.representsLoopLanes))
            else
              pure (.selectedMatrices {
                selectedMatrixFamily leftFamily.selection selectedBranches with
                representsLoopLanes := leftFamily.representsLoopLanes
              })
          else if uniformBoundaryRelationPairs leftFamily rightFamily then
            match leftFamily.branches[0]?, rightFamily.branches[0]? with
            | some left, some right => do
                let output ← multiplyPair left right
                finishSelected leftFamily.selection leftFamily.count
                  leftFamily.representsLoopLanes #[output]
            | _, _ => throw (.selectedFamilyOperationUnsupported nodeIndex)
          else
            if leftFamily.isEnvelope || rightFamily.isEnvelope then do
              throw (.selectedFamilyOperationUnsupported nodeIndex)
            let mut outputs : Array OperationalFact := #[]
            for branch in [:leftFamily.branches.size] do
              match leftFamily.branches[branch]?, rightFamily.branches[branch]? with
              | some left, some right => outputs := outputs.push (← multiplyPair left right)
              | _, _ => throw (.selectedFamilyOperationUnsupported nodeIndex)
            finishSelected leftFamily.selection leftFamily.count
              leftFamily.representsLoopLanes outputs
      | _, _ => throw (.operandNotMatrix nodeIndex leftWire)
  | .tensor, some matrixType =>
      let leftWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let rightWire ← match node.arguments[1]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex leftWire)
      let aligned ← alignSelectedMatrixInputs nodeIndex
        [(leftWire, ← lookupFact nodeIndex facts leftWire),
          (rightWire, ← lookupFact nodeIndex facts rightWire)]
      let outputs ← aligned.rows.mapM fun row => match row with
        | [left, right] => do
            let polynomial ← tensorOperationalPolynomials matrixType
              left.polynomial right.polynomial |>.mapError (flatErrorAt nodeIndex)
            polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
        | _ => throw (.selectedFamilyOperationUnsupported nodeIndex)
      finishAlignedMatrixOutputs nodeIndex leftWire aligned.selection outputs
  | .crtRecompose plaintextModuli reconstructionCoefficients, some matrixType =>
      if node.arguments.isEmpty || node.arguments.length != plaintextModuli.length ||
          node.arguments.length != reconstructionCoefficients.length then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let moduli ← plaintextModuli.mapM (evaluateIntInvariant environment loopDomains)
      let coefficients ← reconstructionCoefficients.mapM
        (evaluateIntInvariant environment loopDomains)
      let inputFacts ← node.arguments.mapM fun wire =>
        return (wire, ← lookupFact nodeIndex facts wire)
      let aligned ← alignSelectedMatrixInputs nodeIndex inputFacts
      let modulus ← evaluateIntInvariant environment loopDomains matrixType.modulus
      if modulus <= 0 || moduli.any (fun value => value <= 1 || value >= modulus) ||
          coefficients.any (fun value => value < 0 || value >= modulus) then
        throw (.invalidMatrixParameters nodeIndex)
      let outputs ← aligned.rows.mapM fun inputs => do
        let inputs ← inputs.mapM fun input =>
          retypeMatrixFact nodeIndex matrixType input environment
        if inputs.any (·.matrixParams.rows != 1) then
          throw (.invalidMatrixParameters nodeIndex)
        let polynomial := (coefficients.zip inputs).foldl
          (fun result pair ↦ addOperationalPolynomials result
            (scaleOperationalPolynomial pair.1 pair.2.polynomial)) []
        polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
      finishAlignedMatrixOutputs nodeIndex
        (node.arguments.headD { node := 0, port := 0 }) aligned.selection outputs
  | .packPolynomialCoefficients _ coefficientBits, some matrixType =>
      let bits ← evaluateIntMaximum environment loopDomains coefficientBits
      if bits <= 0 then throw (.invalidBound nodeIndex bits)
      if node.arguments.length != 1 then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let cap ← match matrixCap matrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      let params ← match matrixType.evaluate environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      if params.rows != 1 || params.columns != 1 || (2 : Int) ^ bits.toNat < params.modulus then
        throw (.invalidMatrixParameters nodeIndex)
      let inputWire := node.arguments.headD { node := 0, port := 0 }
      let input ← lookupFact nodeIndex facts inputWire
      let expectedCount := Int.ofNat params.ringDimension * bits
      if booleanFamilyCount input != some expectedCount then
        throw (.loopInputModeMismatch nodeIndex 0)
      classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
        (if params.modulus > 0 then .below params.modulus.toNat else .unknown)
  | .trapdoorSample _ maximum, some matrixType =>
      let maximum ← evaluateIntMaximum environment loopDomains maximum
      if maximum < 0 then throw (.invalidBound nodeIndex maximum)
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
      match result with
      | .matrix result => pure (.matrix ({
          result with identity := some (.sampledTrapdoor temporaryScope
            { node := nodeIndex, port := 0 })
        }).refreshPrimitivePolynomial)
      | _ => throw (.malformedRelation nodeIndex)
  | .trapdoorSample _ maximum, none =>
      let boundExpr := OperationalBoundExpr.contextual .maximum environment loopDomains maximum
      let bound ← boundExpr.evaluate environment #[]
      match outputType with
      | .trapdoor matrixType _ _ _ _ =>
          let cap ← match matrixCap matrixType environment with
            | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
          let maximum := min cap bound
          let params ← match matrixType.evaluate environment (.constant maximum) with
            | some params => pure params
            | none => throw (.invalidMatrixParameters nodeIndex)
          pure (.trapdoor {
            subject := { node := nodeIndex, port := outputPort }
            matrixType
            matrixParams := params
            maximum := .minimum (.closedInt (.constant cap)) boundExpr
            publicIdentity := .sampledTrapdoor temporaryScope { node := nodeIndex, port := 0 }
          })
      | _ => defaultFact nodeIndex outputPort outputType environment
  | .trapdoorPublic, some matrixType =>
      let trapdoorWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let trapdoor ← match ← lookupFact nodeIndex facts trapdoorWire with
        | .trapdoor fact => pure fact
        | _ => throw (.missingPublicIdentity nodeIndex trapdoorWire)
      let cap ← match matrixCap matrixType environment with
        | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
      let bound := publicIdentityMaximum cap trapdoor.publicIdentity
      let large := publicIdentityIsLarge trapdoor.publicIdentity
      let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment bound large
      match result with
      | .matrix result => pure (.matrix ({
          result with identity := some trapdoor.publicIdentity
        }).refreshPrimitivePolynomial)
      | _ => throw (.malformedRelation nodeIndex)
  | .gadgetTrapdoor _ base, some matrixType =>
      let bound ← evaluateIntInvariant environment loopDomains base
      let params ← match matrixType.evaluate environment (.constant 0) with
        | some params => pure params | none => throw (.invalidMatrixParameters nodeIndex)
      let descriptor ← resolveGadgetLayout nodeIndex layouts params
      let count := descriptor.regularDigitCount
      if bound != descriptor.base then throw (.gadgetLayoutMismatch nodeIndex)
      let identity := PublicMatrixIdentity.gadget descriptor.paramsId params
        params.rows bound false count
      let cap ← match matrixCap matrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
      match result with
      | .matrix result => pure (.matrix ({
          result with identity := some identity
        }).refreshPrimitivePolynomial)
      | _ => throw (.malformedRelation nodeIndex)
  | .gadgetTrapdoor _ base, none =>
      let bound ← evaluateIntInvariant environment loopDomains base
      match outputType with
      | .trapdoor matrixType _ _ _ _ =>
          let params ← match matrixType.evaluate environment (.constant 0) with
            | some params => pure params | none => throw (.invalidMatrixParameters nodeIndex)
          let descriptor ← resolveGadgetLayout nodeIndex layouts params
          if bound != descriptor.base then throw (.gadgetLayoutMismatch nodeIndex)
          let identity := PublicMatrixIdentity.gadget descriptor.paramsId params
            params.rows bound false descriptor.regularDigitCount
          pure (.trapdoor {
            subject := { node := nodeIndex, port := outputPort }
            matrixType
            matrixParams := params
            maximum := .closedInt (.constant (absolute bound))
            publicIdentity := identity
          })
      | _ => defaultFact nodeIndex outputPort outputType environment
  | .unitRowMatrix _ _, some matrixType | .unitColumnMatrix _ _, some matrixType =>
      classifiedMatrixFact nodeIndex outputPort matrixType environment 1 false (.below 2)
        { isConstantPolynomial := true }
  | .rotationMatrix _ _, some matrixType =>
      classifiedMatrixFact nodeIndex outputPort matrixType environment 1 false
  | .gadgetMatrix _ base, some matrixType | .smallGadgetMatrix _ base, some matrixType =>
      let bound ← evaluateIntMaximumAbsolute environment loopDomains base
      let params ← match matrixType.evaluate environment (.constant 0) with
        | some params => pure params | none => throw (.invalidMatrixParameters nodeIndex)
      let descriptor ← resolveGadgetLayout nodeIndex layouts params
      let small := match node.kind with | .smallGadgetMatrix _ _ => true | _ => false
      let count := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
      if bound != descriptor.base then throw (.gadgetLayoutMismatch nodeIndex)
      let cap ← match matrixCap matrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      let result ← classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
      match result with
      | .matrix result => pure (.matrix ({
          result with identity := some (.gadget descriptor.paramsId params params.rows bound small count)
        }).refreshPrimitivePolynomial)
      | _ => throw (.malformedRelation nodeIndex)
  | .powerOfBaseMatrix _ base _, some matrixType =>
      let _ ← evaluateIntMaximumAbsolute environment loopDomains base
      let cap ← match matrixCap matrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters nodeIndex)
      classifiedMatrixFact nodeIndex outputPort matrixType environment cap true
  | _, some _ => throw (.unsupportedNode nodeIndex)
  | _, none => throw (.unsupportedNode nodeIndex)

def lookupCheckedDefinition
    (name : String)
    (definitions : List (String × Scope))
    (derivations : List (String × ScopeDerivation)) :
    Except OperationalError (Scope × ScopeDerivation) :=
  match definitions, derivations with
  | [], _ => .error (.missingDefinition name)
  | _, [] => .error (.missingDefinition name)
  | (definitionName, scope) :: definitionTail,
      (derivationName, derivation) :: derivationTail =>
      if definitionName != derivationName then .error (.missingDefinition name)
      else if definitionName = name then .ok (scope, derivation)
      else lookupCheckedDefinition name definitionTail derivationTail

private def validateScopeInputs (scope : Scope) : Except OperationalError Unit := do
  let nodeNames := scope.nodes.filterMap fun node => match node.kind with
    | .input name => some name
    | _ => none
  for name in scope.inputNames do
    if scope.inputNames.count name != 1 then throw (.duplicateInputName name)
    if nodeNames.count name = 0 then throw (.missingInputNode name)
    if nodeNames.count name != 1 then throw (.duplicateInputName name)
  for name in nodeNames do
    if !scope.inputNames.contains name then throw (.unexpectedInputNode name)

private def findDefinitionIndex
    (name : String) : List (String × Scope) → Nat → Option Nat
  | [], _ => none
  | (candidate, _) :: tail, index =>
      if candidate == name then some index else findDefinitionIndex name tail (index + 1)

private def prepareOperationalScope
    (definitions : List (String × Scope))
    (scope : Scope)
    (derivation : ScopeDerivation) : Except OperationalError PreparedOperationalScope := do
  validateScopeInputs scope
  let inputIndices := scope.nodes.map fun node => match node.kind with
    | .input name => scope.inputNames.idxOf? name
    | _ => none
  let definitionIndices := scope.nodes.map fun node => match node.kind with
    | .subgraphCall name _ => findDefinitionIndex name definitions 0
    | .parallelLoop name _ _ _ _ => findDefinitionIndex name definitions 0
    | .sequentialLoop name _ _ _ _ => findDefinitionIndex name definitions 0
    | _ => none
  for (node, index) in scope.nodes.zipIdx do
    match node.kind with
    | .subgraphCall name _ | .parallelLoop name _ _ _ _ | .sequentialLoop name _ _ _ _ =>
        match definitionIndices[index]? with
        | some (some _) => pure ()
        | _ => throw (OperationalError.missingDefinition name)
    | _ => pure ()
  let mut attachmentBuckets := Array.replicate scope.nodes.size #[]
  for attachment in derivation.attachments do
    validateDerivationAttachment scope attachment
    let readyNode := attachment.roles.foldl (fun current role => max current role.2.node) 0
    match attachmentBuckets[readyNode]? with
    | some bucket => attachmentBuckets := attachmentBuckets.set! readyNode (bucket.push attachment)
    | none => throw (.invalidDerivationAttachment attachment.ownerNamespace attachment.ruleName)
  pure { scope, derivation, inputIndices, definitionIndices, attachmentBuckets }

/-- Checks a frozen program once and resolves every structure-only lookup used by later requests. -/
def prepareProgramOperational
    (program : Prog)
    (derivation : ProgramDerivation) : Except OperationalError PreparedOperationalProgram := do
  match checkProgramDerivation program derivation with
  | .error error => throw (.derivation error)
  | .ok () => pure ()
  let root ← prepareOperationalScope program.definitions program.root derivation.root
  let definitionPairs := program.definitions.zip derivation.definitions
  let definitions ← definitionPairs.mapM fun pair => do
    let ((name, scope), (derivationName, scopeDerivation)) := pair
    if name != derivationName then throw (.missingDefinition name)
    return (name, ← prepareOperationalScope program.definitions scope scopeDerivation)
  pure { root, definitions := definitions.toArray }

private def preparedDefinitionAt
    (node : Nat)
    (prepared : PreparedOperationalScope)
    (definitions : Array (String × PreparedOperationalScope)) :
    Except OperationalError PreparedOperationalScope := do
  let definitionIndex ← match prepared.definitionIndices[node]? with
    | some (some index) => pure index
    | _ => throw (OperationalError.missingDefinition s!"node-{node}")
  match definitions[definitionIndex]? with
  | some (_, definition) => pure definition
  | none => throw (OperationalError.missingDefinition s!"node-{node}")

def deriveOrdinaryOutputs
    (scopeKey : ScopeTemplateKey)
    (nodeIndex : Nat)
    (node : Node)
    (rule : DerivationRule)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (layouts : List Mxx.GadgetLayoutDescriptor)
    (facts : OperationalScopeFacts) :
    Nat → List WireTypeExpr → Except OperationalError (List OperationalFact)
  | _, [] => pure []
  | port, outputType :: tail => do
      let output ← genericNodeFact scopeKey nodeIndex node rule port outputType facts
        environment loopDomains layouts
      let output := namespaceFreshOutput scopeKey { node := nodeIndex, port } output
      return output :: (← deriveOrdinaryOutputs scopeKey nodeIndex node rule environment
        loopDomains layouts facts (port + 1) tail)

def evaluatePreparedScope
    (definitions : Array (String × PreparedOperationalScope))
    (layouts : List Mxx.GadgetLayoutDescriptor) :
    ScopeTemplateKey → Nat → PreparedOperationalScope → ParamEnvironment →
      List OperationalParameterDomain →
      OperationalExprArena →
      List OperationalFact →
      Except OperationalError OperationalScopeFacts
  | _, 0, _, _, _, _, _ => .error .definitionFuelExhausted
  | scopeKey, fuel + 1, prepared, environment, loopDomains, initialArena, inputFacts => do
      let scope := prepared.scope
      let derivation := prepared.derivation
      if !inputFacts.isEmpty && inputFacts.length != scope.inputNames.length then
        throw (.childInputMismatch 0 scope.inputNames.length inputFacts.length)
      let rec collectChildOutputs
          (callerNode port : Nat)
          (outputs : List (String × WireRef))
          (facts : OperationalScopeFacts) : Except OperationalError (List OperationalFact) := do
        match outputs with
        | [] => pure []
        | (_, wire) :: tail =>
            let fact ← lookupFact callerNode facts wire
            let rebound ← rebindSubject { node := callerNode, port } fact
            return rebound :: (← collectChildOutputs callerNode (port + 1) tail facts)
      let rec scopeOutputFacts
          (callerNode : Nat)
          (outputs : List (String × WireRef))
          (facts : OperationalScopeFacts) : Except OperationalError (List OperationalFact) := do
        match outputs with
        | [] => pure []
        | (_, wire) :: tail =>
            return (← lookupFact callerNode facts wire) ::
              (← scopeOutputFacts callerNode tail facts)
      let rec prepareParallelInputs
          (nodeIndex count argumentIndex : Nat)
          (modes : List LoopInputMode)
          (inputs : List OperationalFact) : Except OperationalError (List OperationalFact) := do
        match modes, inputs with
        | [], [] => pure []
        | mode :: modeTail, input :: inputTail =>
            return (← loopTemplateArgumentFact nodeIndex argumentIndex count mode input) ::
              (← prepareParallelInputs nodeIndex count (argumentIndex + 1) modeTail inputTail)
        | _, _ => throw (.loopInputModeMismatch nodeIndex argumentIndex)
      let mut facts : OperationalScopeFacts := { arena := initialArena }
      for node in scope.nodes do
            let index := facts.values.size
            if node.outputCount != node.outputTypes.length then
              throw (.unsupportedOutputArity index node.outputCount)
            let step ← match derivation.steps[index]? with
              | some step => pure step
              | none => throw (.derivation (.missingNode index))
            let outputs ← match node.kind with
              | .input _ =>
                  if inputFacts.isEmpty then
                    deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                      layouts facts 0 node.outputTypes
                  else
                    match prepared.inputIndices[index]? with
                    | some (some inputIndex) =>
                        match inputFacts[inputIndex]? with
                        | some input => do
                            let rebound ← rebindSubject { node := index, port := 0 } input
                            pure [rebound]
                        | none => throw (OperationalError.childInputMismatch index
                            scope.inputNames.length inputFacts.length)
                    | _ => throw (OperationalError.childInputMismatch index
                        scope.inputNames.length inputFacts.length)
              | .subgraphCall _ bindings =>
                  let actualInputs ← node.arguments.mapM (lookupFact index facts)
                  let boundParams ← match evaluateBindings environment bindings with
                    | some values => pure values
                    | none => throw .nonClosedExpression
                  let childDomains ← extendParameterDomains environment loopDomains bindings
                  let child ← preparedDefinitionAt index prepared definitions
                  let childKey := .callBody scopeKey index
                  let childFacts ← (evaluatePreparedScope definitions layouts
                    childKey fuel child (boundParams ++ environment)
                    childDomains facts.arena actualInputs).mapError (.inScope childKey)
                  facts := { facts with arena := childFacts.arena }
                  collectChildOutputs index 0 child.scope.outputs childFacts
              | .familyPack =>
                  let elements ← node.arguments.mapM (lookupFact index facts)
                  pure [packedFacts elements]
              | .familyGetStatic familyIndex =>
                  let familyWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let requested ← match familyIndex.evaluate environment with
                    | some value => pure value
                    | none => throw .nonClosedExpression
                  match ← lookupFact index facts familyWire with
                  | .familyUniform _ coordinate element count =>
                      if requested < 0 || requested >= count then
                        throw (.invalidCount index requested)
                      else
                        let subject : WireRef := { node := index, port := 0 }
                        match element with
                        | .matrixExpr root =>
                            let mapFact (fact : OperationalMatrixFact) :=
                              let mapped := match coordinate with
                                | some (.loopBinder _ _ slot) =>
                                    instantiateFactLoopIndex slot requested.toNat (.matrix fact)
                                | some (.loopBinderOffset _ _ slot offset) =>
                                    instantiateFactLoopIndex slot (requested.toNat + offset)
                                      (.matrix fact)
                                | none => selectProtocolFamilyElement requested.toNat (.matrix fact)
                              match rebindSubject subject mapped with
                              | .ok (.matrix result) => result
                              | _ => fact
                            let mapSelection (selection : DynamicSelectionIdentity) := {
                              index := match coordinate with
                                | some (.loopBinder _ _ slot) =>
                                    instantiateValueOriginLoopIndex slot requested.toNat
                                      selection.index
                                | some (.loopBinderOffset _ _ slot offset) =>
                                    instantiateValueOriginLoopIndex slot (requested.toNat + offset)
                                      selection.index
                                | none => selectProtocolValueOrigin requested.toNat selection.index
                            }
                            let (arena, mapped) ←
                              mapOperationalExpr facts.arena root mapFact mapSelection
                            facts := { facts with arena }
                            pure [.matrixExpr mapped]
                        | element =>
                            let element := match coordinate with
                              | some (.loopBinder _ _ slot) =>
                                  instantiateFactLoopIndex slot requested.toNat element
                              | some (.loopBinderOffset _ _ slot offset) =>
                                  instantiateFactLoopIndex slot (requested.toNat + offset) element
                              | none => selectProtocolFamilyElement requested.toNat element
                            pure [← rebindSubject subject element]
                  | .familyPacked elements count summary =>
                      if requested < 0 || requested >= count then
                        throw (.invalidCount index requested)
                      else if elements.size == count then
                        match elements[requested.toNat]? with
                        | some element => pure [← rebindSubject { node := index, port := 0 } element]
                        | none => throw (.invalidCount index requested)
                      else match summary, elements[0]? with
                        | some summary, some element =>
                            if summary.relationFree then
                              pure [← rebindSubject { node := index, port := 0 }
                                (selectProtocolFamilyElement requested.toNat element)]
                            else throw (.selectedFamilyOperationUnsupported index)
                        | _, _ => throw (.invalidCount index requested)
                  | _ => throw (.loopInputModeMismatch index 0)
              | .familyGetDynamic =>
                  let familyWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let indexWire ← match node.arguments[1]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let selectionFact ← match ← lookupFact index facts indexWire with
                    | .integer fact => pure fact
                    | _ => throw (.loopInputModeMismatch index 1)
                  let selection := selectionFact.origin
                  match ← lookupFact index facts familyWire with
                  | .familyUniform binder _ element count =>
                      -- Operational magnitude analysis is conditional on successful executable
                      -- evaluation. A dynamic access that succeeds returns one family element, so
                      -- a correlated interval need only overlap the valid range. Proving that every
                      -- runtime input stays in range belongs to graph validation/end-to-end
                      -- correctness, not to the noise transfer. Entirely invalid ranges still fail.
                      if count <= 0 || selectionFact.upper < 0 || selectionFact.lower >= count then
                        throw (.invalidCount index selectionFact.upper)
                      else
                        let subject : WireRef := { node := index, port := 0 }
                        match element with
                        | .matrixExpr root =>
                            let mapFact := selectDynamicMatrixFact binder selection subject
                            let mapSelection (nested : DynamicSelectionIdentity) := {
                              index := selectDynamicValueOrigin binder selection nested.index
                            }
                            let (arena, mapped) ←
                              mapOperationalExpr facts.arena root mapFact mapSelection
                            facts := { facts with arena }
                            pure [.matrixExpr mapped]
                        | element =>
                            pure [← selectDynamicUniformFact binder selection subject element]
                  | .familyPacked elements count summary =>
                      if count == 0 || selectionFact.upper < 0 ||
                          selectionFact.lower >= Int.ofNat count then
                        throw (.invalidCount index selectionFact.upper)
                      else if elements.size == count then
                        let binder : FamilyTemplateBinder := {
                          owner := dynamicSelectionScope selection
                          producerNode := index
                          binderSlot := 0
                        }
                        let selectedBranches ← elements.mapM fun branch => do
                          match ← selectDynamicUniformFact binder selection
                              { node := index, port := 0 } branch with
                          | .matrix selected => pure selected
                          | _ => throw (.loopInputModeMismatch index 0)
                        let (arena, root) ← facts.arena.pushExactSelection
                          { index := selection }
                          (selectedBranches.map OperationalFact.matrix)
                        facts := { facts with arena }
                        pure [.matrixExpr root]
                      else match elements[0]? with
                      | some element => match element with
                          | .selectedMatrices alternatives =>
                              if elements.size != 1 then
                                throw (.selectedFamilyOperationUnsupported index)
                              let binder : FamilyTemplateBinder := {
                                owner := dynamicSelectionScope selection
                                producerNode := index
                                binderSlot := 0
                              }
                              let selectedBranches ← alternatives.branches.mapM fun branch => do
                                match ← selectDynamicUniformFact binder selection
                                    { node := index, port := 0 } (.matrix branch) with
                                | .matrix selected => pure selected
                                | _ => throw (.loopInputModeMismatch index 0)
                              pure [.selectedMatrices
                                (selectedMatrixFamily alternatives.selection selectedBranches)]
                          | _ => match summary, elements[0]? with
                              | some summary, some element => match element with
                                  | .matrix representative =>
                                      let binder : FamilyTemplateBinder := {
                                        owner := dynamicSelectionScope selection
                                        producerNode := index
                                        binderSlot := 0
                                      }
                                      let selected ← match ← selectDynamicUniformFact binder selection
                                          { node := index, port := 0 } (.matrix representative) with
                                        | .matrix selected => pure selected
                                        | _ => throw (.loopInputModeMismatch index 0)
                                      let transferred ← match
                                          selectedMatrixSummaryAfterInstantiation summary selected with
                                        | some transferred => pure transferred
                                        | none => throw (.selectedFamilyOperationUnsupported index)
                                      let (arena, representativeId) :=
                                        facts.arena.pushConcrete selected
                                      let (arena, root) ← arena.pushSelect { index := selection }
                                        (.schemaEnvelope count representativeId transferred)
                                      facts := { facts with arena }
                                      pure [.matrixExpr root]
                                  | _ => throw (.loopInputModeMismatch index 0)
                              | _, _ => throw (.selectedFamilyOperationUnsupported index)
                      | none => throw (.selectedFamilyOperationUnsupported index)
                  | _ => throw (.loopInputModeMismatch index 0)
              | .parallelLoop _ count indexSlot bindings modes =>
                  let evaluatedCount ← match count.evaluate environment with
                    | some value => pure value
                    | none => throw .nonClosedExpression
                  if evaluatedCount < 0 then throw (.invalidCount index evaluatedCount)
                  let actualInputs ← node.arguments.mapM (lookupFact index facts)
                  let parentDomains := .loopIndex indexSlot evaluatedCount.toNat ::
                    loopDomains.filter fun domain => match domain with
                      | .loopIndex candidate _ => candidate != indexSlot
                      | .parameter _ _ _ _ => true
                  let child ← preparedDefinitionAt index prepared definitions
                  let childKey := .parallelBody scopeKey index
                  let templateInputs ←
                    prepareParallelInputs index evaluatedCount.toNat 0 modes actualInputs
                  let iterationEnvironment :=
                    (ParamKey.loopIndex indexSlot, ParamValue.integer 0) :: environment
                  let boundParams ← match evaluateBindings iterationEnvironment bindings with
                    | some values => pure values
                    | none => throw .nonClosedExpression
                  let childDomains ←
                    extendParameterDomains iterationEnvironment parentDomains bindings
                  let childFacts ← (evaluatePreparedScope definitions layouts
                    childKey fuel child (boundParams ++ iterationEnvironment)
                    childDomains facts.arena templateInputs).mapError (.inScope childKey)
                  facts := { facts with arena := childFacts.arena }
                  let childOutputs ← scopeOutputFacts index child.scope.outputs childFacts
                  if childOutputs.length != node.outputCount then
                    throw (.childInputMismatch index node.outputCount childOutputs.length)
                  childOutputs.zipIdx.mapM fun (output, port) =>
                    match output with
                    | .selectedMatrices family =>
                        if family.representsLoopLanes then
                          let summary := if family.summary.uniformSchema.isSome then
                            some family.summary else none
                          rebindSubject { node := index, port }
                            (.familyPacked (family.branches.map OperationalFact.matrix)
                              family.count summary)
                        else
                          rebindSubject { node := index, port } (.familyUniform
                            { owner := scopeKey, producerNode := index, binderSlot := indexSlot }
                            (some (.loopBinder scopeKey index indexSlot)) output evaluatedCount)
                    | output =>
                        rebindSubject { node := index, port } (.familyUniform
                          { owner := scopeKey, producerNode := index, binderSlot := indexSlot }
                          (some (.loopBinder scopeKey index indexSlot)) output evaluatedCount)
              | .sequentialLoop _ count indexSlot bindings carriedCount =>
                  let evaluatedCount ← match count.evaluate environment with
                    | some value => pure value
                    | none => throw .nonClosedExpression
                  if evaluatedCount < 0 then throw (.invalidCount index evaluatedCount)
                  let actualInputs ← node.arguments.mapM (lookupFact index facts)
                  let carriedFacts := actualInputs.take carriedCount
                  let invariantFacts := actualInputs.drop carriedCount
                  if carriedFacts.length != carriedCount then
                    throw (.childInputMismatch index carriedCount carriedFacts.length)
                  for (fact, slot) in carriedFacts.zipIdx do
                    if factHasRelation fact then
                      throw (.relationBearingCarriedValue scopeKey index slot)
                  let abstractCarried := carriedFacts.zipIdx.map fun (fact, slot) =>
                    abstractCarriedMaximum slot fact
                  let shiftedInvariantFacts := invariantFacts.map shiftFactPreviousDepth
                  let iterationEnvironment := replaceLoopIndex environment indexSlot 0
                  let sequentialDomains := .loopIndex indexSlot evaluatedCount.toNat ::
                    loopDomains.filter fun domain => match domain with
                      | .loopIndex candidate _ => candidate != indexSlot
                      | .parameter _ _ _ _ => true
                  let boundParams ← match evaluateBindings iterationEnvironment bindings with
                    | some values => pure values
                    | none => throw .nonClosedExpression
                  let childDomains ← extendParameterDomains iterationEnvironment sequentialDomains bindings
                  let child ← preparedDefinitionAt index prepared definitions
                  let childKey := .sequentialBody scopeKey index
                  let childFacts ← (evaluatePreparedScope definitions layouts
                    childKey fuel child
                    (boundParams ++ iterationEnvironment) childDomains
                    facts.arena (abstractCarried ++ shiftedInvariantFacts)).mapError (.inScope childKey)
                  facts := { facts with arena := childFacts.arena }
                  let outputTemplates ← scopeOutputFacts index child.scope.outputs childFacts
                  if outputTemplates.length != carriedCount then
                    throw (.childInputMismatch index carriedCount outputTemplates.length)
                  for slot in List.range carriedCount do
                    match carriedFacts[slot]?, outputTemplates[slot]? with
                    | some initial, some output =>
                        if !sameCarriedSchema initial output || factHasRelation output then
                          if factHasRelation output then
                            throw (.relationBearingCarriedValue scopeKey index slot)
                          else throw (.sequentialSchemaMismatch scopeKey index slot
                            (carriedLargeFactorCounts initial) (carriedLargeFactorCounts output))
                    | _, _ => throw (.childInputMismatch index carriedCount outputTemplates.length)
                  let initialComponents := carriedFacts.zipIdx.flatMap fun (carried, slot) =>
                    factNumericExpressions slot carried
                  let transitionComponents := outputTemplates.zipIdx.flatMap fun (output, slot) =>
                    factNumericExpressions slot output
                  let paths := initialComponents.map (·.1)
                  if paths != transitionComponents.map (·.1) then
                    throw (.sequentialSchemaMismatch scopeKey index 0 [] [])
                  let initialExpressions := initialComponents.map (·.2)
                  let transitions := transitionComponents.map (·.2)
                  let outputs ←
                    if evaluatedCount = 0 then pure carriedFacts
                    else outputTemplates.zipIdx.mapM (fun pair =>
                      setFactRecurrenceState evaluatedCount.toNat paths initialExpressions transitions
                        pair.2 environment pair.1)
                  outputs.zipIdx.mapM fun (output, port) =>
                    rebindSubject { node := index, port } output
              | .concat axis =>
                  let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                  if inputs.any fun input => match input with
                      | .matrixExpr _ => true
                      | _ => false then
                    let matrixType ← match node.outputTypes with
                      | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                      | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                    let (arena, output) ← concatOperationalExprFacts index 0 axis matrixType
                      environment facts.arena inputs
                    facts := { facts with arena }
                    pure [output]
                  else
                    deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                      layouts facts 0 node.outputTypes
              | .crtRecompose plaintextModuli reconstructionCoefficients =>
                  let inputs ← node.arguments.toArray.mapM (lookupFact index facts)
                  if inputs.any fun input => match input with
                      | .matrixExpr _ => true
                      | _ => false then
                    if inputs.isEmpty || inputs.size != plaintextModuli.length ||
                        inputs.size != reconstructionCoefficients.length then
                      throw (.unsupportedOutputArity index inputs.size)
                    let matrixType ← match node.outputTypes with
                      | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                      | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                    let moduli ← plaintextModuli.mapM
                      (evaluateIntInvariant environment loopDomains)
                    let coefficients ← reconstructionCoefficients.mapM
                      (evaluateIntInvariant environment loopDomains)
                    let modulus ← evaluateIntInvariant environment loopDomains matrixType.modulus
                    if modulus <= 0 || moduli.any (fun value => value <= 1 || value >= modulus) ||
                        coefficients.any (fun value => value < 0 || value >= modulus) then
                      throw (.invalidMatrixParameters index)
                    let mut arena := facts.arena
                    let mut scaled : Array OperationalFact := #[]
                    for (input, coefficient) in inputs.toList.zip coefficients do
                      let scalar := IntExpr.constant coefficient
                      let (nextArena, output) ← scaleOperationalExprFact index 0 matrixType scalar
                        [coefficient] environment loopDomains arena input
                      arena := nextArena
                      scaled := scaled.push output
                    let mut output ← match scaled[0]? with
                      | some output => pure output
                      | none => throw (.invalidCount index 0)
                    for next in scaled.extract 1 scaled.size do
                      let (nextArena, sum) ← addOperationalExprFacts index 0 matrixType false
                        environment arena output next
                      arena := nextArena
                      output := sum
                    facts := { facts with arena }
                    pure [output]
                  else
                    deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                      layouts facts 0 node.outputTypes
              | .gadgetDecompose _ _ _ _ =>
                  let inputWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let input ← lookupFact index facts inputWire
                  match input with
                  | .matrixExpr root =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let rec mapExpression : OperationalExprArena → OperationalExprId → Nat →
                          Except OperationalError (OperationalExprArena × OperationalExprId)
                        | _, current, 0 => throw (.unsupportedOperationalExpr current)
                        | arena, current, remaining + 1 => do
                            let expression ← match arena.get? current with
                              | some expression => pure expression
                              | none => throw (.invalidOperationalExprRef current)
                            match expression.node with
                            | .concrete branch =>
                                let branchFacts ← replaceOperationalFact index facts inputWire
                                  (.matrix branch)
                                let output ← genericNodeFact scopeKey index node step.rule 0
                                  (.preimage matrixType) branchFacts environment loopDomains layouts
                                let output := namespaceFreshOutput scopeKey
                                  { node := index, port := 0 } output
                                let output ← match output with
                                  | .matrix output => pure output
                                  | _ => throw (.operandNotMatrix index inputWire)
                                pure (arena.pushConcrete output)
                            | .select selection (.exact branches) =>
                                let mut arena := arena
                                let mut outputs : Array OperationalExprId := #[]
                                for branch in branches do
                                  let (nextArena, output) ← mapExpression arena branch remaining
                                  arena := nextArena
                                  outputs := outputs.push output
                                arena.pushSelect selection (.exact outputs)
                            | .select selection
                                (.schemaEnvelope count representative summary) =>
                                let (arena, output) ←
                                  mapExpression arena representative remaining
                                let outputFact ← arena.concreteFact output
                                let outputSummary ← match
                                    recomputeSelectedMatrixSummary summary outputFact with
                                  | some value => pure value
                                  | none => throw (.unsupportedOperationalExpr representative)
                                arena.pushSelect selection
                                  (.schemaEnvelope count output outputSummary)
                            | _ => throw (.unsupportedOperationalExpr current)
                      let (arena, output) ← mapExpression facts.arena root
                        (facts.arena.nodes.size + 1)
                      facts := { facts with arena }
                      pure [.matrixExpr output]
                  | _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
              | .matrixScale scalar =>
                  let inputWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let input ← lookupFact index facts inputWire
                  match input with
                  | .matrixExpr _ =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let scalarValues ← evaluateIntOverLoops environment loopDomains scalar
                      let (arena, output) ← scaleOperationalExprFact index 0 matrixType scalar
                        scalarValues environment loopDomains facts.arena input
                      facts := { facts with arena }
                      pure [output]
                  | _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
              | .transpose | .matrixNegate | .slice _ _ | .reshape _ _ |
                  .constantCoefficient _ =>
                  let inputWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let input ← lookupFact index facts inputWire
                  match input with
                  | .matrixExpr _ =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let operations : List OperationalFactorTransform := match node.kind with
                        | .transpose => [OperationalFactorTransform.transpose]
                        | .matrixNegate => [OperationalFactorTransform.negate]
                        | .slice rows columns =>
                            rows.toList.map (fun (start, stop) =>
                              OperationalFactorTransform.rowSlice start stop) ++
                            columns.toList.map (fun (start, stop) =>
                              OperationalFactorTransform.columnSlice start stop)
                        | .reshape rows columns =>
                            [OperationalFactorTransform.reshape rows columns]
                        | .constantCoefficient coefficient =>
                            [OperationalFactorTransform.constantCoefficient coefficient]
                        | _ => []
                      match node.kind with
                      | .constantCoefficient coefficient =>
                          let minimum ← evaluateIntMinimum environment loopDomains coefficient
                          let maximum ← evaluateIntMaximum environment loopDomains coefficient
                          let root ← match input with
                            | .matrixExpr root => pure root
                            | _ => throw (.operandNotMatrix index inputWire)
                          let expression ← match facts.arena.get? root with
                            | some expression => pure expression
                            | none => throw (.invalidOperationalExprRef root)
                          let params ← match expression.matrixType.evaluate environment (.constant 0) with
                            | some params => pure params
                            | none => throw (.invalidMatrixParameters index)
                          if node.arguments.length != 1 || params.rows != 1 || params.columns != 1 ||
                              minimum < 0 || maximum >= Int.ofNat params.ringDimension then
                            throw (.invalidCount index maximum)
                      | _ => pure ()
                      let mut arena := facts.arena
                      let mut output := input
                      for operation in operations do
                        let (nextArena, nextOutput) ← transformOperationalExprFact index 0
                          matrixType operation environment arena output
                        arena := nextArena
                        output := nextOutput
                      facts := { facts with arena }
                      pure [output]
                  | _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
              | .matrixAdd | .matrixSubtract =>
                  let leftWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let rightWire ← match node.arguments[1]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index leftWire)
                  let left ← lookupFact index facts leftWire
                  let right ← lookupFact index facts rightWire
                  match left, right with
                  | .matrixExpr _, _ | _, .matrixExpr _ =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let subtract := match node.kind with
                        | .matrixSubtract => true
                        | _ => false
                      let (arena, output) ← addOperationalExprFacts index 0 matrixType subtract
                        environment facts.arena left right
                      facts := { facts with arena }
                      pure [output]
                  | _, _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
              | .matrixMultiply =>
                  let leftWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let rightWire ← match node.arguments[1]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index leftWire)
                  let left ← lookupFact index facts leftWire
                  let right ← lookupFact index facts rightWire
                  match left, right with
                  | .matrixExpr _, _ | _, .matrixExpr _ =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let (arena, output) ← multiplyOperationalExprFacts index 0 matrixType
                        step.rule rightWire environment facts.arena left right
                      facts := { facts with arena }
                      pure [output]
                  | _, _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
              | .tensor =>
                  let leftWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let rightWire ← match node.arguments[1]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index leftWire)
                  let left ← lookupFact index facts leftWire
                  let right ← lookupFact index facts rightWire
                  match left, right with
                  | .matrixExpr _, _ | _, .matrixExpr _ =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let (arena, output) ← tensorOperationalExprFacts index 0 matrixType
                        environment facts.arena left right
                      facts := { facts with arena }
                      pure [output]
                  | _, _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
              | _ =>
                  deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                    layouts facts 0 node.outputTypes
            let mut namespacedOutputs : Array OperationalFact := #[]
            for (output, port) in outputs.toArray.zipIdx do
              match output with
              | .matrixExpr root =>
                  let wire : WireRef := { node := index, port }
                  let mapFact (fact : OperationalMatrixFact) :=
                    match namespaceFreshOutput scopeKey wire (.matrix fact) with
                    | .matrix mapped => mapped
                    | _ => fact
                  let mapSelection (selection : DynamicSelectionIdentity) := {
                    index := namespaceFreshValueOrigin scopeKey wire selection.index
                  }
                  let (arena, mapped) ←
                    mapOperationalExpr facts.arena root mapFact mapSelection
                  facts := { facts with arena }
                  namespacedOutputs := namespacedOutputs.push (.matrixExpr mapped)
              | output => namespacedOutputs := namespacedOutputs.push output
            let outputs := namespacedOutputs
            facts := { facts with values := facts.values.push outputs }
            let attachments := prepared.attachmentBuckets[index]?.getD #[]
            facts := ← applyPreparedDerivationAttachments index attachments facts
      pure facts
termination_by
  _ fuel _ _ _ _ => (fuel, 0)

def evaluatePreparedProgramOperationalWithKey
    (programKey : ProgramInstanceKey)
    (program : PreparedOperationalProgram)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalScopeFacts :=
  evaluatePreparedScope program.definitions layouts
    (.root programKey) (program.definitions.size + 1) program.root environment [] {} []

def evaluateProgramOperationalWithKey
    (programKey : ProgramInstanceKey)
    (program : Prog)
    (derivation : ProgramDerivation)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalScopeFacts := do
  let prepared ← prepareProgramOperational program derivation
  evaluatePreparedProgramOperationalWithKey programKey prepared environment layouts

private def findInputWireType (scope : Scope) (name : String) : Option (WireRef × WireTypeExpr) :=
  scope.nodes.zipIdx.findSome? fun (node, index) =>
    if node.kind == .input name then
      node.outputTypes[0]?.map fun wireType => ({ node := index, port := 0 }, wireType)
    else none

private def evaluateDeclaredBound
    (environment : ParamEnvironment) : DeclaredBoundExpr → Except OperationalError Int
  | .constant value => pure (Int.ofNat value)
  | .parameter value =>
      match value.evaluate environment with
      | some result => pure result
      | none => throw .nonClosedExpression
  | .absolute value =>
      match value.evaluate environment with
      | some result => pure (absolute result)
      | none => throw .nonClosedExpression
  | .add left right => return (← evaluateDeclaredBound environment left) +
      (← evaluateDeclaredBound environment right)
  | .multiply left right => return (← evaluateDeclaredBound environment left) *
      (← evaluateDeclaredBound environment right)
  | .maximum left right => do
      let left ← evaluateDeclaredBound environment left
      let right ← evaluateDeclaredBound environment right
      pure (max left right)
  | .minimum left right => do
      let left ← evaluateDeclaredBound environment left
      let right ← evaluateDeclaredBound environment right
      pure (min left right)
  | .floorDivide value divisor => do
      if divisor = 0 then throw .divisionByZero else
        return (← evaluateDeclaredBound environment value) / Int.ofNat divisor
  | .matrixProduct ringDimension innerDimension left right => do
      let ringDimension ← match ringDimension.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      let innerDimension ← match innerDimension.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      return ringDimension * innerDimension *
        (← evaluateDeclaredBound environment left) * (← evaluateDeclaredBound environment right)

private def contractFact
    (scopeKey : ScopeTemplateKey)
    (subject : WireRef)
    (protocolInput : ProtocolInputId)
    (wireType : WireTypeExpr)
    (contract : InputValueContract)
    (environment : ParamEnvironment) : Except OperationalError OperationalFact := do
  let origin : OperationalValueOrigin := .protocolInput protocolInput
  let setMatrixOrigin : OperationalFact → OperationalFact
    | .matrix fact => .matrix { fact with origin := .protocolInput protocolInput }
    | fact => fact
  match contract, wireType with
  | .matrixExact contractType, .matrix wireMatrixType =>
      if contractType != wireMatrixType then throw (.inputContractMismatch "matrix")
      let cap ← match matrixCap wireMatrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters subject.node)
      return setMatrixOrigin (← classifiedMatrixFact subject.node subject.port wireMatrixType
        environment cap true)
  | .matrixBounded contractType bound, .matrix wireMatrixType =>
      if contractType != wireMatrixType then throw (.inputContractMismatch "matrix")
      let maximum ← evaluateDeclaredBound environment bound
      return setMatrixOrigin (← cappedMatrixFact subject.node subject.port wireMatrixType
        environment maximum)
  | .integerRange lower upper, .integer | .integerRange lower upper, .constantInt =>
      let evaluatedLower ← match lower.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      let evaluatedUpper ← match upper.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      if evaluatedLower > evaluatedUpper then throw (.inputContractMismatch "integer range")
      pure (.integer {
        subject
        origin
        lower := evaluatedLower
        upper := evaluatedUpper
        lowerExpression := .closedInt (.constant evaluatedLower)
        upperExpression := .closedInt (.constant evaluatedUpper)
      })
  | .boolean, .boolean | .boolean, .constantBool => pure .boolean
  | .bytes contractLength, .bytes wireLength =>
      let contractLength ← match contractLength.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      let wireLength ← match wireLength.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      if contractLength != wireLength then throw (.inputContractMismatch "bytes")
      pure (.bytes { subject, origin, length := contractLength })
  | .family contractCount elementContract, .indexedFamily elementType wireCount =>
      let contractCount ← match contractCount.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      let wireCount ← match wireCount.evaluate environment with
        | some value => pure value | none => throw .nonClosedExpression
      if contractCount != wireCount || contractCount < 0 then
        throw (.inputContractMismatch "family count")
      let element ← contractFact scopeKey subject protocolInput elementType elementContract environment
      pure (.familyUniform
        { owner := scopeKey, producerNode := subject.node, binderSlot := subject.port }
        none element contractCount)
  | _, _ => throw (.inputContractMismatch "wire type")

structure OperationalStageResult where
  stage : String
  outputs : List (String × OperationalFact)
  facts : OperationalScopeFacts

/-- A closed, generic parameter obligation derived from operational facts. Applications select
the relevant output fact, but do not implement their own arithmetic acceptance condition. -/
inductive OperationalNoiseObligation where
  | decoderThreshold
      (plaintextModulus ciphertextModulus noiseBound : Int)
  deriving BEq, DecidableEq, Repr

/-- Stable, protocol-independent reasons why a closed operational report is rejected. -/
inductive OperationalNoiseRejection where
  | invalidPlaintextModulus (value : Int)
  | invalidCiphertextModulus (value : Int)
  | invalidNoiseBound (value : Int)
  | decoderThresholdNotMet
      (plaintextModulus ciphertextModulus noiseBound : Int)
  deriving BEq, DecidableEq, Repr

/-- Result consumed by parameter search. The evaluated workflow outputs are retained so callers
can inspect the facts that produced each obligation; acceptance is only the conjunction of the
listed closed obligations. Wall-clock timing belongs to the IO caller, not this pure result. -/
structure OperationalNoiseCheckReport where
  outputs : List OperationalStageResult
  obligations : List OperationalNoiseObligation
  accepted : Bool
  rejection : Option OperationalNoiseRejection

private def checkDecoderThreshold
    (plaintextModulus ciphertextModulus noiseBound : Int) :
    Bool × Option OperationalNoiseRejection :=
  if plaintextModulus <= 1 then
    (false, some (.invalidPlaintextModulus plaintextModulus))
  else if ciphertextModulus <= 0 then
    (false, some (.invalidCiphertextModulus ciphertextModulus))
  else if noiseBound < 0 then
    (false, some (.invalidNoiseBound noiseBound))
  else if 2 * plaintextModulus * noiseBound < ciphertextModulus then
    (true, none)
  else
    (false, some (.decoderThresholdNotMet
      plaintextModulus ciphertextModulus noiseBound))

/-- Builds the generic decoder report used by parameter search. This definition intentionally
uses multiplication rather than an integer division such as `noise < q / 4`, so boundary behavior
is exactly the stated strict inequality for every plaintext modulus. -/
def decoderNoiseCheckReport
    (outputs : List OperationalStageResult)
    (residual : OperationalMatrixFact)
    (environment : ParamEnvironment)
    (plaintextModulus ciphertextModulus : Int) :
    Except OperationalError OperationalNoiseCheckReport := do
  let noiseBound ← residual.evaluateNoiseHardBound environment
  let obligation := OperationalNoiseObligation.decoderThreshold
    plaintextModulus ciphertextModulus noiseBound
  let (accepted, rejection) :=
    checkDecoderThreshold plaintextModulus ciphertextModulus noiseBound
  pure { outputs, obligations := [obligation], accepted, rejection }

private def collectOperationalExprNoiseBoundsWithFuel
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalExprId → Nat →
    Except OperationalError (List Int)
  | root, 0 => throw (.unsupportedOperationalExpr root)
  | root, fuel + 1 => do
      let expression ← match arena.get? root with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef root)
      match expression.node with
      | .concrete residual => return [← residual.evaluateNoiseHardBound environment]
      | .select _ (.exact branches) =>
          if branches.isEmpty then throw (.invalidCount 0 0)
          let rows ← branches.toList.mapM fun branch =>
            collectOperationalExprNoiseBoundsWithFuel arena environment branch fuel
          pure rows.flatten
      | .select _ (.schemaEnvelope count representative _) =>
          if count = 0 then throw (.invalidCount 0 0)
          collectOperationalExprNoiseBoundsWithFuel arena environment representative fuel
      | .add .. | .subtract .. | .multiply .. | .tensor .. | .concat .. | .transform .. =>
          throw (.unsupportedOperationalExpr root)

private def collectOperationalExprNoiseBounds
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId) : Except OperationalError (List Int) :=
  collectOperationalExprNoiseBoundsWithFuel arena environment root (arena.nodes.size + 1)

private partial def collectDecoderResidualBounds
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalFact → Except OperationalError (List Int)
  | .matrix residual => return [← residual.evaluateNoiseHardBound environment]
  | .matrixExpr root => collectOperationalExprNoiseBounds arena environment root
  | .familyUniform _ _ element count => do
      if count <= 0 then
        throw (.invalidCount 0 count)
      collectDecoderResidualBounds arena environment element
  | .familyPacked elements _ _ => do
      if elements.isEmpty then throw (.invalidCount 0 0)
      let rows ← elements.toList.mapM (collectDecoderResidualBounds arena environment)
      pure rows.flatten
  | .selectedMatrices family => do
      if family.branches.isEmpty then throw (.invalidCount 0 0)
      family.branches.toList.mapM fun branch => branch.evaluateNoiseHardBound environment
  | _ => throw (.operandNotMatrix 0 { node := 0, port := 0 })

/-- Builds one decoder obligation for a matrix residual or an entire residual family. Packed
families are checked member-by-member and use their maximum bound. A uniform family uses the
element fact whose uniformity was established by operational elaboration; empty and non-matrix
families fail closed. -/
def decoderNoiseCheckReportForFact
    (outputs : List OperationalStageResult)
    (arena : OperationalExprArena)
    (residual : OperationalFact)
    (environment : ParamEnvironment)
    (plaintextModulus ciphertextModulus : Int) :
    Except OperationalError OperationalNoiseCheckReport := do
  let bounds ← collectDecoderResidualBounds arena environment residual
  let noiseBound ← match bounds with
    | head :: tail => pure (tail.foldl max head)
    | [] => throw (OperationalError.invalidCount 0 0)
  let obligation := OperationalNoiseObligation.decoderThreshold
    plaintextModulus ciphertextModulus noiseBound
  let (accepted, rejection) :=
    checkDecoderThreshold plaintextModulus ciphertextModulus noiseBound
  pure { outputs, obligations := [obligation], accepted, rejection }

private def collectOperationalOutputs
    (scope : Scope)
    (facts : OperationalScopeFacts) : Except OperationalError (List (String × OperationalFact)) :=
  scope.outputs.mapM fun (name, wire) => return (name, ← lookupFact scope.nodes.size facts wire)

private def findStageOutput
    (results : List OperationalStageResult)
    (stage output : String) : Except OperationalError OperationalFact := do
  let result ← match results.find? fun result => result.stage == stage with
    | some result => pure result
    | none => throw (.missingStageResult stage output)
  match result.outputs.find? fun candidate => candidate.1 == output with
  | some (_, fact) => pure fact
  | none => throw (.missingStageResult stage output)

structure PreparedOperationalStageInput where
  subject : WireRef
  wireType : WireTypeExpr
  source : InputSource

structure PreparedOperationalStage where
  id : String
  program : PreparedOperationalProgram
  inputs : Array PreparedOperationalStageInput

structure PreparedOperationalWorkflow where
  stages : Array PreparedOperationalStage
  inputContract : InputContract

/-- Validates every frozen stage and resolves its structural lookups exactly once. -/
def prepareWorkflowOperational
    (bundle : OperationalWorkflowSpec)
    (stageDerivations : List (String × ProgramDerivation)) :
    Except OperationalError PreparedOperationalWorkflow := do
  let mut stages := #[]
  for stage in bundle.workflow.stages do
    let derivation ← match stageDerivations.find? fun candidate => candidate.1 == stage.id with
      | some (_, derivation) => pure derivation
      | none => throw (.missingStageDerivation stage.id)
    let program ← prepareProgramOperational stage.program derivation
    let inputs ← stage.inputs.mapM fun (inputName, source) => do
      let (subject, wireType) ← match findInputWireType stage.program.root inputName with
        | some result => pure result
        | none => throw (.missingInputNode inputName)
      pure { subject, wireType, source }
    stages := stages.push { id := stage.id, program, inputs := inputs.toArray }
  pure { stages, inputContract := bundle.inputContract }

/-- Evaluates request-dependent bounds using a workflow whose structure is already checked. -/
def evaluatePreparedWorkflowOperational
    (prepared : PreparedOperationalWorkflow)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) :
    Except OperationalError (List OperationalStageResult) := do
  let mut results := []
  let mut arena : OperationalExprArena := {}
  for stage in prepared.stages do
    let scopeKey := ScopeTemplateKey.root (.workflowStage ⟨stage.id⟩)
    let inputFacts ← stage.inputs.toList.mapM fun input => match input.source with
      | .artifact producer output => do
          rebindSubject input.subject (← findStageOutput results producer output)
      | .protocol protocolName => do
          let (protocolInput, contract) ← match prepared.inputContract.inputs.find? fun entry =>
              entry.1.name == protocolName with
            | some (protocolInput, _, contract) => pure (protocolInput, contract)
            | none => throw (.missingProtocolContract protocolName)
          contractFact scopeKey input.subject protocolInput input.wireType contract environment
    let facts ← evaluatePreparedScope stage.program.definitions layouts scopeKey
      (stage.program.definitions.size + 1) stage.program.root environment [] arena inputFacts
    arena := facts.arena
    let outputs ← collectOperationalOutputs stage.program.root.scope facts
    results := results ++ [{ stage := stage.id, outputs, facts }]
  pure results

/-- Evaluates the exact frozen workflow in stage order. Protocol inputs are constructed from the
reviewed input contract; artifact inputs are the producer's actual operational output facts, so
relations and identities cross a stage boundary without graph search or user annotations. -/
def evaluateWorkflowOperational
    (bundle : OperationalWorkflowSpec)
    (stageDerivations : List (String × ProgramDerivation))
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) :
    Except OperationalError (List OperationalStageResult) := do
  let prepared ← prepareWorkflowOperational bundle stageDerivations
  evaluatePreparedWorkflowOperational prepared environment layouts

def evaluateProgramOperationalWithLayouts
    (program : Prog)
    (derivation : ProgramDerivation)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalScopeFacts :=
  evaluateProgramOperationalWithKey (.standalone 0) program derivation environment layouts

def evaluateScopeOperationalWithKey
    (scopeKey : ScopeTemplateKey)
    (scope : Scope)
    (derivation : ScopeDerivation)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor)
    (inputFacts : List OperationalFact := []) : Except OperationalError OperationalScopeFacts := do
  let program : Prog := { root := scope }
  let programDerivation : ProgramDerivation := { root := derivation }
  let prepared ← prepareProgramOperational program programDerivation
  evaluatePreparedScope prepared.definitions layouts scopeKey 1 prepared.root environment [] {}
    inputFacts

def evaluateScopeOperationalWithLayouts
    (scope : Scope)
    (derivation : ScopeDerivation)
    (environment : ParamEnvironment)
    (layouts : List Mxx.GadgetLayoutDescriptor) : Except OperationalError OperationalScopeFacts :=
  evaluateScopeOperationalWithKey (.root (.standalone 0))
    scope derivation environment layouts

/-- Future local proof target for ordinary addition.  It intentionally states the runtime
connection without presenting the operational estimate as an established theorem. -/
def MatrixAddOperationalSoundnessClaim : Prop :=
  ∀ (scope : Scope) (derivation : ScopeDerivation) (environment : ParamEnvironment),
    checkScopeDerivation scope derivation = .ok () →
      ∃ facts, evaluateScopeOperationalWithLayouts scope derivation environment [] = .ok facts

/-- The same protocol input has one root identity across workflow stages even though each stage
binds it to a different local subject wire. -/
example : (do
    let input : ProtocolInputId := ⟨"shared-key"⟩
    let left ← contractFact (.root (.workflowStage ⟨"left"⟩)) { node := 0, port := 0 }
      input (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let right ← contractFact (.root (.workflowStage ⟨"right"⟩)) { node := 7, port := 0 }
      input (.bytes (.constant 32)) (.bytes (.constant 32)) []
    match left, right with
    | .bytes left, .bytes right => pure (left.origin == right.origin)
    | _, _ => pure false) = .ok true := by
  native_decide

/-- Equal-looking values from different protocol inputs remain distinct. -/
example : (do
    let left ← contractFact (.root (.workflowStage ⟨"left"⟩)) { node := 0, port := 0 }
      ⟨"left-key"⟩ (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let right ← contractFact (.root (.workflowStage ⟨"right"⟩)) { node := 0, port := 0 }
      ⟨"right-key"⟩ (.bytes (.constant 32)) (.bytes (.constant 32)) []
    match left, right with
    | .bytes left, .bytes right => pure (left.origin != right.origin)
    | _, _ => pure false) = .ok true := by
  native_decide

/-- Static elements of one external family retain the root input identity and the selected index. -/
example : (do
    let family ← contractFact (.root (.workflowStage ⟨"stage"⟩)) { node := 0, port := 0 }
      ⟨"keys"⟩ (.indexedFamily (.bytes (.constant 32)) (.constant 2))
      (.family (.constant 2) (.bytes (.constant 32))) []
    match family with
    | .familyUniform _ _ element _ =>
        match selectProtocolFamilyElement 0 element, selectProtocolFamilyElement 1 element with
        | .bytes first, .bytes second => pure (first.origin != second.origin)
        | _, _ => pure false
    | _ => pure false) = .ok true := by
  native_decide

/-- Repeating the same dynamic external-family access preserves the selected value identity. -/
example : (do
    let element ← contractFact (.root (.workflowStage ⟨"stage"⟩)) { node := 0, port := 0 }
      ⟨"keys"⟩ (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let binder : FamilyTemplateBinder := {
      owner := .root (.workflowStage ⟨"stage"⟩), producerNode := 0, binderSlot := 0 }
    let first ← selectDynamicUniformFact binder
      (.local (.root (.workflowStage ⟨"stage"⟩)) { node := 4, port := 0 })
      { node := 5, port := 0 } element
    let second ← selectDynamicUniformFact binder
      (.local (.root (.workflowStage ⟨"stage"⟩)) { node := 4, port := 0 })
      { node := 6, port := 0 } element
    match first, second with
    | .bytes first, .bytes second => pure (first.origin == second.origin)
    | _, _ => pure false) = .ok true := by
  native_decide

private def fixtureType : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 1, columns := .constant 1
}

private def fixtureParams : Mxx.SamplerParams := {
  maxCoefficientBound := 8
  modulus := 17
  ringDimension := 1
  rows := 1
  columns := 1
}

/-- An exact external matrix is not a zero matrix. Without an explicit bounded contract it keeps
the conservative centered-residue cap and a Large primitive factor. -/
example : (do
    let fact ← contractFact (.root (.workflowStage ⟨"stage"⟩)) { node := 0, port := 0 }
      ⟨"matrix"⟩ (.matrix fixtureType) (.matrixExact fixtureType) []
    match fact with
    | .matrix matrix =>
        if matrix.polynomial.any operationalTermIsSignal then
          matrix.totalHardBound.evaluate [] #[]
        else pure (-1)
    | _ => pure (-1)) = .ok 8 := by
  native_decide

private def fixtureFamilyBinder : FamilyTemplateBinder := {
  owner := .root (.standalone 7)
  producerNode := 4
  binderSlot := 0
}

private def fixtureSampledIdentity : PublicMatrixIdentity :=
  .sampledTrapdoor (.parallelBody (.root (.standalone 7)) 4) { node := 0, port := 0 }

private def fixturePublicFact : OperationalFact := OperationalFact.matrix (({
  subject := { node := 0, port := 0 }
  origin := .value (.parallelBody (.root (.standalone 7)) 4) { node := 0, port := 0 }
  matrixType := fixtureType
  matrixParams := fixtureParams
  totalHardBound := .closedInt (.constant 8)
  identity := some fixtureSampledIdentity
} : OperationalMatrixFact).initializePrimitivePolynomial .large)

private def fixtureTrapdoorFact : OperationalFact := .trapdoor {
  subject := { node := 0, port := 1 }
  matrixType := fixtureType
  matrixParams := fixtureParams
  maximum := .closedInt (.constant 3)
  publicIdentity := fixtureSampledIdentity
}

private def sharedPreimageBaseScope : Scope := {
  nodes := #[
    {
      kind := .trapdoorSample fixtureType (.constant 3)
      arguments := []
      outputCount := 2
      outputTypes := [
        .matrix fixtureType,
        .trapdoor fixtureType (.rational 1) (.constant 2) (.constant 1) (.constant 3)
      ]
    },
    {
      kind := .gaussianSample fixtureType (.constant 2)
      arguments := []
      outputTypes := [.matrix fixtureType]
    },
    {
      kind := .identityMatrix fixtureType
      arguments := []
      outputTypes := [.matrix fixtureType]
    },
    {
      kind := .preimageSample fixtureType (.constant 3)
      arguments := [
        { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 1, port := 0 }
      ]
      outputTypes := [.preimage fixtureType]
    },
    {
      kind := .preimageSample fixtureType (.constant 3)
      arguments := [
        { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 2, port := 0 }
      ]
      outputTypes := [.preimage fixtureType]
    }
  ]
  outputs := [
    ("first", { node := 3, port := 0 }),
    ("second", { node := 4, port := 0 })
  ]
  inputNames := []
}

private def sharedPreimageBaseDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .trapdoorSample, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .identityMatrix, arguments := [] },
  { sourceNode := 3, rule := .preimageSample,
    arguments := [
      { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 1, port := 0 }
    ] },
  { sourceNode := 4, rule := .preimageSample,
    arguments := [
      { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 2, port := 0 }
    ] }
] }

/-- Branch-specific targets create distinct preimages and target snapshots, but both relations
retain the one source public matrix identity. This is the Diamond transition shape
`B*K_d = P_d (mod R_q)`: the digit changes `K_d` and `P_d`, never `B`. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts sharedPreimageBaseScope
      sharedPreimageBaseDerivation [] []
    let first ← matrixFactAt 4 facts { node := 3, port := 0 }
    let second ← matrixFactAt 4 facts { node := 4, port := 0 }
    match first.relations, second.relations with
    | [.preimage left], [.preimage right] =>
        pure (left.publicIdentity == right.publicIdentity &&
          left.targetOrigin != right.targetOrigin && left.producer != right.producer)
    | _, _ => pure false) = .ok true := by
  native_decide

/-- The exact same family and exact same executable index wire preserve the public/private pair. -/
example : (do
    let selection : OperationalValueOrigin :=
      .local (.root (.standalone 7)) { node := 3, port := 0 }
    let publicFact ← selectDynamicUniformFact fixtureFamilyBinder selection
      { node := 5, port := 0 } fixturePublicFact
    let trapdoor ← selectDynamicUniformFact fixtureFamilyBinder selection
      { node := 6, port := 0 } fixtureTrapdoorFact
    match publicFact, trapdoor with
    | .matrix publicFact, .trapdoor trapdoor =>
        pure (publicFact.identity == some trapdoor.publicIdentity)
    | _, _ => pure false) = .ok true := by
  native_decide

/-- Merely equal-looking selections from different executable index wires do not compare equal. -/
example : (do
    let publicFact ← selectDynamicUniformFact fixtureFamilyBinder
      (.local (.root (.standalone 7)) { node := 3, port := 0 })
      { node := 5, port := 0 } fixturePublicFact
    let trapdoor ← selectDynamicUniformFact fixtureFamilyBinder
      (.local (.root (.standalone 7)) { node := 4, port := 0 })
      { node := 6, port := 0 } fixtureTrapdoorFact
    match publicFact, trapdoor with
    | .matrix publicFact, .trapdoor trapdoor =>
        pure (!(publicFact.identity == some trapdoor.publicIdentity))
    | _, _ => pure false) = .ok true := by
  native_decide

/-- The flat polynomial, not merely the outer fact, preserves dynamic-selection identity. -/
example : (do
    let selection : OperationalValueOrigin :=
      .local (.root (.standalone 7)) { node := 3, port := 0 }
    let first ← selectDynamicUniformFact fixtureFamilyBinder selection
      { node := 5, port := 0 } fixturePublicFact
    let same ← selectDynamicUniformFact fixtureFamilyBinder selection
      { node := 6, port := 0 } fixturePublicFact
    let different ← selectDynamicUniformFact fixtureFamilyBinder
      (.local (.root (.standalone 7)) { node := 4, port := 0 })
      { node := 7, port := 0 } fixturePublicFact
    match first, same, different with
    | .matrix first, .matrix same, .matrix different =>
        pure (first.polynomial == same.polynomial && first.polynomial != different.polynomial)
    | _, _, _ => pure false) = .ok true := by
  native_decide

private def fixtureScope : Scope := {
  nodes := #[
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 2, port := 0 })], inputNames := []
}

private def fixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd, arguments := [{ node := 0, port := 0 },
    { node := 1, port := 0 }] }
] }

example : (do
    let facts ← evaluateScopeOperationalWithLayouts fixtureScope fixtureDerivation [] []
    matrixMaximum 2 { node := 2, port := 0 } facts) = .ok 3 := by
  native_decide

/-- A fresh sample produced by one parallel-body template denotes a different source in each
lane, so subtraction across distinct lanes cannot cancel structurally. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts fixtureScope fixtureDerivation [] []
    let sample ← lookupFact 2 facts { node := 1, port := 0 }
    match instantiateFactLoopIndex 0 0 sample, instantiateFactLoopIndex 0 1 sample with
    | .matrix first, .matrix second =>
        pure (!(subtractOperationalPolynomials first.polynomial second.polynomial).isEmpty)
    | _, _ => pure false) = .ok true := by
  native_decide

private def scaledNoiseScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixScale (.constant 2), arguments := [{ node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := []
}

private def scaledNoiseDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .matrixScale, arguments := [{ node := 0, port := 0 }] }
] }

/-- The additive coefficient outside a compressed bounded product remains part of its bound. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let fact ← matrixFactAt 1 facts { node := 1, port := 0 }
    fact.evaluateNoiseHardBound []) = .ok 6 := by
  native_decide

private def mixedSignalNoiseScope : Scope := {
  nodes := #[
    { kind := .uniformResidueSample fixtureType, arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 2, port := 0 })]
  inputNames := []
}

private def mixedSignalNoiseDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .uniformResidueSample, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
] }

/-- A mixed signal/noise value keeps an unconditional whole-value cap while exposing noise
separately for the endpoint inequality. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts mixedSignalNoiseScope
      mixedSignalNoiseDerivation [] []
    let fact ← matrixFactAt 2 facts { node := 2, port := 0 }
    let total ← fact.totalHardBound.evaluate [] #[]
    let noise ← fact.evaluateNoiseHardBound []
    pure (total, noise)) = .ok (8, 3) := by
  native_decide

private def flatCancellationScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixSubtract,
      arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := []
}

private def flatCancellationDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .matrixSubtract,
    arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }] }
] }

/-- Exact factor-list equality, rather than equality of numeric bounds, eliminates `E-E`. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts flatCancellationScope
      flatCancellationDerivation [] []
    let result ← matrixFactAt 1 facts { node := 1, port := 0 }
    pure result.polynomial.isEmpty) = .ok true := by
  native_decide

private def flatNoiseOrderScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixSubtract,
      arguments := [{ node := 2, port := 0 }, { node := 3, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 4, port := 0 })]
  inputNames := []
}

private def flatNoiseOrderDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .matrixAdd,
    arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }] },
  { sourceNode := 4, rule := .matrixSubtract,
    arguments := [{ node := 2, port := 0 }, { node := 3, port := 0 }] }
] }

/-- Canonical bounded-noise provenance is independent of additive construction order. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts flatNoiseOrderScope
      flatNoiseOrderDerivation [] []
    let result ← matrixFactAt 4 facts { node := 4, port := 0 }
    pure result.polynomial.isEmpty) = .ok true := by
  native_decide

private def flatMultiLargeScope : Scope := {
  nodes := #[
    { kind := .uniformResidueSample fixtureType, arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .uniformResidueSample fixtureType, arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 2, port := 0 }, { node := 2, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 3, port := 0 })]
  inputNames := []
}

private def flatMultiLargeDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .uniformResidueSample, arguments := [] },
  { sourceNode := 1, rule := .uniformResidueSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .matrixMultiplyBound,
    arguments := [{ node := 2, port := 0 }, { node := 2, port := 0 }] }
] }

/-- Multiplication distributes over signal sums; two Large factors remain signal, not opaque. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts flatMultiLargeScope
      flatMultiLargeDerivation [] []
    let result ← matrixFactAt 3 facts { node := 3, port := 0 }
    pure (result.polynomial.length, result.polynomial.all fun term =>
      operationalLargeFactorCount term = 2)) = .ok (4, true) := by
  native_decide

example : checkScopeDerivation fixtureScope { steps := #[
  { sourceNode := 1, rule := .gaussianSample, arguments := [] }
] } = .error (.sourceNodeMismatch 0 1) := by
  native_decide

private def gadgetFixtureScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] }
  ],
  outputs := [("result", { node := 1, port := 0 })], inputNames := []
}

private def gadgetFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] }
] }

/-- A request cannot silently derive a CRT layout from a graph-visible modulus. -/
example : (match evaluateScopeOperationalWithLayouts gadgetFixtureScope gadgetFixtureDerivation [] [] with
    | .error (.missingGadgetLayout 1) => true
    | _ => false) = true := by
  native_decide

private def fixtureLayout : Mxx.GadgetLayoutDescriptor := {
  paramsId := "fixture"
  ringDimension := 1
  crtModuli := [17]
  crtBits := 1
  baseBits := 1
  base := 2
  regularDigitCount := 1
  smallDigitCount := 1
  smallestCrtModulus := 17
}

private def fixtureRows2Type : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 2, columns := .constant 1
}

private def fixtureColumns2Type : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 1, columns := .constant 2
}

private def fixtureSquare2Type : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 2, columns := .constant 2
}

private def matrixTransformCoverageScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .identityMatrix fixtureType, arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixSubtract,
      arguments := [{ node := 2, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 1, port := 0 }, { node := 3, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixNegate, arguments := [{ node := 4, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixScale (.constant (-2)), arguments := [{ node := 5, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .transpose, arguments := [{ node := 6, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .slice none none, arguments := [{ node := 7, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .reshape (.constant 1) (.constant 1), arguments := [{ node := 8, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .constantCoefficient (.constant 0), arguments := [{ node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .tensor, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .concat .rows, arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureRows2Type] },
    { kind := .transpose, arguments := [{ node := 12, port := 0 }],
      outputTypes := [.matrix fixtureColumns2Type] },
    { kind := .concat .columns,
      arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureColumns2Type] },
    { kind := .concat .diagonal,
      arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureSquare2Type] },
    { kind := .slice (some (.constant 0, .constant 1))
        (some (.constant 0, .constant 1)), arguments := [{ node := 15, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .uniformIntervalSample fixtureType (.constant (-2)) (.constant 4), arguments := [],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 17, port := 0 })]
  inputNames := []
}

private def matrixTransformCoverageDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .identityMatrix, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .matrixSubtract,
    arguments := [{ node := 2, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 4, rule := .matrixMultiplyBound,
    arguments := [{ node := 1, port := 0 }, { node := 3, port := 0 }] },
  { sourceNode := 5, rule := .matrixNegate, arguments := [{ node := 4, port := 0 }] },
  { sourceNode := 6, rule := .matrixScale, arguments := [{ node := 5, port := 0 }] },
  { sourceNode := 7, rule := .transpose, arguments := [{ node := 6, port := 0 }] },
  { sourceNode := 8, rule := .slice, arguments := [{ node := 7, port := 0 }] },
  { sourceNode := 9, rule := .reshape, arguments := [{ node := 8, port := 0 }] },
  { sourceNode := 10, rule := .constantCoefficient, arguments := [{ node := 0, port := 0 }] },
  { sourceNode := 11, rule := .tensor,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 12, rule := .concat,
    arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }] },
  { sourceNode := 13, rule := .transpose, arguments := [{ node := 12, port := 0 }] },
  { sourceNode := 14, rule := .concat,
    arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }] },
  { sourceNode := 15, rule := .concat,
    arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }] },
  { sourceNode := 16, rule := .slice, arguments := [{ node := 15, port := 0 }] },
  { sourceNode := 17, rule := .uniformIntervalSample, arguments := [] }
] }

/-- Every non-relation matrix arithmetic/transform variant reaches an explicit operational
transfer. The equalities below also pin conservative inter-node bounded-summary subtraction,
centered-cap scaling, coefficient selection, tensor-with-identity, and interval sampling. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts matrixTransformCoverageScope
      matrixTransformCoverageDerivation [] []
    let afterCancellation ← matrixMaximum 17 { node := 3, port := 0 } facts
    let afterScale ← matrixMaximum 17 { node := 6, port := 0 } facts
    let coefficient ← matrixMaximum 17 { node := 10, port := 0 } facts
    let tensor ← matrixMaximum 17 { node := 11, port := 0 } facts
    let interval ← matrixMaximum 17 { node := 17, port := 0 } facts
    pure (afterCancellation, afterScale, coefficient, tensor, interval)) =
      .ok (5, 8, 3, 3, 4) := by
  native_decide

private def samplerAndDecodeCoverageScope : Scope := {
  nodes := #[
    { kind := .trapdoorSample fixtureType (.constant 3), arguments := [], outputCount := 2,
      outputTypes := [
        .matrix fixtureType,
        .trapdoor fixtureType (.rational 1) (.constant 2) (.constant 1) (.constant 3)
      ] },
    { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .preimageSample fixtureType (.constant 3),
      arguments := [
        { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 1, port := 0 }
      ], outputTypes := [.preimage fixtureType] },
    { kind := .trapdoorPublic, arguments := [{ node := 0, port := 1 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .thresholdDecodeBool (.constant 17) (.constant 2) (.constant 1),
      arguments := [{ node := 1, port := 0 }], outputTypes := [.boolean] },
    { kind := .thresholdDecodeInt (.constant 17) (.constant 3) (.constant 1),
      arguments := [{ node := 1, port := 0 }], outputTypes := [.integer] },
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .identityMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .crtRecompose [.constant 2, .constant 3] [.constant 9, .constant 6],
      arguments := [{ node := 6, port := 0 }, { node := 7, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .constantBool false, arguments := [], outputTypes := [.boolean] },
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .constantBool false, arguments := [], outputTypes := [.boolean] },
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .constantBool false, arguments := [], outputTypes := [.boolean] },
    { kind := .familyPack,
      arguments := [
        { node := 9, port := 0 }, { node := 10, port := 0 }, { node := 11, port := 0 },
        { node := 12, port := 0 }, { node := 13, port := 0 }
      ], outputTypes := [.indexedFamily .boolean (.constant 5)] },
    { kind := .packPolynomialCoefficients fixtureType (.constant 5),
      arguments := [{ node := 14, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("packed", { node := 15, port := 0 })]
  inputNames := []
}

private def samplerAndDecodeCoverageDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .trapdoorSample, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .preimageSample,
    arguments := [
      { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 1, port := 0 }
    ] },
  { sourceNode := 3, rule := .trapdoorPublic, arguments := [{ node := 0, port := 1 }] },
  { sourceNode := 4, rule := .thresholdDecodeBool, arguments := [{ node := 1, port := 0 }] },
  { sourceNode := 5, rule := .thresholdDecodeInt, arguments := [{ node := 1, port := 0 }] },
  { sourceNode := 6, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 7, rule := .identityMatrix, arguments := [] },
  { sourceNode := 8, rule := .crtRecompose,
    arguments := [{ node := 6, port := 0 }, { node := 7, port := 0 }] },
  { sourceNode := 9, rule := .constantBool, arguments := [] },
  { sourceNode := 10, rule := .constantBool, arguments := [] },
  { sourceNode := 11, rule := .constantBool, arguments := [] },
  { sourceNode := 12, rule := .constantBool, arguments := [] },
  { sourceNode := 13, rule := .constantBool, arguments := [] },
  { sourceNode := 14, rule := .familyPack,
    arguments := [
      { node := 9, port := 0 }, { node := 10, port := 0 }, { node := 11, port := 0 },
      { node := 12, port := 0 }, { node := 13, port := 0 }
    ] },
  { sourceNode := 15, rule := .packPolynomialCoefficients,
    arguments := [{ node := 14, port := 0 }] }
] }

/-- Sampler pairing, preimage ownership, threshold outputs, CRT recomposition, Boolean-family
packing, and residue reconstruction all reach explicit transfers in one closed fixture. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts samplerAndDecodeCoverageScope
      samplerAndDecodeCoverageDerivation [] []
    let publicFact ← matrixFactAt 15 facts { node := 0, port := 0 }
    let recovered ← matrixFactAt 15 facts { node := 3, port := 0 }
    let preimage ← matrixFactAt 15 facts { node := 2, port := 0 }
    let decoded ← integerFactAt 15 facts { node := 5, port := 0 }
    let packed ← matrixFactAt 15 facts { node := 15, port := 0 }
    pure (publicFact.identity == recovered.identity, preimage.relations.length,
      decoded.lower, decoded.upper, packed.polynomial.any operationalTermIsSignal)) =
      .ok (true, 1, 0, 2, true) := by
  native_decide

private def hashIdentityFixtureScope : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .hashSample fixtureType .plain [109, 120, 120] [.constant 7] [] [] none none,
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] },
    { kind := .hashSample fixtureType .decomposed [109, 120, 120] [.constant 7] [] []
        (some (.constant 2)) (some (.constant 1)),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] }
  ]
  outputs := [("plain", { node := 1, port := 0 }), ("decomposed", { node := 2, port := 0 })]
  inputNames := ["key"]
}

private def hashIdentityFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .hashSample, arguments := [{ node := 0, port := 0 }] },
  { sourceNode := 2, rule := .hashSample, arguments := [{ node := 0, port := 0 }] }
] }

/-- Plain and decomposed modes of the same fully evaluated hash query share the target identity. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts hashIdentityFixtureScope
      hashIdentityFixtureDerivation [] [fixtureLayout]
    let plain ← matrixFactAt 2 facts { node := 1, port := 0 }
    let decomposed ← matrixFactAt 2 facts { node := 2, port := 0 }
    match decomposed.relations with
    | [.decomposition relation] => pure (plain.origin == relation.inputOrigin)
    | _ => pure false) = .ok true := by
  native_decide

private def trailingHashIdentityFixtureScope : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .constantInt 9, arguments := [], outputTypes := [.integer] },
    { kind := .hashSample fixtureType .plain [109, 120, 120] [.constant 7] [] [] none none,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .hashSample fixtureType .decomposed [109, 120, 120] [.constant 7] [] []
        (some (.constant 2)) (some (.constant 1)),
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.preimage fixtureType] }
  ]
  outputs := [("plain", { node := 2, port := 0 }), ("decomposed", { node := 3, port := 0 })]
  inputNames := ["key"]
}

private def trailingHashIdentityFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .constantInt, arguments := [] },
  { sourceNode := 2, rule := .hashSample,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .hashSample,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
] }

/-- A trailing integer operand participates in the plain/decomposed query identity in exact
argument order rather than being silently discarded. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts trailingHashIdentityFixtureScope
      trailingHashIdentityFixtureDerivation [] [fixtureLayout]
    let plain ← matrixFactAt 3 facts { node := 2, port := 0 }
    let decomposed ← matrixFactAt 3 facts { node := 3, port := 0 }
    match decomposed.relations with
    | [.decomposition relation] => pure (plain.origin == relation.inputOrigin)
    | _ => pure false) = .ok true := by
  native_decide

/-- Two stages hashing the same protocol key with the same complete query receive one semantic
hash origin even though their formal input and output wires are separately namespaced. -/
example : (do
    let input : ProtocolInputId := ⟨"shared-key"⟩
    let leftScope := ScopeTemplateKey.root (.workflowStage ⟨"left"⟩)
    let rightScope := ScopeTemplateKey.root (.workflowStage ⟨"right"⟩)
    let leftInput ← contractFact leftScope { node := 0, port := 0 } input
      (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let rightInput ← contractFact rightScope { node := 0, port := 0 } input
      (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let leftFacts ← evaluateScopeOperationalWithKey leftScope hashIdentityFixtureScope
      hashIdentityFixtureDerivation [] [fixtureLayout] [leftInput]
    let rightFacts ← evaluateScopeOperationalWithKey rightScope hashIdentityFixtureScope
      hashIdentityFixtureDerivation [] [fixtureLayout] [rightInput]
    let left ← matrixFactAt 2 leftFacts { node := 1, port := 0 }
    let right ← matrixFactAt 2 rightFacts { node := 1, port := 0 }
    pure (left.origin == right.origin)) = .ok true := by
  native_decide

private def scalarIntervalFixtureScope : Scope := {
  nodes := #[
    { kind := .constantInt (-2), arguments := [], outputTypes := [.integer] },
    { kind := .constantInt 3, arguments := [], outputTypes := [.integer] },
    { kind := .intBinary .multiply,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.integer] },
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .boolToInt, arguments := [{ node := 3, port := 0 }], outputTypes := [.integer] }
  ]
  outputs := [
    ("product", { node := 2, port := 0 }),
    ("bit", { node := 4, port := 0 })
  ]
  inputNames := []
}

private def scalarIntervalFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] },
  { sourceNode := 1, rule := .constantInt, arguments := [] },
  { sourceNode := 2, rule := .intBinary,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .constantBool, arguments := [] },
  { sourceNode := 4, rule := .boolToInt, arguments := [{ node := 3, port := 0 }] }
] }

/-- Scalar facts are derived from executable semantics rather than the former `[0, 0]`
fallback. Signed multiplication and Boolean conversion retain sound intervals. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scalarIntervalFixtureScope
      scalarIntervalFixtureDerivation [] []
    let product ← integerFactAt 5 facts { node := 2, port := 0 }
    let bit ← integerFactAt 5 facts { node := 4, port := 0 }
    pure (product.lower, product.upper, bit.lower, bit.upper)) = .ok (-6, -6, 0, 1) := by
  native_decide

private def malformedScalarOutputScope : Scope := {
  nodes := #[
    { kind := .constantInt 1, arguments := [], outputTypes := [.boolean] }
  ]
  outputs := []
  inputNames := []
}

private def malformedScalarOutputDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] }
] }

/-- A derivation cannot disguise an integer producer as a Boolean output. -/
example : (match evaluateScopeOperationalWithLayouts malformedScalarOutputScope
    malformedScalarOutputDerivation [] [] with
  | .error (.outputTypeMismatch 0) => true
  | _ => false) = true := by
  native_decide

private def negativeBitScope : Scope := {
  nodes := #[
    { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
    { kind := .bitExtract (.constant (-1)), arguments := [{ node := 0, port := 0 }],
      outputTypes := [.boolean] }
  ]
  outputs := []
  inputNames := []
}

private def negativeBitDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] },
  { sourceNode := 1, rule := .bitExtract, arguments := [{ node := 0, port := 0 }] }
] }

/-- A negative bit position is rejected rather than coerced to a natural number. -/
example : (match evaluateScopeOperationalWithLayouts negativeBitScope negativeBitDerivation [] [] with
  | .error (.invalidCount 1 (-1)) => true
  | _ => false) = true := by
  native_decide

private def scalarTypeMismatchScope : Scope := {
  nodes := #[
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .boolToInt, arguments := [{ node := 0, port := 0 }], outputTypes := [.integer] }
  ]
  outputs := []
  inputNames := []
}

private def scalarTypeMismatchDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 1, rule := .boolToInt, arguments := [{ node := 0, port := 0 }] }
] }

/-- Scalar transfer rules reject operands of a different executable wire type. -/
example : (match evaluateScopeOperationalWithLayouts scalarTypeMismatchScope
    scalarTypeMismatchDerivation [] [] with
  | .error (.operandNotBoolean 1 { node := 0, port := 0 }) => true
  | _ => false) = true := by
  native_decide

private def selectRangeMismatchScope : Scope := {
  nodes := #[
    { kind := .constantInt 2, arguments := [], outputTypes := [.integer] },
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .identityMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .select,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 },
        { node := 2, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := []
  inputNames := []
}

private def selectRangeMismatchDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] },
  { sourceNode := 1, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 2, rule := .identityMatrix, arguments := [] },
  { sourceNode := 3, rule := .select,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 },
      { node := 2, port := 0 }] }
] }

/-- A dynamic selector must be proved inside the executable branch range. -/
example : (match evaluateScopeOperationalWithLayouts selectRangeMismatchScope
    selectRangeMismatchDerivation [] [] with
  | .error (.invalidCount 3 2) => true
  | _ => false) = true := by
  native_decide

private def crtMetadataMismatchScope : Scope := {
  nodes := #[
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .crtRecompose [.constant 2, .constant 3] [.constant 1, .constant 1],
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := []
  inputNames := []
}

private def crtMetadataMismatchDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 1, rule := .crtRecompose, arguments := [{ node := 0, port := 0 }] }
] }

/-- CRT metadata is positional and must have exactly one entry for every operand. -/
example : (match evaluateScopeOperationalWithLayouts crtMetadataMismatchScope
    crtMetadataMismatchDerivation [] [] with
  | .error (.unsupportedOutputArity 1 1) => true
  | _ => false) = true := by
  native_decide

private def packedPolynomialInputMismatchScope : Scope := {
  nodes := #[
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .packPolynomialCoefficients fixtureType (.constant 5),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := []
  inputNames := []
}

private def packedPolynomialInputMismatchDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantBool, arguments := [] },
  { sourceNode := 1, rule := .packPolynomialCoefficients,
    arguments := [{ node := 0, port := 0 }] }
] }

/-- Polynomial reconstruction accepts only the exact Boolean family shape required by the IR. -/
example : (match evaluateScopeOperationalWithLayouts packedPolynomialInputMismatchScope
    packedPolynomialInputMismatchDerivation [] [] with
  | .error (.loopInputModeMismatch 1 0) => true
  | _ => false) = true := by
  native_decide

private def loopHashBody : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .hashSample fixtureType .plain [109, 120, 120] [.loopIndex 0] [] [] none none,
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := ["key"]
}

private def loopHashProgram : Prog := {
  root := {
    nodes := #[
      { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
      { kind := .parallelLoop "body" (.constant 2) 0 [] [.broadcast],
        arguments := [{ node := 0, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .familyGetStatic (.constant 0), arguments := [{ node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] },
      { kind := .familyGetStatic (.constant 1), arguments := [{ node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("first", { node := 2, port := 0 }), ("second", { node := 3, port := 0 })]
    inputNames := ["key"]
  }
  definitions := [("body", loopHashBody)]
}

private def loopHashDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .parallelLoop, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 2, rule := .familyGetStatic, arguments := [{ node := 1, port := 0 }] },
    { sourceNode := 3, rule := .familyGetStatic, arguments := [{ node := 1, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .hashSample, arguments := [{ node := 0, port := 0 }] }
  ] })]
}

/-- Static extraction instantiates the loop-dependent hash query, so two lanes cannot acquire the
same deterministic source identity merely because the body was analyzed once. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts loopHashProgram loopHashDerivation [] []
    let first ← matrixFactAt 3 facts { node := 2, port := 0 }
    let second ← matrixFactAt 3 facts { node := 3, port := 0 }
    pure (first.origin != second.origin)) = .ok true := by
  native_decide

private def aliasedHashBody : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .hashSample fixtureType .plain [109, 120, 120] [.parameter "tag"] [] [] none none,
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := ["key"]
}

private def aliasedLoopBody : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .subgraphCall "hash" [("tag", .loopIndex 0)],
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := ["key"]
}

private def aliasedLoopHashProgram : Prog := {
  root := loopHashProgram.root
  definitions := [("body", aliasedLoopBody), ("hash", aliasedHashBody)]
}

private def aliasedLoopHashDerivation : ProgramDerivation := {
  root := loopHashDerivation.root
  definitions := [
    ("body", { steps := #[
      { sourceNode := 0, rule := .input, arguments := [] },
      { sourceNode := 1, rule := .subgraphCall, arguments := [{ node := 0, port := 0 }] }
    ] }),
    ("hash", { steps := #[
      { sourceNode := 0, rule := .input, arguments := [] },
      { sourceNode := 1, rule := .hashSample, arguments := [{ node := 0, port := 0 }] }
    ] })
  ]
}

/-- A child parameter bound to an enclosing loop index retains that binding frame in the hash
identity. Flattening the child environment at template index zero would make these origins equal. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts aliasedLoopHashProgram
      aliasedLoopHashDerivation [] []
    let first ← matrixFactAt 3 facts { node := 2, port := 0 }
    let second ← matrixFactAt 3 facts { node := 3, port := 0 }
    pure (first.origin != second.origin)) = .ok true := by
  native_decide

private def relationFixtureScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gadgetMatrix fixtureType (.constant 2), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 3, port := 0 })], inputNames := []
}

private def relationFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .gadgetMatrix, arguments := [] },
  { sourceNode := 2, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] },
  { sourceNode := 3, rule := .matrixMultiplyRelation { node := 2, port := 0 },
    arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }] }
] }

example : (do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    matrixMaximum 3 { node := 3, port := 0 } facts) = .ok 3 := by
  native_decide

private def wrongRelationFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .gadgetMatrix, arguments := [] },
  { sourceNode := 2, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] },
  { sourceNode := 3, rule := .matrixMultiplyRelation { node := 1, port := 0 },
    arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }] }
] }

example : checkScopeDerivation relationFixtureScope wrongRelationFixtureDerivation =
    .error (.invalidRelationOperand 3 { node := 1, port := 0 }) := by
  native_decide

private def childRelationScope : Scope := {
  nodes := #[
    { kind := .input "target", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] }
  ],
  outputs := [("preimage", { node := 1, port := 0 })], inputNames := ["target"]
}

private def childRelationDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] }
] }

private def subgraphRelationProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gadgetMatrix fixtureType (.constant 2), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .subgraphCall "decompose" [], arguments := [{ node := 0, port := 0 }],
        outputTypes := [.preimage fixtureType] },
      { kind := .matrixMultiply,
        arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ],
    outputs := [("result", { node := 3, port := 0 })], inputNames := []
  }
  definitions := [("decompose", childRelationScope)]
}

private def subgraphRelationDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gadgetMatrix, arguments := [] },
    { sourceNode := 2, rule := .subgraphCall, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 3, rule := .matrixMultiplyRelation { node := 2, port := 0 },
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }] }
  ] }
  definitions := [("decompose", childRelationDerivation)]
}

example : (do
    let facts ← evaluateProgramOperationalWithLayouts subgraphRelationProgram
      subgraphRelationDerivation [] [fixtureLayout]
    matrixMaximum 3 { node := 3, port := 0 } facts) = .ok 3 := by
  native_decide

private def distinctCallIdentityProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .subgraphCall "decompose" [], arguments := [{ node := 0, port := 0 }],
        outputTypes := [.preimage fixtureType] },
      { kind := .subgraphCall "decompose" [], arguments := [{ node := 0, port := 0 }],
        outputTypes := [.preimage fixtureType] }
    ]
    outputs := [("left", { node := 1, port := 0 }), ("right", { node := 2, port := 0 })]
    inputNames := []
  }
  definitions := [("decompose", childRelationScope)]
}

private def distinctCallIdentityDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .subgraphCall, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 2, rule := .subgraphCall, arguments := [{ node := 0, port := 0 }] }
  ] }
  definitions := [("decompose", childRelationDerivation)]
}

/-- Equal local node/port numbers in two call instances are not the same sampled/derived event. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts distinctCallIdentityProgram
      distinctCallIdentityDerivation [] [fixtureLayout]
    let left ← matrixFactAt 2 facts { node := 1, port := 0 }
    let right ← matrixFactAt 2 facts { node := 2, port := 0 }
    pure (left.origin != right.origin)) = .ok true := by
  native_decide

private def packedFamilyFixtureScope : Scope := {
  nodes := relationFixtureScope.nodes ++ #[
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] },
    { kind := .familyPack,
      arguments := [{ node := 2, port := 0 }, { node := 4, port := 0 }],
      outputTypes := [.indexedFamily (.preimage fixtureType) (.constant 2)] },
    { kind := .familyGetStatic (.constant 0), arguments := [{ node := 5, port := 0 }],
      outputTypes := [.preimage fixtureType] },
    { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
    { kind := .familyGetDynamic,
      arguments := [{ node := 5, port := 0 }, { node := 7, port := 0 }],
      outputTypes := [.preimage fixtureType] },
    { kind := .familyPack,
      arguments := [{ node := 1, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
    { kind := .familyGetDynamic,
      arguments := [{ node := 9, port := 0 }, { node := 7, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 10, port := 0 }, { node := 8, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("static", { node := 6, port := 0 }), ("dynamic", { node := 8, port := 0 }),
    ("rewritten", { node := 11, port := 0 })]
  inputNames := []
}

private def packedFamilyFixtureDerivation : ScopeDerivation := {
  steps := relationFixtureDerivation.steps ++ #[
    { sourceNode := 4, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 5, rule := .familyPack,
      arguments := [{ node := 2, port := 0 }, { node := 4, port := 0 }] },
    { sourceNode := 6, rule := .familyGetStatic, arguments := [{ node := 5, port := 0 }] },
    { sourceNode := 7, rule := .constantInt, arguments := [] },
    { sourceNode := 8, rule := .familyGetDynamic,
      arguments := [{ node := 5, port := 0 }, { node := 7, port := 0 }] },
    { sourceNode := 9, rule := .familyPack,
      arguments := [{ node := 1, port := 0 }, { node := 1, port := 0 }] },
    { sourceNode := 10, rule := .familyGetDynamic,
      arguments := [{ node := 9, port := 0 }, { node := 7, port := 0 }] },
    { sourceNode := 11, rule := .matrixMultiplyRelation { node := 8, port := 0 },
      arguments := [{ node := 10, port := 0 }, { node := 8, port := 0 }] }
  ]
}

/-- Dynamic extraction keeps relation-bearing branches aligned under one compact expression
selection. Relation-consuming multiplication then rewrites every branch independently and agrees
with the explicitly unrolled bound. -/
private def exactRelationSelectionFixtureResult : Except OperationalError Bool := do
    let facts ← evaluateScopeOperationalWithLayouts packedFamilyFixtureScope
      packedFamilyFixtureDerivation [] [fixtureLayout]
    let dynamicOk ← match ← lookupFact 8 facts { node := 8, port := 0 } with
    | .matrixExpr root => match facts.arena.get? root with
        | some { node := .select _ (.exact branches), .. } => match branches[0]? with
            | some firstId => match facts.arena.get? firstId with
                | some { node := .concrete first, .. } =>
                    pure (branches.size == 2 && !first.relations.isEmpty)
                | _ => pure false
            | none => pure false
        | _ => pure false
    | _ => pure false
    let (rewrittenBounds, rewrittenRoot) ←
        match ← lookupFact 11 facts { node := 11, port := 0 } with
    | .matrixExpr root => match facts.arena.get? root with
        | some { node := .select _ (.exact branches), .. } => do
            let bounds ← branches.toList.mapM fun branch => match facts.arena.get? branch with
              | some { node := .concrete fact, .. } =>
                  fact.totalHardBound.evaluateWithStates [] []
              | _ => throw (OperationalError.unsupportedOperationalExpr branch)
            pure (bounds, root)
        | _ => throw (OperationalError.unsupportedOperationalExpr root)
    | _ => throw (OperationalError.operandNotMatrix 11 { node := 11, port := 0 })
    let representative : OperationalMatrixFact := {
      subject := { node := 20, port := 0 }
      origin := .value temporaryScope { node := 20, port := 0 }
      matrixType := fixtureType
      matrixParams := fixtureParams
      totalHardBound := .closedInt (.constant 7)
    }
    let summary := selectedMatrixSummary #[representative]
    let (envelopeArena, representativeId) :=
      ({} : OperationalExprArena).pushConcrete representative
    let envelopeSelection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 21, port := 0 }
    }
    let (envelopeArena, envelopeRoot) ← envelopeArena.pushSelect envelopeSelection
      (.schemaEnvelope 30720 representativeId summary)
    let (envelopeBound, _) ← evaluateOperationalExprBound envelopeArena [] envelopeRoot
      (OperationalExprEvaluationState.empty envelopeArena)
    let staleRepresentative := { representative with
      totalHardBound := OperationalBoundExpr.closedInt (.constant 8) }
    let (staleArena, staleId) := ({} : OperationalExprArena).pushConcrete staleRepresentative
    let staleRejected := match staleArena.pushSelect envelopeSelection
        (.schemaEnvelope 2 staleId summary) with
      | .error (.unsupportedOperationalExpr _) => true
      | _ => false
    let report ← decoderNoiseCheckReportForFact [] facts.arena (.matrixExpr rewrittenRoot) [] 2 25
    pure (dynamicOk && rewrittenBounds == [3, 3] && envelopeArena.nodes.size == 2 &&
      envelopeBound == 7 && staleRejected &&
      report.obligations == [.decoderThreshold 2 25 3])

example : exactRelationSelectionFixtureResult = .ok true := by
  native_decide

private def selectFixtureScope : Scope := {
  nodes := #[
    { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .select, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 },
      { node := 2, port := 0 }], outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 3, port := 0 })], inputNames := []
}

private def selectFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .gaussianSample, arguments := [] },
  { sourceNode := 3, rule := .select, arguments := [{ node := 0, port := 0 },
    { node := 1, port := 0 }, { node := 2, port := 0 }] }
] }

example : (do
    let facts ← evaluateScopeOperationalWithLayouts selectFixtureScope selectFixtureDerivation [] []
    matrixMaximum 3 { node := 3, port := 0 } facts) = .ok 5 := by
  native_decide

private def loopBoundBody : Scope := {
  nodes := #[{
    kind := .gaussianSample fixtureType (.parameter "lane_bound")
    arguments := []
    outputTypes := [.matrix fixtureType]
  }]
  outputs := [("result", { node := 0, port := 0 })]
  inputNames := []
}

private def loopBoundProgram : Prog := {
  root := {
    nodes := #[{
      kind := .parallelLoop "body" (.constant 4) 0
        [("lane_bound", .add (.loopIndex 0) (.constant 1))] []
      arguments := []
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 4)]
    }, {
      kind := .familyGetStatic (.constant 2)
      arguments := [{ node := 0, port := 0 }]
      outputTypes := [.matrix fixtureType]
    }]
    outputs := [("results", { node := 0, port := 0 }), ("selected", { node := 1, port := 0 })]
    inputNames := []
  }
  definitions := [("body", loopBoundBody)]
}

private def loopBoundDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .parallelLoop, arguments := [] },
    { sourceNode := 1, rule := .familyGetStatic, arguments := [{ node := 0, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] }
  ] })]
}

/-- A loop-dependent child parameter is evaluated numerically over all four indices while the
body graph itself is evaluated once. The resulting uniform family stores the exact maximum 4. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts loopBoundProgram loopBoundDerivation [] []
    match ← lookupFact 1 facts { node := 0, port := 0 } with
    | .familyUniform _ _ (.matrix fact) _ => fact.totalHardBound.evaluate [] #[]
    | _ => throw (OperationalError.loopInputModeMismatch 0 0)) = .ok 4 := by
  native_decide

example : (do
    let facts ← evaluateProgramOperationalWithLayouts loopBoundProgram loopBoundDerivation [] []
    matrixMaximum 2 { node := 1, port := 0 } facts) = .ok 3 := by
  native_decide

private def sequentialRelationBody : Scope := {
  nodes := #[
    { kind := .input "target", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .input "public", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 3, port := 0 })]
  inputNames := ["target", "public"]
}

private def sequentialRelationProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gadgetMatrix fixtureType (.constant 2), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .sequentialLoop "body" (.constant 3) 0 [] 1,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("result", { node := 2, port := 0 })]
    inputNames := []
  }
  definitions := [("body", sequentialRelationBody)]
}

private def sequentialRelationDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gadgetMatrix, arguments := [] },
    { sourceNode := 2, rule := .sequentialLoop,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .input, arguments := [] },
    { sourceNode := 2, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 3, rule := .matrixMultiplyRelation { node := 2, port := 0 },
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }] }
  ] })]
}

/-- A relation may depend on the previous carried bound when it is created and consumed inside
one body execution. Only the resulting relation-free target fact becomes the next carried state. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts sequentialRelationProgram
      sequentialRelationDerivation [] [fixtureLayout]
    matrixMaximum 2 { node := 2, port := 0 } facts) = .ok 2 := by
  native_decide

private def relationCarryBody : Scope := {
  nodes := #[{ kind := .input "carried", arguments := [], outputTypes := [.preimage fixtureType] }]
  outputs := [("result", { node := 0, port := 0 })]
  inputNames := ["carried"]
}

private def relationCarryProgram : Prog := {
  root := {
    nodes := relationFixtureScope.nodes.take 3 ++ #[{
      kind := .sequentialLoop "body" (.constant 1) 0 [] 1
      arguments := [{ node := 2, port := 0 }]
      outputTypes := [.preimage fixtureType]
    }]
    outputs := [("result", { node := 3, port := 0 })]
    inputNames := []
  }
  definitions := [("body", relationCarryBody)]
}

private def relationCarryDerivation : ProgramDerivation := {
  root := { steps := relationFixtureDerivation.steps.take 3 ++ #[{
    sourceNode := 3
    rule := .sequentialLoop
    arguments := [{ node := 2, port := 0 }]
  }] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] }
  ] })]
}

/-- Relations are body-local tokens; carrying one across iterations rejects before abstraction. -/
example : (match evaluateProgramOperationalWithLayouts relationCarryProgram relationCarryDerivation
    [] [fixtureLayout] with
  | .error (.relationBearingCarriedValue (.root (.standalone 0)) 3 0) => true
  | _ => false) = true := by
  native_decide

private def simultaneousRecurrence (slot : Nat) : OperationalBoundExpr :=
  .recurrence 2 [
      .closedInt (.constant 2),
      .closedInt (.constant 5)
    ] [
      .add (.previous (.matrixMaximum 0 0)) (.closedInt (.constant 3)),
      .add (.previous (.matrixMaximum 0 1)) (.previous (.matrixMaximum 0 0))
    ] slot

/-- All carried slots read the previous state. The second slot must not observe the first slot's
new value from the same iteration. -/
example : (simultaneousRecurrence 0).evaluate [] #[] = .ok 8 := by
  native_decide

example : (simultaneousRecurrence 1).evaluate [] #[] = .ok 12 := by
  native_decide

private def nestedRecurrence : OperationalBoundExpr :=
  .recurrence 2 [.closedInt (.constant 2)] [
    .recurrence 2 [.previous (.matrixMaximum 0 0)] [
      .add (.previous (.matrixMaximum 0 0)) (.previous (.matrixMaximum 1 0))
    ] 0
  ] 0

/-- The inner depth zero denotes the inner state and depth one denotes the enclosing state. -/
example : nestedRecurrence.evaluate [] #[] = .ok 18 := by
  native_decide

/-- A zero-count recurrence returns the initial slot without evaluating its transition. -/
example : (.recurrence 0 [.closedInt (.constant 7)]
    [.previous (.matrixMaximum 0 99)] 0 : OperationalBoundExpr).evaluate [] #[] = .ok 7 := by
  native_decide

/-- A typed carried placeholder has no meaning outside recurrence evaluation. -/
example : (.previous (.matrixMaximum 0 0) : OperationalBoundExpr).evaluate [] #[] =
    .error (.invalidPreviousPath (.matrixMaximum 0 0)) := by
  native_decide

private def sampledLoopIdentity : PublicMatrixIdentity :=
  .sampledTrapdoor (.parallelBody (.root (.standalone 0)) 4) { node := 2, port := 0 }

/-- Independent samples produced at one body wire receive distinct concrete loop identities. -/
example : instantiatePublicIdentityLoopIndex 0 0 sampledLoopIdentity !=
    instantiatePublicIdentityLoopIndex 0 1 sampledLoopIdentity := by
  native_decide

/-- Nested loop instantiation retains both concrete selections. -/
example : instantiatePublicIdentityLoopIndex 1 3
    (instantiatePublicIdentityLoopIndex 0 2 sampledLoopIdentity) =
    .loopInstance 0 2 (.loopInstance 1 3 sampledLoopIdentity) := by
  native_decide

/-- Deterministic gadget matrices are not spuriously made lane-local. -/
example : instantiatePublicIdentityLoopIndex 0 7
    (.gadget "fixture" fixtureParams 1 2 false 3) =
    (.gadget "fixture" fixtureParams 1 2 false 3) := by
  native_decide

private def mismatchedFixtureType : MatrixTypeExpr :=
  { fixtureType with rows := .constant 2 }

/-- A frozen leaf cannot claim an output matrix type different from the type it executes. -/
example : (match evaluateScopeOperationalWithLayouts {
    nodes := #[{
      kind := .zeroMatrix fixtureType
      arguments := []
      outputTypes := [.matrix mismatchedFixtureType]
    }]
    outputs := [("result", { node := 0, port := 0 })]
    inputNames := []
  } {
    steps := #[{ sourceNode := 0, rule := .zeroMatrix, arguments := [] }]
  } [] [] with
  | .error (.outputTypeMismatch 0) => true
  | _ => false) = true := by
  native_decide

/-- Arithmetic operands must have the exact declared output matrix type. -/
example : (match evaluateScopeOperationalWithLayouts {
    nodes := #[
      { kind := .zeroMatrix fixtureType, arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .zeroMatrix mismatchedFixtureType, arguments := [],
        outputTypes := [.matrix mismatchedFixtureType] },
      { kind := .matrixAdd,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("result", { node := 2, port := 0 })]
    inputNames := []
  } {
    steps := #[
      { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
      { sourceNode := 1, rule := .zeroMatrix, arguments := [] },
      { sourceNode := 2, rule := .matrixAdd,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
    ]
  } [] [] with
  | .error (.outputTypeMismatch 2) => true
  | _ => false) = true := by
  native_decide

/-- Output arity is checked before any operational fact is constructed. -/
example : (match evaluateScopeOperationalWithLayouts {
    nodes := #[{
      kind := .zeroMatrix fixtureType
      arguments := []
      outputCount := 2
      outputTypes := [.matrix fixtureType]
    }]
    outputs := [("result", { node := 0, port := 0 })]
    inputNames := []
  } {
    steps := #[{ sourceNode := 0, rule := .zeroMatrix, arguments := [] }]
  } [] [] with
  | .error (.unsupportedOutputArity 0 2) => true
  | _ => false) = true := by
  native_decide

/-- The generic decoder obligation uses the exact strict product inequality. At noise three and
plaintext modulus two, ciphertext modulus thirteen passes while the boundary value twelve fails. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let residual ← matrixFactAt 1 facts { node := 1, port := 0 }
    let accepted ← decoderNoiseCheckReport [] residual [] 2 25
    let rejected ← decoderNoiseCheckReport [] residual [] 2 24
    pure (accepted.accepted, accepted.rejection, rejected.accepted, rejected.rejection)) =
    .ok (true, none, false, some (.decoderThresholdNotMet 2 24 6)) := by
  native_decide

/-- An invalid plaintext modulus is rejected by the generic report rather than interpreted by an
application-specific checker. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let residual ← matrixFactAt 1 facts { node := 1, port := 0 }
    let report ← decoderNoiseCheckReport [] residual [] 1 100
    pure (report.accepted, report.rejection)) =
    .ok (false, some (.invalidPlaintextModulus 1)) := by
  native_decide

/-- Packed residual families inspect every member rather than using a representative lane. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let first ← lookupFact 2 facts { node := 0, port := 0 }
    let second ← lookupFact 2 facts { node := 1, port := 0 }
    let report ← decoderNoiseCheckReportForFact [] {} (packedFacts [first, second]) [] 2 25
    pure report.obligations) = .ok [.decoderThreshold 2 25 6] := by
  native_decide

/-- A compact selected residual computes each complete branch bound and then takes the maximum. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let first ← matrixFactAt 2 facts { node := 0, port := 0 }
    let second ← matrixFactAt 2 facts { node := 1, port := 0 }
    let selection := OperationalValueOrigin.local temporaryScope { node := 9, port := 0 }
    let report ← decoderNoiseCheckReportForFact [] {}
      (.selectedMatrices (selectedMatrixFamily selection #[first, second])) [] 2 25
    pure report.obligations) = .ok [.decoderThreshold 2 25 6] := by
  native_decide

/-- Relation-free branches whose complete operational schemas differ only in producer identity
collapse to one selection-namespaced envelope.  The decoder consumes the same complete bound. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let first ← matrixFactAt 2 facts { node := 0, port := 0 }
    let branch (index : Nat) : OperationalMatrixFact := {
      first with
      origin := .loopInstance 0 index first.origin
      polynomial := mapOperationalPolynomial
        (fun origin => .loopInstance 0 index origin)
        (fun identity => .loopInstance 0 index identity)
        (fun origin => .loopInstance 0 index origin)
        id id first.polynomial
    }
    let selection := OperationalValueOrigin.local temporaryScope { node := 9, port := 0 }
    let compressed ← compressUniformSelectedMatrices 10 { node := 10, port := 0 }
      selection #[branch 0, branch 1]
    let report ← decoderNoiseCheckReportForFact [] {} compressed [] 2 25
    pure (match compressed with | .matrix _ => true | _ => false, report.obligations)) =
    .ok (true, [.decoderThreshold 2 25 3]) := by
  native_decide

/-- Equal shapes are insufficient for uniform collapse: differing complete branch bounds retain the
explicit selected family so the decoder still takes the branch-wise maximum. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let first ← matrixFactAt 2 facts { node := 0, port := 0 }
    let second ← matrixFactAt 2 facts { node := 1, port := 0 }
    let selection := OperationalValueOrigin.local temporaryScope { node := 9, port := 0 }
    let retained ← compressUniformSelectedMatrices 10 { node := 10, port := 0 }
      selection #[first, second]
    let report ← decoderNoiseCheckReportForFact [] {} retained [] 2 25
    pure (match retained with | .selectedMatrices _ => true | _ => false,
      report.obligations)) = .ok (true, [.decoderThreshold 2 25 6]) := by
  native_decide

/-- Ordinary inputs broadcast across a selected input, while two different selections never zip. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let first ← matrixFactAt 2 facts { node := 0, port := 0 }
    let second ← matrixFactAt 2 facts { node := 1, port := 0 }
    let leftSelection := OperationalValueOrigin.local temporaryScope { node := 9, port := 0 }
    let rightSelection := OperationalValueOrigin.local temporaryScope { node := 10, port := 0 }
    let aligned ← alignSelectedMatrixInputs 11 [
      ({ node := 0, port := 0 },
        .selectedMatrices (selectedMatrixFamily leftSelection #[first, second])),
      ({ node := 1, port := 0 }, .matrix first)]
    let mismatch := alignSelectedMatrixInputs 12 [
      ({ node := 0, port := 0 },
        .selectedMatrices (selectedMatrixFamily leftSelection #[first, second])),
      ({ node := 1, port := 0 },
        .selectedMatrices (selectedMatrixFamily rightSelection #[first, second]))]
    pure (aligned.selection == some leftSelection && aligned.rows.size == 2 &&
      aligned.rows.all (·.length == 2) &&
      (match mismatch with
        | .error (.selectedFamilyOperationUnsupported 12) => true
        | _ => false))) = .ok true := by
  native_decide

/-- ZipOffset selects the exact corresponding relation-bearing packed branch. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts sharedPreimageBaseScope
      sharedPreimageBaseDerivation [] [fixtureLayout]
    let first ← lookupFact 5 facts { node := 3, port := 0 }
    let second ← lookupFact 5 facts { node := 4, port := 0 }
    let selected ← loopTemplateArgumentFact 20 0 1 (.zipOffset 1)
      (packedFacts [first, second])
    match selected, second with
    | .selectedMatrices family, .matrix expected => match family.branches[0]? with
        | some actual =>
            let actual : OperationalMatrixFact := actual
            pure (family.branches.size == 1 && actual.origin == expected.origin)
        | none => pure false
    | _, _ => pure false) = Except.ok true := by
  native_decide

/-- A checked uniform family evaluates its element template once, independently of its count. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let residual ← lookupFact 2 facts { node := 1, port := 0 }
    let family := OperationalFact.familyUniform fixtureFamilyBinder none residual 100
    let report ← decoderNoiseCheckReportForFact [] {} family [] 2 25
    pure report.obligations) = .ok [.decoderThreshold 2 25 6] := by
  native_decide

/-- Empty residual families are rejected instead of being assigned a zero bound. -/
example : (match decoderNoiseCheckReportForFact [] {} (.familyPacked #[] 0 none) [] 2 25 with
    | .error (.invalidCount 0 0) => true
    | _ => false) = true := by
  native_decide

/-! The expression-arena fixtures use `decide`, not `native_decide`: request-local IDs and memo
statistics are ordinary checker data and do not enlarge the trusted evaluation base. -/

private def operationalExprFixtureFact (node : Nat) (bound : Int) : OperationalMatrixFact := {
  subject := { node, port := 0 }
  origin := .value temporaryScope { node, port := 0 }
  matrixType := fixtureType
  matrixParams := fixtureParams
  totalHardBound := .closedInt (.constant bound)
}

/-- A selected expression evaluates each complete branch once, takes their maximum, and then hits
the O(1) ID-keyed memo entry when the same root is requested again. -/
example : (do
    let first := operationalExprFixtureFact 0 3
    let second := operationalExprFixtureFact 1 6
    let (arena, firstId) := ({} : OperationalExprArena).pushConcrete first
    let (arena, secondId) := arena.pushConcrete second
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 9, port := 0 }
    }
    let (arena, selectedId) ← arena.pushSelect selection
      (.exact #[firstId, secondId])
    let initial := OperationalExprEvaluationState.empty arena
    let (firstBound, state) ← evaluateOperationalExprBound arena [] selectedId initial
    let (secondBound, state) ← evaluateOperationalExprBound arena [] selectedId state
    pure (arena.nodes.size, firstBound, secondBound, state.stats)) =
    .ok (3, 6, 6, { evaluations := 3, memoHits := 1, memoMisses := 3 }) := by
  simp [operationalExprFixtureFact, fixtureType, OperationalExprArena.pushConcrete,
    OperationalExprArena.push, OperationalExprArena.pushSelect,
    OperationalExprArena.checkedType, OperationalExprArena.get?,
    OperationalExprEvaluationState.empty, evaluateOperationalExprBound,
    evaluateOperationalExprBoundWithFuel]
  rfl

/-- Exact equal-branch reduction reuses the existing expression ID and allocates no select node. -/
example : (do
    let first := operationalExprFixtureFact 0 3
    let (arena, firstId) := ({} : OperationalExprArena).pushConcrete first
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 9, port := 0 }
    }
    let (arena, selectedId) ← arena.pushSelect selection (.exact #[firstId, firstId])
    pure (arena.nodes.size, selectedId == firstId)) = .ok (1, true) := by
  simp [operationalExprFixtureFact, fixtureType, OperationalExprArena.pushConcrete,
    OperationalExprArena.push, OperationalExprArena.pushSelect,
    OperationalExprArena.checkedType, OperationalExprArena.get?]
  rfl

/-- Static exact selection distinguishes an invalid index from the intentionally unavailable
schema-envelope lookup. -/
example : (OperationalSelectionBranches.exact #[4, 7]).staticBranch 2 =
    .error (.invalidCount 0 2) := by
  decide

end Mxx.Certificate
