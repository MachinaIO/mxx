import Mxx.Certificate.Derivation
import Mxx.Certificate.IndexedFacts
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

def ProgramInstanceKey.toGatherProgramInstanceKey : ProgramInstanceKey → GatherProgramInstanceKey
  | .temporary => .temporary
  | .workflowStage stage => .workflowStage stage
  | .ideal => .ideal
  | .requirement index => .requirement index
  | .standalone ordinal => .standalone ordinal

def ScopeTemplateKey.toGatherScopeTemplateKey : ScopeTemplateKey → GatherScopeTemplateKey
  | .root program => .root program.toGatherProgramInstanceKey
  | .callBody parent node => .callBody parent.toGatherScopeTemplateKey node
  | .parallelBody parent node => .parallelBody parent.toGatherScopeTemplateKey node
  | .sequentialBody parent node => .sequentialBody parent.toGatherScopeTemplateKey node

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
  | protocolFamilyElement (input : ProtocolInputId) (index : IndexExpr)
  | loopInstance (slot : Nat) (index : IndexExpr) (source : OperationalValueOrigin)
  | indexed
      (binder : FamilyTemplateBinder)
      (expression : IndexExpr)
      (source : OperationalValueOrigin)
  deriving BEq, DecidableEq, Repr

structure DynamicSelectionIdentity where
  index : OperationalValueOrigin
  expression : IndexExpr
  deriving BEq, DecidableEq, Repr

inductive SelectionDomainKind where
  | loopLane
  | protocolSelection
  deriving BEq, DecidableEq, Repr

def scopeIsLoopLane : ScopeTemplateKey → Bool
  | .parallelBody .. => true
  | .callBody parent _ | .sequentialBody parent _ => scopeIsLoopLane parent
  | .root _ => false

def selectionDomainKind : OperationalValueOrigin → SelectionDomainKind
  | .loopInstance .. => .loopLane
  | .local scope _ => if scopeIsLoopLane scope then .loopLane else .protocolSelection
  | .indexed _ _ source => selectionDomainKind source
  | .protocolInput _ | .protocolFamilyElement _ _ => .protocolSelection

structure SelectionDomainKey where
  kind : SelectionDomainKind
  identity : DynamicSelectionIdentity
  count : Nat
  deriving BEq

/-- Request-local canonical identity for one mutually-exclusive selection domain.  `ordinal` is
the comparison key after full-key interning; the remaining fields make the canonical branch count
and provenance available without a second lookup or duplicated count. -/
structure SelectionDomainId where
  ordinal : Nat
  kind : SelectionDomainKind
  identity : DynamicSelectionIdentity
  count : Nat
  deriving DecidableEq, Repr

instance : BEq SelectionDomainId where
  beq left right := left.ordinal == right.ordinal

instance : Coe SelectionDomainId DynamicSelectionIdentity where
  coe domain := domain.identity

def SelectionDomainId.index (domain : SelectionDomainId) : OperationalValueOrigin :=
  domain.identity.index

class SelectionIdentityLike (α : Type) where
  identity : α → DynamicSelectionIdentity
  domainCount? : α → Option Nat

instance : SelectionIdentityLike DynamicSelectionIdentity where
  identity := id
  domainCount? _ := none

instance : SelectionIdentityLike SelectionDomainId where
  identity domain := domain.identity
  domainCount? domain := some domain.count

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
  | protocolFamilyElement (input : ProtocolInputId) (index : IndexExpr)
  | deterministicHash (query : DeterministicHashIdentity)
  | loopInstance (slot : Nat) (index : IndexExpr) (source : MatrixOriginIdentity)
  | indexed
      (binder : FamilyTemplateBinder)
      (expression : IndexExpr)
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
  | indexed
      (binder : FamilyTemplateBinder)
      (expression : IndexExpr)
      (source : PublicMatrixIdentity)
  | loopInstance (slot : Nat) (index : IndexExpr) (source : PublicMatrixIdentity)
  deriving BEq, DecidableEq, Repr

def temporaryScope : ScopeTemplateKey := .root .temporary

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
  | indexedArtifact (input : ProtocolInputId) (index : IndexExpr)
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

/-- Analysis-local IDs used only while normalizing one polynomial boundary.  Canonical factor and
product structures remain authoritative; IDs merely avoid repeating their comparison while
coefficients are accumulated. -/
abbrev OperationalFactorId := Nat
abbrev OperationalMonomialId := Nat

structure OperationalMonomialKey where
  factors : Array OperationalFactorId
  modes : Array OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq

structure OperationalFactorInterner where
  factors : Array OperationalFactorKey := #[]
  buckets : Std.HashMap UInt64 (Array OperationalFactorId) := {}
  deriving BEq

structure OperationalMonomialEntry where
  key : OperationalMonomialKey
  product : OperationalProductKey
  deriving BEq

structure OperationalMonomialInterner where
  monomials : Array OperationalMonomialEntry := #[]
  buckets : Std.HashMap UInt64 (Array OperationalMonomialId) := {}
  deriving BEq

/-- Request-local, analysis-only identity tables.  IDs are meaningful only while this arena is
threaded through one operational evaluation; canonical factor/product structures remain the
authority for equality and serialization. -/
structure OperationalInterningArena where
  factors : OperationalFactorInterner := {}
  monomials : OperationalMonomialInterner := {}
  factorHits : Nat := 0
  factorMisses : Nat := 0
  monomialHits : Nat := 0
  monomialMisses : Nat := 0
  deriving BEq

def mixOperationalFingerprint (state value : UInt64) : UInt64 :=
  state * 1099511628211 + value + 1469598103934665603

def operationalRoleFingerprint : OperationalFactorRole → UInt64
  | .bounded => 1
  | .large => 2

def operationalModeFingerprint : OperationalProductMode → UInt64
  | .ordinaryMatrixProduct => 1
  | .leftPolynomialScalarBroadcast => 2
  | .rightPolynomialScalarBroadcast => 3
  | .swappedRowVectorScalarProduct => 4

def operationalIntExprFingerprint : IntExpr → UInt64
  | .constant value => mixOperationalFingerprint 73 (UInt64.ofInt value)
  | .parameter name => mixOperationalFingerprint 79 (hash name)
  | .loopIndex slot => mixOperationalFingerprint 83 (UInt64.ofNat slot)
  | .add left right => mixOperationalFingerprint
      (mixOperationalFingerprint 89 (operationalIntExprFingerprint left))
      (operationalIntExprFingerprint right)
  | .subtract left right => mixOperationalFingerprint
      (mixOperationalFingerprint 97 (operationalIntExprFingerprint left))
      (operationalIntExprFingerprint right)
  | .multiply left right => mixOperationalFingerprint
      (mixOperationalFingerprint 101 (operationalIntExprFingerprint left))
      (operationalIntExprFingerprint right)
  | .divide left right => mixOperationalFingerprint
      (mixOperationalFingerprint 103 (operationalIntExprFingerprint left))
      (operationalIntExprFingerprint right)
  | .roundDivide left right => mixOperationalFingerprint
      (mixOperationalFingerprint 107 (operationalIntExprFingerprint left))
      (operationalIntExprFingerprint right)
  | .log2Ceil value => mixOperationalFingerprint 109 (operationalIntExprFingerprint value)

def operationalProgramFingerprint : ProgramInstanceKey → UInt64
  | .temporary => 1
  | .workflowStage stage => mixOperationalFingerprint 2 (hash stage.name)
  | .ideal => 3
  | .requirement index => mixOperationalFingerprint 4 (UInt64.ofNat index)
  | .standalone ordinal => mixOperationalFingerprint 5 (UInt64.ofNat ordinal)

def operationalScopeFingerprint : ScopeTemplateKey → UInt64
  | .root program => mixOperationalFingerprint 7 (operationalProgramFingerprint program)
  | .callBody parent node => mixOperationalFingerprint
      (mixOperationalFingerprint 11 (operationalScopeFingerprint parent)) (UInt64.ofNat node)
  | .parallelBody parent node => mixOperationalFingerprint
      (mixOperationalFingerprint 13 (operationalScopeFingerprint parent)) (UInt64.ofNat node)
  | .sequentialBody parent node => mixOperationalFingerprint
      (mixOperationalFingerprint 17 (operationalScopeFingerprint parent)) (UInt64.ofNat node)

def operationalBinderFingerprint (binder : FamilyTemplateBinder) : UInt64 :=
  mixOperationalFingerprint
    (mixOperationalFingerprint (operationalScopeFingerprint binder.owner)
      (UInt64.ofNat binder.producerNode))
    (UInt64.ofNat binder.binderSlot)

def operationalValueOriginFingerprint : OperationalValueOrigin → UInt64
  | .local scope wire => mixOperationalFingerprint
      (mixOperationalFingerprint (operationalScopeFingerprint scope) (UInt64.ofNat wire.node))
      (UInt64.ofNat wire.port)
  | .protocolInput input => mixOperationalFingerprint 19 (hash input.name)
  | .protocolFamilyElement input index =>
      mixOperationalFingerprint (mixOperationalFingerprint 23 (hash input.name))
        (hash (reprStr index))
  | .loopInstance slot index source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 29 (operationalValueOriginFingerprint source))
          (UInt64.ofNat slot))
        (hash (reprStr index))
  | .indexed binder expression source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 31 (operationalBinderFingerprint binder))
          (hash (reprStr expression)))
        (operationalValueOriginFingerprint source)
def operationalSelectionFingerprint (selection : DynamicSelectionIdentity) : UInt64 :=
  mixOperationalFingerprint (operationalValueOriginFingerprint selection.index)
    (hash (reprStr selection.expression))

def operationalMatrixOriginFingerprint : MatrixOriginIdentity → UInt64
  | .value scope wire => mixOperationalFingerprint
      (mixOperationalFingerprint (operationalScopeFingerprint scope) (UInt64.ofNat wire.node))
      (UInt64.ofNat wire.port)
  | .protocolInput input => mixOperationalFingerprint 37 (hash input.name)
  | .protocolFamilyElement input index =>
      mixOperationalFingerprint (mixOperationalFingerprint 41 (hash input.name))
        (hash (reprStr index))
  | .deterministicHash query =>
      let seed := mixOperationalFingerprint 43
        (operationalValueOriginFingerprint query.keyOrigin)
      let seed := query.tagPrefix.foldl
        (fun state byte => mixOperationalFingerprint state (UInt64.ofNat byte)) seed
      let seed := query.tagExpressions.foldl
        (fun state value => mixOperationalFingerprint state
          (operationalIntExprFingerprint value)) seed
      let seed := query.tagDecimalExpressions.foldl
        (fun state value => mixOperationalFingerprint state
          (operationalIntExprFingerprint value)) seed
      let seed := query.tagU64LeExpressions.foldl
        (fun state value => mixOperationalFingerprint state
          (operationalIntExprFingerprint value)) seed
      let seed := query.trailingIntegerOrigins.foldl
        (fun state origin => mixOperationalFingerprint state
          (operationalValueOriginFingerprint origin)) seed
      mixOperationalFingerprint seed
        (UInt64.ofNat (query.tagExpressions.length + query.tagDecimalExpressions.length +
          query.tagU64LeExpressions.length))
  | .loopInstance slot index source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 47 (operationalMatrixOriginFingerprint source))
          (UInt64.ofNat slot))
        (hash (reprStr index))
  | .indexed binder expression source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 53 (operationalBinderFingerprint binder))
          (hash (reprStr expression)))
        (operationalMatrixOriginFingerprint source)

def operationalPublicMatrixFingerprint : PublicMatrixIdentity → UInt64
  | .sampledTrapdoor scope wire => mixOperationalFingerprint
      (mixOperationalFingerprint
        (mixOperationalFingerprint 59 (operationalScopeFingerprint scope))
        (UInt64.ofNat wire.node))
      (UInt64.ofNat wire.port)
  | .gadget paramsId params inputRows base small digitCount =>
      let seed := mixOperationalFingerprint 61 (hash paramsId)
      let seed := mixOperationalFingerprint seed (UInt64.ofNat params.ringDimension)
      let seed := mixOperationalFingerprint seed (UInt64.ofNat inputRows)
      let seed := mixOperationalFingerprint seed (UInt64.ofInt base)
      let seed := mixOperationalFingerprint seed (if small then 1 else 0)
      mixOperationalFingerprint seed (UInt64.ofNat digitCount)
  | .indexed binder expression source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 67 (operationalBinderFingerprint binder))
          (hash (reprStr expression)))
        (operationalPublicMatrixFingerprint source)
  | .loopInstance slot index source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 71 (operationalPublicMatrixFingerprint source))
          (UInt64.ofNat slot))
        (hash (reprStr index))

def operationalPrimitiveFingerprint : OperationalPrimitiveIdentity → UInt64
  | .matrix identity => mixOperationalFingerprint 13 (operationalMatrixOriginFingerprint identity)
  | .publicMatrix identity => mixOperationalFingerprint 17
      (operationalPublicMatrixFingerprint identity)
  | .value identity => mixOperationalFingerprint 19
      (operationalValueOriginFingerprint identity)
  | .parameterScalar _ domains value => mixOperationalFingerprint
      (mixOperationalFingerprint 23 (UInt64.ofNat domains.length))
      (operationalIntExprFingerprint value)
  | .identityMatrix _ => 29
  | .indexedArtifact input index => mixOperationalFingerprint
      (mixOperationalFingerprint 31 (hash input.name)) (hash (reprStr index))
  | .recurrenceResult _ node path =>
      mixOperationalFingerprint (mixOperationalFingerprint 37 (UInt64.ofNat node))
        (UInt64.ofNat path)
  | .carriedInput path => mixOperationalFingerprint 41 (UInt64.ofNat path)

def operationalLeafFingerprint : OperationalFactorLeaf → UInt64
  | .primitive identity => operationalPrimitiveFingerprint identity
  | .boundedSummary _ _ => 2
  | .exactTransform _ _ => 3

/-- A deliberately compact collision-prone fingerprint.  It is only a bucket selector; complete
`OperationalFactorKey` equality below remains the semantic check. -/
def operationalFactorFingerprint (factor : OperationalFactorKey) : UInt64 :=
  let seed := mixOperationalFingerprint (operationalLeafFingerprint factor.leaf)
    (operationalRoleFingerprint factor.role)
  let seed := mixOperationalFingerprint seed (UInt64.ofNat factor.transforms.length)
  let seed := mixOperationalFingerprint seed (UInt64.ofNat factor.protections.length)
  mixOperationalFingerprint seed (UInt64.ofNat factor.relations.length)

def internOperationalFactor
    (arena : OperationalInterningArena)
    (factor : OperationalFactorKey) : OperationalInterningArena × OperationalFactorId :=
  let fingerprint := operationalFactorFingerprint factor
  let candidates := arena.factors.buckets.getD fingerprint #[]
  match candidates.find? fun candidate => arena.factors.factors[candidate]? == some factor with
  | some id => ({ arena with factorHits := arena.factorHits + 1 }, id)
  | none =>
      let id := arena.factors.factors.size
      ({ arena with
        factors := {
          factors := arena.factors.factors.push factor
          buckets := arena.factors.buckets.insert fingerprint (candidates.push id)
        }
        factorMisses := arena.factorMisses + 1
      }, id)

def operationalMonomialFingerprint (key : OperationalMonomialKey) : UInt64 :=
  let seed := key.factors.foldl
    (fun state factor => mixOperationalFingerprint state (UInt64.ofNat factor)) 17
  let seed := key.modes.foldl
    (fun state mode => mixOperationalFingerprint state (operationalModeFingerprint mode)) seed
  mixOperationalFingerprint seed (UInt64.ofNat key.factors.size)

def operationalProductFingerprint (product : OperationalProductKey) : UInt64 :=
  let seed := product.factors.foldl
    (fun state factor => mixOperationalFingerprint state (operationalFactorFingerprint factor)) 17
  product.modes.foldl
    (fun state mode => mixOperationalFingerprint state (operationalModeFingerprint mode)) seed

def internOperationalProduct
    (arena : OperationalInterningArena)
    (product : OperationalProductKey) :
    OperationalInterningArena × OperationalMonomialId :=
  let (arena, factorIds) := product.factors.foldl (fun (state, ids) factor =>
    let (state, id) := internOperationalFactor state factor
    (state, ids.push id)) (arena, #[])
  let key : OperationalMonomialKey := {
    factors := factorIds
    modes := product.modes.toArray
    outputType := product.outputType
  }
  let fingerprint := operationalMonomialFingerprint key
  let candidates := arena.monomials.buckets.getD fingerprint #[]
  match candidates.find? fun candidate =>
      arena.monomials.monomials[candidate]?.map (·.key) == some key with
  | some id => ({ arena with monomialHits := arena.monomialHits + 1 }, id)
  | none =>
      let id := arena.monomials.monomials.size
      ({ arena with
        monomials := {
          monomials := arena.monomials.monomials.push { key, product }
          buckets := arena.monomials.buckets.insert fingerprint (candidates.push id)
        }
        monomialMisses := arena.monomialMisses + 1
      }, id)

def relationSnapshotLeaf : OperationalFactorLeaf → RelationSnapshotFactorLeaf
  | .primitive identity => .primitive identity
  | .boundedSummary origin summary => .boundedSummary origin summary
  | .exactTransform tokens type => .exactTransform tokens type

def relationSnapshotFactor (factor : OperationalFactorKey) : RelationSnapshotFactor := {
  leaf := relationSnapshotLeaf factor.leaf
  transforms := factor.transforms
  inputType := factor.inputType
  outputType := factor.outputType
  role := factor.role
  boundedSummary := factor.boundedSummary
  protections := factor.protections.filter (· != .relationOwner)
}

def relationSnapshotPolynomial
    (polynomial : OperationalPolynomial) : RelationSnapshotPolynomial :=
  polynomial.map fun term => {
    coefficient := term.coefficient
    product := {
      factors := term.product.factors.map relationSnapshotFactor
      modes := term.product.modes
      outputType := term.product.outputType
    }
  }

def operationalLeafFromSnapshot : RelationSnapshotFactorLeaf → OperationalFactorLeaf
  | .primitive identity => .primitive identity
  | .boundedSummary origin summary => .boundedSummary origin summary
  | .exactTransform tokens type => .exactTransform tokens type

def operationalPolynomialFromSnapshot
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

def operationalAbsoluteCoefficient (value : Int) : Int :=
  if value < 0 then -value else value

def operationalCoefficientContent : List OperationalTerm → Nat
  | [] => 1
  | head :: tail =>
      let content := tail.foldl (fun current term => Nat.gcd current term.coefficient.natAbs)
        head.coefficient.natAbs
      if content = 0 then 1 else content

def insertCanonicalOperationalTerm
    (term : OperationalTerm) : OperationalPolynomial → OperationalPolynomial
  | [] => [term]
  | head :: tail =>
      if operationalProductFingerprint term.product < operationalProductFingerprint head.product then
        term :: head :: tail
      else head :: insertCanonicalOperationalTerm term tail

def sortOperationalTerms (terms : OperationalPolynomial) : OperationalPolynomial :=
  terms.foldl (fun sorted term => insertCanonicalOperationalTerm term sorted) []

def normalizeOperationalDimension : IntExpr → IntExpr
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

def operationalDimensionEqual (left right : IntExpr) : Bool :=
  normalizeOperationalDimension left == normalizeOperationalDimension right

def operationalSameRing (left right : MatrixTypeExpr) : Bool :=
  operationalDimensionEqual left.modulus right.modulus &&
    operationalDimensionEqual left.ringDimension right.ringDimension

def operationalIsOne : IntExpr → Bool
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

def operationalInnerDimension
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

structure OperationalNormalizationState where
  terms : Array OperationalTerm := #[]
  positions : Std.HashMap OperationalMonomialId Nat := {}
  interning : OperationalInterningArena := {}

def insertOperationalTerm
    (state : OperationalNormalizationState)
    (term : OperationalTerm) : OperationalNormalizationState :=
  if term.coefficient = 0 then state else
  let (interning, monomial) := internOperationalProduct state.interning term.product
  match state.positions[monomial]? with
  | some index => match state.terms[index]? with
      | some existing =>
          { state with
            terms := state.terms.set! index {
              existing with coefficient := existing.coefficient + term.coefficient }
            interning
          }
      | none => state
  | none =>
      let index := state.terms.size
      {
        terms := state.terms.push term
        positions := state.positions.insert monomial index
        interning
      }

def finishOperationalNormalization
    (state : OperationalNormalizationState) : OperationalPolynomial :=
  state.terms.toList.filter (·.coefficient != 0)

def normalizeOperationalTermsIn
    (interning : OperationalInterningArena)
    (terms : OperationalPolynomial) : OperationalInterningArena × OperationalPolynomial :=
  let state := terms.foldl insertOperationalTerm { interning := interning }
  (state.interning, finishOperationalNormalization state)

def normalizeOperationalTerms (terms : OperationalPolynomial) : OperationalPolynomial :=
  (normalizeOperationalTermsIn {} terms).2

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

def multiplyOperationalTerms
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
  let mut result : OperationalNormalizationState := {}
  for leftTerm in left do
    for rightTerm in right do
      result := insertOperationalTerm result (← multiplyOperationalTerms leftTerm rightTerm)
  pure (finishOperationalNormalization result)

def operationalLargeFactorCount (term : OperationalTerm) : Nat :=
  term.product.factors.countP fun factor ↦ factor.role == .large

def operationalTermIsNoise (term : OperationalTerm) : Bool :=
  operationalLargeFactorCount term = 0

def operationalTermIsSignal (term : OperationalTerm) : Bool :=
  0 < operationalLargeFactorCount term

def operationalTermIsCompressionProtected (term : OperationalTerm) : Bool :=
  term.product.factors.any fun factor => !factor.protections.isEmpty

def factorBoundedSummary
    (factor : OperationalFactorKey) : Except OperationalFlatError OperationalBoundedFactorSummary :=
  match factor.role, factor.boundedSummary with
  | .bounded, some summary => pure summary
  | _, _ => throw .missingBoundedSummary

def boundedRunTokens
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

def summarizeEntireBoundedProduct
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

def boundedNoiseTermSummary
    (term : OperationalTerm) : Except OperationalFlatError OperationalBoundedFactorSummary := do
  if !operationalTermIsNoise term then throw .cannotPreserveNoiseSeparation
  summarizeEntireBoundedProduct term.product

def boundedNoiseTermTokens
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

structure OperationalBlockPartition where
  matrixType : MatrixTypeExpr
  polynomial : OperationalPolynomial
  deriving BEq

structure OperationalBlockLayout where
  axis : ConcatAxis
  partitions : Array OperationalBlockPartition
  deriving BEq

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
  blockLayout : Option OperationalBlockLayout := none
  deriving BEq

abbrev OperationalExprId := Nat

structure UniformBoundedSchema where
  matrixType : MatrixTypeExpr
  hardBound : OperationalBoundExpr
  metadata : OperationalMatrixMetadata
  deriving BEq

structure UniformSnapshotFactorSchema where
  transforms : List OperationalFactorTransform
  inputType : MatrixTypeExpr
  outputType : MatrixTypeExpr
  role : OperationalFactorRole
  bounded : Option UniformBoundedSchema
  deriving BEq

structure UniformSnapshotTermSchema where
  coefficient : Int
  factors : List UniformSnapshotFactorSchema
  modes : List OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq

structure UniformTargetSchema where
  matrixType : MatrixTypeExpr
  matrixParams : Mxx.SamplerParams
  totalHardBound : OperationalBoundExpr
  canonicalRange : CanonicalRange
  polynomial : List UniformSnapshotTermSchema
  deriving BEq

structure UniformRelationSchema where
  isPreimage : Bool
  target : UniformTargetSchema
  base : Int := 0
  small : Bool := false
  digitCount : Nat := 0
  status : ReconstructionStatus := .available
  deriving BEq

structure UniformFactorSchema where
  transforms : List OperationalFactorTransform
  inputType : MatrixTypeExpr
  outputType : MatrixTypeExpr
  role : OperationalFactorRole
  bounded : Option UniformBoundedSchema
  protections : List OperationalCompressionProtection
  relations : List UniformRelationSchema
  deriving BEq

structure UniformTermSchema where
  coefficient : Int
  factors : List UniformFactorSchema
  modes : List OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq

structure UniformMatrixSchema where
  matrixType : MatrixTypeExpr
  matrixParams : Mxx.SamplerParams
  totalHardBound : OperationalBoundExpr
  polynomial : List UniformTermSchema
  metadata : OperationalMatrixMetadata
  canonicalRange : CanonicalRange
  hasPublicIdentity : Bool
  relations : List UniformRelationSchema
  deriving BEq

def uniformBoundedSchema
    (summary : OperationalBoundedFactorSummary) : UniformBoundedSchema := {
  matrixType := summary.matrixType
  hardBound := summary.hardBound
  metadata := summary.metadata
}

def uniformSnapshotFactorSchema
    (factor : RelationSnapshotFactor) : UniformSnapshotFactorSchema := {
  transforms := factor.transforms
  inputType := factor.inputType
  outputType := factor.outputType
  role := factor.role
  bounded := factor.boundedSummary.map uniformBoundedSchema
}

def uniformSnapshotTermSchema
    (term : RelationSnapshotTerm) : UniformSnapshotTermSchema := {
  coefficient := term.coefficient
  factors := term.product.factors.map uniformSnapshotFactorSchema
  modes := term.product.modes
  outputType := term.product.outputType
}

def uniformTargetSchema (target : RelationTargetSummary) : UniformTargetSchema := {
  matrixType := target.matrixType
  matrixParams := target.matrixParams
  totalHardBound := target.totalHardBound
  canonicalRange := target.canonicalRange
  polynomial := target.polynomial.map uniformSnapshotTermSchema
}

def uniformRelationSchema : OperationalMatrixRelation → UniformRelationSchema
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

def uniformFactorSchema (factor : OperationalFactorKey) : UniformFactorSchema := {
  transforms := factor.transforms
  inputType := factor.inputType
  outputType := factor.outputType
  role := factor.role
  bounded := factor.boundedSummary.map uniformBoundedSchema
  protections := factor.protections
  relations := factor.relations.map uniformRelationSchema
}

def uniformTermSchema (term : OperationalTerm) : UniformTermSchema := {
  coefficient := term.coefficient
  factors := term.product.factors.map uniformFactorSchema
  modes := term.product.modes
  outputType := term.product.outputType
}

def operationalUniformSchema (fact : OperationalMatrixFact) : UniformMatrixSchema := {
  matrixType := fact.matrixType
  matrixParams := fact.matrixParams
  totalHardBound := fact.totalHardBound
  polynomial := fact.polynomial.map uniformTermSchema
  metadata := fact.metadata
  canonicalRange := fact.canonicalRange
  hasPublicIdentity := fact.identity.isSome
  relations := fact.relations.map uniformRelationSchema
}

def matrixFactHasRelation (fact : OperationalMatrixFact) : Bool :=
  !fact.relations.isEmpty || fact.polynomial.any fun term =>
    term.product.factors.any fun factor => !factor.relations.isEmpty

/-- Exact relations carried by a fact. Polynomial distribution may duplicate one relation onto
multiple terms, so preservation compares membership in both directions rather than multiplicity.
Identity, producer, target, and relation parameters remain part of the comparison. -/
def operationalRelationInventory
    (fact : OperationalMatrixFact) : List OperationalMatrixRelation :=
  fact.relations ++ fact.polynomial.flatMap fun term =>
    term.product.factors.flatMap (·.relations)

def sameOperationalRelationInventory
    (left right : List OperationalMatrixRelation) : Bool :=
  left.all (fun relation => right.any (· == relation)) &&
    right.all (fun relation => left.any (· == relation))

def boundaryLastPublicIdentity?
    (fact : OperationalMatrixFact) : Option PublicMatrixIdentity := do
  let term ← match fact.polynomial with | [term] => some term | _ => none
  let factor ← term.product.factors.getLast?
  match factor.leaf with
  | .primitive (.publicMatrix identity) => some identity
  | _ => none

def boundaryFirstRelationPublicIdentity?
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

def publicIdentityTemplateEqual : PublicMatrixIdentity → PublicMatrixIdentity → Bool
  | .sampledTrapdoor leftScope leftWire, .sampledTrapdoor rightScope rightWire =>
      leftScope == rightScope && leftWire == rightWire
  | .gadget leftId leftParams leftRows leftBase leftSmall leftDigits,
      .gadget rightId rightParams rightRows rightBase rightSmall rightDigits =>
      leftId == rightId && leftParams == rightParams && leftRows == rightRows &&
        leftBase == rightBase && leftSmall == rightSmall && leftDigits == rightDigits
  | .loopInstance leftSlot _ leftSource, .loopInstance rightSlot _ rightSource =>
      leftSlot == rightSlot && publicIdentityTemplateEqual leftSource rightSource
  | .indexed leftBinder leftExpression leftSource,
      .indexed rightBinder rightExpression rightSource =>
      leftBinder == rightBinder && leftExpression == rightExpression &&
        publicIdentityTemplateEqual leftSource rightSource
  | _, _ => false

structure SelectedMatrixSummary where
  uniformSchema : Option UniformMatrixSchema
  /-- Complete conservative value for the selected outer domain.  The structural representative
  retains branch-local identities and nested choices; this fact owns the joined bounds, metadata,
  canonical range, and relation schema used by transfer and endpoint evaluation. -/
  conservativeFact : Option OperationalMatrixFact
  relationFree : Bool
  sharedLastPublicIdentity : Option PublicMatrixIdentity
  selectionOrigin : Option SelectionDomainKind := none
  deriving BEq

/-- Request-local handle for an all-branch matrix schema.  Equality of handles is constant time;
the interner always confirms the complete schema key after fingerprint bucket selection. -/
structure ValidatedSchemaId where
  ordinal : Nat
  deriving BEq, DecidableEq, Repr

structure OperationalExprEvaluationStats where
  evaluations : Nat := 0
  memoHits : Nat := 0
  memoMisses : Nat := 0
  deriving BEq, DecidableEq, Inhabited, Repr

structure OperationalExprEvaluationState where
  environment : Option ParamEnvironment := none
  totalMemo : Array (Option Int) := #[]
  noiseMemo : Array (Option Int) := #[]
  schemaFactMemo : Array (Option OperationalMatrixFact) := #[]
  schemaMemo : Array (Option ValidatedSchemaId) := #[]
  totalStats : OperationalExprEvaluationStats := {}
  noiseStats : OperationalExprEvaluationStats := {}
  schemaStats : OperationalExprEvaluationStats := {}
  deriving BEq

def selectedMatrixSummary
    (branches : Array OperationalMatrixFact) : SelectedMatrixSummary :=
  match branches[0]? with
  | none => {
      uniformSchema := none
      conservativeFact := none
      relationFree := false
      sharedLastPublicIdentity := none
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
      {
        uniformSchema := if uniform then some schema else none
        conservativeFact := if uniform then some first else none
        relationFree
        sharedLastPublicIdentity := sharedLast
      }

inductive EnvelopeSummaryTransferOperation where
  | instantiationMap
  | recurrenceBoundShift
  | addSubtract
  | multiplyOrdinary
  | tensor
  | concat
  | transform
  | scale
  | bggGrouping
  | unregistered
  deriving BEq, DecidableEq, Repr

/-- Fail-closed registry for operations that may transfer a checked uniform envelope. Every
source must already carry a complete uniform schema. Registered operations recompute every output
field from the post-operation representative; no pre-operation boundary template is copied. -/
def transferSelectedMatrixSummary
    (operation : EnvelopeSummaryTransferOperation)
    (sources : Array SelectedMatrixSummary)
    (conservativeOutput : OperationalMatrixFact) : Option SelectedMatrixSummary := do
  if sources.isEmpty || sources.any (fun source =>
      source.uniformSchema.isNone || source.conservativeFact.isNone) then
    none
  let first ← sources[0]?
  let alignedOrigin := sources.all (·.selectionOrigin == first.selectionOrigin)
  let recomputed := {
    selectedMatrixSummary #[conservativeOutput] with selectionOrigin := first.selectionOrigin }
  if recomputed.uniformSchema.isNone || recomputed.conservativeFact.isNone then
    none
  match operation with
  | .instantiationMap | .recurrenceBoundShift | .transform | .scale =>
      if sources.size == 1 then some recomputed else none
  | .multiplyOrdinary =>
      if sources.size <= 2 && alignedOrigin then
        some recomputed
      else none
  | .addSubtract | .tensor =>
      if sources.size <= 2 && alignedOrigin then some recomputed else none
  | .concat | .bggGrouping =>
      if alignedOrigin then some recomputed else none
  | .unregistered => none

def primitiveOperationalPolynomial
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

def OperationalMatrixFact.initializePrimitivePolynomial
    (fact : OperationalMatrixFact)
    (role : OperationalFactorRole) : OperationalMatrixFact := {
  fact with polynomial := (primitiveOperationalPolynomial fact.origin fact.matrixType
    fact.totalHardBound role fact.identity fact.relations fact.metadata)
}

def OperationalMatrixFact.primitiveRole (fact : OperationalMatrixFact) :
    OperationalFactorRole :=
  match fact.polynomial.head? >>= fun term => term.product.factors.head? with
  | some factor => factor.role
  | none => .bounded

def OperationalMatrixFact.refreshPrimitivePolynomial
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

abbrev OperationalScalarExprId := Nat
abbrev OperationalIndexedValueId := Nat

inductive OperationalPayloadRef where
  | matrix (id : OperationalExprId)
  | directValue (id : OperationalIndexedValueId)
  | scalar (id : OperationalScalarExprId)
  deriving BEq, Repr

def OperationalPayloadRef.root : OperationalPayloadRef → Nat
  | .matrix root | .directValue root | .scalar root => root

instance : Coe OperationalPayloadRef Nat := ⟨OperationalPayloadRef.root⟩

abbrev IndexedOperationalFact := IndexedFact OperationalPayloadRef

/-- Non-matrix operational atoms stored in the request-local arena.  This type is deliberately
nonrecursive: an indexed family is represented by an arena selection node, never by nesting an
`OperationalFact` inside another `OperationalFact`. -/
inductive OperationalScalarFact where
  | integer (fact : OperationalIntegerFact)
  | boolean
  | real
  | trapdoor (fact : OperationalTrapdoorFact)
  | bytes (fact : OperationalBytesFact)
  | typedBlob (typeName : String)
  | unknown (wireType : WireTypeExpr)
  deriving BEq

inductive OperationalScalarPrimitiveKind where
  | boolToInt
  | intBinary (operation : IntBinaryOp)
  | intCompare (operation : IntCompareOp)
  | intToReal
  | realBinary (operation : RealBinaryOp)
  | realSqrt
  deriving BEq

/-- Matrix-to-scalar kernels that retain the input's indexed assignment.  These are graph
operations, not a conversion through the legacy scalar selection arena. -/
inductive DirectValueScalarOperationKind where
  | extractCoefficient (position : IntExpr)
  | thresholdDecodeBool (ciphertextModulus plaintextModulus length : IntExpr)
  | thresholdDecodeInt (ciphertextModulus plaintextModulus length : IntExpr)
  deriving BEq

structure DirectValueScalarOperation where
  kind : DirectValueScalarOperationKind
  ownerScope : Option ScopeTemplateKey
  ownerNode : Nat
  outputPort : Nat
  parameterEnvironment : ParamEnvironment
  deriving BEq

/-- Scalar-to-matrix kernels retain the scalar's indexed assignment.  Keeping this distinct from
matrix primitives prevents a scalar lift from being smuggled through a matrix-only descriptor. -/
inductive DirectValueMatrixOperationKind where
  | liftIntegerToConstantPolynomial (matrixType : MatrixTypeExpr)
  deriving BEq

structure DirectValueMatrixOperation where
  kind : DirectValueMatrixOperationKind
  ownerScope : Option ScopeTemplateKey
  ownerNode : Nat
  outputPort : Nat
  parameterEnvironment : ParamEnvironment
  deriving BEq

inductive OperationalScalarExprNode where
  | concrete (fact : OperationalScalarFact)
  | primitive
      (kind : OperationalScalarPrimitiveKind)
      (arguments : Array Nat)
      (result : OperationalScalarFact)
  | selectExact (domain : SelectionDomainId) (branches : Array Nat)
  | selectShared
      (domain : SelectionDomainId)
      (binder : FamilyTemplateBinder)
      (subject : WireRef)
      (representative : Nat)
  deriving BEq

abbrev OperationalFact := IndexedOperationalFact

def OperationalFact.operationalExprRoot? : OperationalFact → Option OperationalExprId
  | { payload := .matrix root, .. } => some root
  | _ => none

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
  | invalidOperationalDecoderTarget (targetId : String)
  | emptyOperationalDecoderTargetRegistry
  | unknownOperationalDecoderTarget (targetId : String)
  | duplicateOperationalDecoderTarget (targetId : String)
  | missingProtocolContract (name : String)
  | inputContractMismatch (name : String)
  | unknownDerivationAttachment (ownerNamespace ruleName : String)
  | missingDerivationAttachmentRole (ownerNamespace ruleName roleName : String)
  | invalidDerivationAttachment (ownerNamespace ruleName : String)
  | invalidOperationalExprRef (id : Nat)
  | operationalExprTypeMismatch (left right : Nat)
  /-- Decoder acceptance may only bound a fully signal-free residual. -/
  | residualContainsLargeTerm (node : Nat)
  | incompatibleRelationDomains (node leftDomain rightDomain : Nat)
  | unknownRelationRequirement (node expression : Nat)
  | unresolvedConcreteStructure (node expression : Nat)
  | unsupportedOperationalExpr (id : Nat)
  | unsupportedNode (node : Nat)
  deriving BEq, DecidableEq, Repr


end Mxx.Certificate
