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

inductive SelectionDomainKind where
  | loopLane
  | protocolSelection
  deriving BEq, DecidableEq, Repr

private def scopeIsLoopLane : ScopeTemplateKey → Bool
  | .parallelBody .. => true
  | .callBody parent _ | .sequentialBody parent _ => scopeIsLoopLane parent
  | .root _ => false

private def selectionDomainKind : OperationalValueOrigin → SelectionDomainKind
  | .loopInstance .. => .loopLane
  | .local scope _ => if scopeIsLoopLane scope then .loopLane else .protocolSelection
  | .selected _ index _ => selectionDomainKind index
  | .protocolInput _ | .protocolFamilyElement _ _ => .protocolSelection

private structure SelectionDomainKey where
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

private class SelectionIdentityLike (α : Type) where
  identity : α → DynamicSelectionIdentity
  domainCount? : α → Option Nat

private instance : SelectionIdentityLike DynamicSelectionIdentity where
  identity := id
  domainCount? _ := none

private instance : SelectionIdentityLike SelectionDomainId where
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

private structure OperationalMonomialKey where
  factors : Array OperationalFactorId
  modes : Array OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq

private structure OperationalFactorInterner where
  factors : Array OperationalFactorKey := #[]
  buckets : Std.HashMap UInt64 (Array OperationalFactorId) := {}
  deriving BEq

private structure OperationalMonomialEntry where
  key : OperationalMonomialKey
  product : OperationalProductKey
  deriving BEq

private structure OperationalMonomialInterner where
  monomials : Array OperationalMonomialEntry := #[]
  buckets : Std.HashMap UInt64 (Array OperationalMonomialId) := {}
  deriving BEq

/-- Request-local, analysis-only identity tables.  IDs are meaningful only while this arena is
threaded through one operational evaluation; canonical factor/product structures remain the
authority for equality and serialization. -/
private structure OperationalInterningArena where
  factors : OperationalFactorInterner := {}
  monomials : OperationalMonomialInterner := {}
  factorHits : Nat := 0
  factorMisses : Nat := 0
  monomialHits : Nat := 0
  monomialMisses : Nat := 0
  deriving BEq

private def mixOperationalFingerprint (state value : UInt64) : UInt64 :=
  state * 1099511628211 + value + 1469598103934665603

private def operationalRoleFingerprint : OperationalFactorRole → UInt64
  | .bounded => 1
  | .large => 2

private def operationalModeFingerprint : OperationalProductMode → UInt64
  | .ordinaryMatrixProduct => 1
  | .leftPolynomialScalarBroadcast => 2
  | .rightPolynomialScalarBroadcast => 3
  | .swappedRowVectorScalarProduct => 4

private def operationalIntExprFingerprint : IntExpr → UInt64
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

private def operationalProgramFingerprint : ProgramInstanceKey → UInt64
  | .temporary => 1
  | .workflowStage stage => mixOperationalFingerprint 2 (hash stage.name)
  | .ideal => 3
  | .requirement index => mixOperationalFingerprint 4 (UInt64.ofNat index)
  | .standalone ordinal => mixOperationalFingerprint 5 (UInt64.ofNat ordinal)

private def operationalScopeFingerprint : ScopeTemplateKey → UInt64
  | .root program => mixOperationalFingerprint 7 (operationalProgramFingerprint program)
  | .callBody parent node => mixOperationalFingerprint
      (mixOperationalFingerprint 11 (operationalScopeFingerprint parent)) (UInt64.ofNat node)
  | .parallelBody parent node => mixOperationalFingerprint
      (mixOperationalFingerprint 13 (operationalScopeFingerprint parent)) (UInt64.ofNat node)
  | .sequentialBody parent node => mixOperationalFingerprint
      (mixOperationalFingerprint 17 (operationalScopeFingerprint parent)) (UInt64.ofNat node)

private def operationalBinderFingerprint (binder : FamilyTemplateBinder) : UInt64 :=
  mixOperationalFingerprint
    (mixOperationalFingerprint (operationalScopeFingerprint binder.owner)
      (UInt64.ofNat binder.producerNode))
    (UInt64.ofNat binder.binderSlot)

private def operationalValueOriginFingerprint : OperationalValueOrigin → UInt64
  | .local scope wire => mixOperationalFingerprint
      (mixOperationalFingerprint (operationalScopeFingerprint scope) (UInt64.ofNat wire.node))
      (UInt64.ofNat wire.port)
  | .protocolInput input => mixOperationalFingerprint 19 (hash input.name)
  | .protocolFamilyElement input index =>
      mixOperationalFingerprint (mixOperationalFingerprint 23 (hash input.name))
        (UInt64.ofNat index)
  | .loopInstance slot index source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 29 (operationalValueOriginFingerprint source))
          (UInt64.ofNat slot))
        (UInt64.ofNat index)
  | .selected binder index source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 31 (operationalBinderFingerprint binder))
          (operationalValueOriginFingerprint index))
        (operationalValueOriginFingerprint source)

private def operationalSelectionFingerprint (selection : DynamicSelectionIdentity) : UInt64 :=
  operationalValueOriginFingerprint selection.index

private def operationalMatrixOriginFingerprint : MatrixOriginIdentity → UInt64
  | .value scope wire => mixOperationalFingerprint
      (mixOperationalFingerprint (operationalScopeFingerprint scope) (UInt64.ofNat wire.node))
      (UInt64.ofNat wire.port)
  | .protocolInput input => mixOperationalFingerprint 37 (hash input.name)
  | .protocolFamilyElement input index =>
      mixOperationalFingerprint (mixOperationalFingerprint 41 (hash input.name))
        (UInt64.ofNat index)
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
        (UInt64.ofNat index)
  | .selected binder selection source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 53 (operationalBinderFingerprint binder))
          (operationalSelectionFingerprint selection))
        (operationalMatrixOriginFingerprint source)

private def operationalPublicMatrixFingerprint : PublicMatrixIdentity → UInt64
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
  | .selected binder selection source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 67 (operationalBinderFingerprint binder))
          (operationalSelectionFingerprint selection))
        (operationalPublicMatrixFingerprint source)
  | .loopInstance slot index source =>
      mixOperationalFingerprint
        (mixOperationalFingerprint
          (mixOperationalFingerprint 71 (operationalPublicMatrixFingerprint source))
          (UInt64.ofNat slot))
        (UInt64.ofNat index)

private def operationalPrimitiveFingerprint : OperationalPrimitiveIdentity → UInt64
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
      (mixOperationalFingerprint 31 (hash input.name)) (operationalIntExprFingerprint index)
  | .recurrenceResult _ node path =>
      mixOperationalFingerprint (mixOperationalFingerprint 37 (UInt64.ofNat node))
        (UInt64.ofNat path)
  | .carriedInput path => mixOperationalFingerprint 41 (UInt64.ofNat path)

private def operationalLeafFingerprint : OperationalFactorLeaf → UInt64
  | .primitive identity => operationalPrimitiveFingerprint identity
  | .boundedSummary _ _ => 2
  | .exactTransform _ _ => 3

/-- A deliberately compact collision-prone fingerprint.  It is only a bucket selector; complete
`OperationalFactorKey` equality below remains the semantic check. -/
private def operationalFactorFingerprint (factor : OperationalFactorKey) : UInt64 :=
  let seed := mixOperationalFingerprint (operationalLeafFingerprint factor.leaf)
    (operationalRoleFingerprint factor.role)
  let seed := mixOperationalFingerprint seed (UInt64.ofNat factor.transforms.length)
  let seed := mixOperationalFingerprint seed (UInt64.ofNat factor.protections.length)
  mixOperationalFingerprint seed (UInt64.ofNat factor.relations.length)

private def internOperationalFactor
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

private def operationalMonomialFingerprint (key : OperationalMonomialKey) : UInt64 :=
  let seed := key.factors.foldl
    (fun state factor => mixOperationalFingerprint state (UInt64.ofNat factor)) 17
  let seed := key.modes.foldl
    (fun state mode => mixOperationalFingerprint state (operationalModeFingerprint mode)) seed
  mixOperationalFingerprint seed (UInt64.ofNat key.factors.size)

private def operationalProductFingerprint (product : OperationalProductKey) : UInt64 :=
  let seed := product.factors.foldl
    (fun state factor => mixOperationalFingerprint state (operationalFactorFingerprint factor)) 17
  product.modes.foldl
    (fun state mode => mixOperationalFingerprint state (operationalModeFingerprint mode)) seed

private def internOperationalProduct
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
      if operationalProductFingerprint term.product < operationalProductFingerprint head.product then
        term :: head :: tail
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

private structure OperationalNormalizationState where
  terms : Array OperationalTerm := #[]
  positions : Std.HashMap OperationalMonomialId Nat := {}
  interning : OperationalInterningArena := {}

private def insertOperationalTerm
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

private def finishOperationalNormalization
    (state : OperationalNormalizationState) : OperationalPolynomial :=
  state.terms.toList.filter (·.coefficient != 0)

private def normalizeOperationalTermsIn
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
  selectionOrigin : Option SelectionDomainKind := none
  deriving BEq

/-- Request-local handle for an all-branch matrix schema.  Equality of handles is constant time;
the interner always confirms the complete schema key after fingerprint bucket selection. -/
structure ValidatedSchemaId where
  ordinal : Nat
  deriving BEq, DecidableEq, Repr

/-- Structural relation demand used by relation-consuming primitive lifting. `branchLocal` names
the mutually-exclusive domain whose concrete lane identity is required by the relation. A Shared
choice remains branch-local: its schema is uniform, but producer and public identities are still
lane-specific. -/
inductive RelationRequirement where
  | none
  | uniform (schema : UniformMatrixSchema)
  | branchLocal (domain : SelectionDomainId)
  | unknown
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

private inductive EnvelopeSummaryTransferOperation where
  | instantiationMap
  | recurrenceBoundShift
  | addSubtract
  | multiplyRelation
  | tensor
  | concat
  | transform
  | scale
  | bggGrouping
  | preimage
  | decomposition
  | unregistered
  deriving BEq, DecidableEq, Repr

/-- Fail-closed registry for operations that may transfer a checked uniform envelope. Every
source must already carry a complete uniform schema. Registered operations recompute every output
field from the post-operation representative; no pre-operation boundary template is copied. -/
private def transferSelectedMatrixSummary
    (operation : EnvelopeSummaryTransferOperation)
    (sources : Array SelectedMatrixSummary)
    (representative : OperationalMatrixFact) : Option SelectedMatrixSummary := do
  if sources.isEmpty || sources.any (fun source => source.uniformSchema.isNone) then none
  let first ← sources[0]?
  let alignedOrigin := sources.all (·.selectionOrigin == first.selectionOrigin)
  let recomputed := {
    selectedMatrixSummary #[representative] with selectionOrigin := first.selectionOrigin }
  if recomputed.uniformSchema.isNone then none
  match operation with
  | .instantiationMap | .recurrenceBoundShift | .transform | .scale |
      .preimage | .decomposition =>
      if sources.size == 1 then some recomputed else none
  | .addSubtract | .multiplyRelation | .tensor =>
      if sources.size <= 2 && alignedOrigin then some recomputed else none
  | .concat | .bggGrouping =>
      if alignedOrigin then some recomputed else none
  | .unregistered => none

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

private def packedMatrixEnvelopeIsComplete
    (elements : Array OperationalFact)
    (count : Nat)
    (summary : Option SelectedMatrixSummary) : Bool :=
  if elements.size == count then true
  else if elements.size != 1 || count <= elements.size then false
  else match summary, elements[0]? with
    | some claimed, some element => match element with
        | OperationalFact.matrix representative =>
            claimed.uniformSchema.isSome && claimed == selectedMatrixSummary #[representative]
        | _ => false
    | _, _ => false

/-- Rebuild a validated family after a registered deterministic map.  A full exact family is
validated only when it is first packed.  Registered maps transfer the checked schema through the
post-map representative; unregistered or incomplete transfers discard the summary fail-closed
instead of rescanning every branch or retaining stale metadata. -/
private def transferPackedOperationalFamily
    (operation : EnvelopeSummaryTransferOperation)
    (elements : Array OperationalFact)
    (count : Nat)
    (sourceSummary : Option SelectedMatrixSummary) : OperationalFact :=
  match sourceSummary, elements[0]? with
  | some source, some element =>
      match element with
      | OperationalFact.matrix representative =>
          match transferSelectedMatrixSummary operation #[source] representative with
          | some summary => .familyPacked elements count (some summary)
          | none => .familyPacked elements count none
      | _ => .familyPacked elements count none
  | _, _ => .familyPacked elements count none

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
  | incompatibleRelationDomains (node leftDomain rightDomain : Nat)
  | unknownRelationRequirement (node expression : Nat)
  | unresolvedConcreteStructure (node expression : Nat)
  | unsupportedOperationalExpr (id : Nat)
  | unsupportedNode (node : Nat)
  deriving BEq, DecidableEq, Repr

/-! ## Selection-preserving operational expressions

The executable checker still evaluates ordinary operations into the flat facts above.  This
request-local arena is the compact boundary for unresolved dynamic selections.  Arena indices are
allocation identities only: they are never matrix, relation, or symbolic-equality evidence. -/

/-- Lossless descriptor for one delayed matrix operation.  It records every input needed to invoke
the existing concrete transfer later; it never carries a replacement noise formula. -/
inductive PrimitiveOperationKind where
  | add (subtract : Bool)
  | multiply (rule : DerivationRule) (rightWire : WireRef)
  | tensor
  | concat (axis : ConcatAxis)
  | transform (operation : OperationalFactorTransform)
  deriving BEq

structure PrimitiveOperation where
  kind : PrimitiveOperationKind
  outputType : MatrixTypeExpr
  ownerScope : Option ScopeTemplateKey
  ownerNode : Nat
  outputPort : Nat
  parameterEnvironment : ParamEnvironment
  deriving BEq

inductive ChoiceStorage where
  | exact (branches : Array OperationalExprId)
  | shared
      (representative : OperationalExprId)
      (schema : ValidatedSchemaId)
  deriving BEq

private structure SelectionDomainInterner where
  keys : Array SelectionDomainKey := #[]
  buckets : Std.HashMap UInt64 (Array Nat) := {}
  deriving BEq

private structure ValidatedSchemaInterner where
  schemas : Array SelectedMatrixSummary := #[]
  buckets : Std.HashMap UInt64 (Array Nat) := {}
  deriving BEq

inductive OperationalMatrixExprNode where
  | concrete (fact : OperationalMatrixFact)
  | primitive (operation : PrimitiveOperation) (arguments : Array OperationalExprId)
  | select
      (domain : SelectionDomainId)
      (branches : ChoiceStorage)
  deriving BEq

structure OperationalMatrixExpr where
  matrixType : MatrixTypeExpr
  node : OperationalMatrixExprNode
  containsSelection : Bool := false
  ownerScope : Option ScopeTemplateKey := none
  ownerNode : Option Nat := none
  deriving BEq

structure OperationalExprArena where
  nodes : Array OperationalMatrixExpr := #[]
  selectionDomains : SelectionDomainInterner := {}
  validatedSchemas : ValidatedSchemaInterner := {}
  activeScope : Option ScopeTemplateKey := none
  activeNode : Option Nat := none
  relationRewriteCount : Nat := 0
  transformCacheHits : Nat := 0
  transformCacheMisses : Nat := 0
  deriving BEq

private def selectionDomainFingerprint (key : SelectionDomainKey) : UInt64 :=
  let kind := match key.kind with | .loopLane => 1 | .protocolSelection => 2
  mixOperationalFingerprint
    (mixOperationalFingerprint kind (operationalSelectionFingerprint key.identity))
    (UInt64.ofNat key.count)

private def OperationalExprArena.internSelectionDomain
    (arena : OperationalExprArena)
    (identity : DynamicSelectionIdentity)
    (count : Nat) : OperationalExprArena × SelectionDomainId :=
  let key : SelectionDomainKey := { kind := selectionDomainKind identity.index, identity, count }
  let fingerprint := selectionDomainFingerprint key
  let candidates := arena.selectionDomains.buckets.getD fingerprint #[]
  match candidates.find? fun candidate => arena.selectionDomains.keys[candidate]? == some key with
  | some ordinal => (arena, { ordinal, kind := key.kind, identity, count })
  | none =>
      let ordinal := arena.selectionDomains.keys.size
      ({ arena with selectionDomains := {
          keys := arena.selectionDomains.keys.push key
          buckets := arena.selectionDomains.buckets.insert fingerprint (candidates.push ordinal)
        } }, { ordinal, kind := key.kind, identity, count })

private def validatedSchemaFingerprint (schema : SelectedMatrixSummary) : UInt64 :=
  let seed := mixOperationalFingerprint 127 (if schema.relationFree then 1 else 0)
  let seed := mixOperationalFingerprint seed (if schema.uniformSchema.isSome then 1 else 0)
  let seed := match schema.sharedLastPublicIdentity with
    | some identity => mixOperationalFingerprint seed (operationalPublicMatrixFingerprint identity)
    | none => mixOperationalFingerprint seed 131
  let seed := match schema.sharedFirstRelationPublicIdentity with
    | some identity => mixOperationalFingerprint seed (operationalPublicMatrixFingerprint identity)
    | none => mixOperationalFingerprint seed 137
  match schema.selectionOrigin with
  | some .loopLane => mixOperationalFingerprint seed 139
  | some .protocolSelection => mixOperationalFingerprint seed 149
  | none => mixOperationalFingerprint seed 151

private def OperationalExprArena.internValidatedSchema
    (arena : OperationalExprArena)
    (schema : SelectedMatrixSummary) : OperationalExprArena × ValidatedSchemaId :=
  let fingerprint := validatedSchemaFingerprint schema
  let candidates := arena.validatedSchemas.buckets.getD fingerprint #[]
  match candidates.find? fun candidate => arena.validatedSchemas.schemas[candidate]? == some schema with
  | some ordinal => (arena, { ordinal })
  | none =>
      let ordinal := arena.validatedSchemas.schemas.size
      ({ arena with validatedSchemas := {
          schemas := arena.validatedSchemas.schemas.push schema
          buckets := arena.validatedSchemas.buckets.insert fingerprint (candidates.push ordinal)
        } }, { ordinal })

private def OperationalExprArena.validatedSchema
    (arena : OperationalExprArena)
    (id : ValidatedSchemaId) : Except OperationalError SelectedMatrixSummary :=
  match arena.validatedSchemas.schemas[id.ordinal]? with
  | some schema => pure schema
  | none => throw (.unsupportedOperationalExpr id.ordinal)

structure OperationalExprEvaluationStats where
  evaluations : Nat := 0
  memoHits : Nat := 0
  memoMisses : Nat := 0
  deriving BEq, DecidableEq, Inhabited, Repr

private structure OperationalExprEvaluationState where
  totalMemo : Array (Option Int)
  noiseMemo : Array (Option Int)
  representativeMemo : Array (Option OperationalMatrixFact)
  schemaFactMemo : Array (Option OperationalMatrixFact)
  relationMemo : Array (Option RelationRequirement)
  totalStats : OperationalExprEvaluationStats := {}
  noiseStats : OperationalExprEvaluationStats := {}
  relationStats : OperationalExprEvaluationStats := {}
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
  let ownerNode := match expression.ownerNode with
    | some owner => some owner
    | none => arena.activeNode
  let childContainsSelection (id : OperationalExprId) :=
    match arena.nodes[id]? with
    | some child => child.containsSelection
    | none => false
  let containsSelection := match expression.node with
    | .concrete _ => false
    | .primitive _ arguments => arguments.any childContainsSelection
    | .select .. => true
  ({ arena with nodes := arena.nodes.push {
      expression with containsSelection, ownerScope := arena.activeScope, ownerNode
    } },
    arena.nodes.size)

private def OperationalExprArena.pushConcrete
    (arena : OperationalExprArena)
    (fact : OperationalMatrixFact) : OperationalExprArena × OperationalExprId :=
  arena.push {
    matrixType := fact.matrixType
    node := .concrete fact
    ownerNode := some fact.subject.node }

private def OperationalExprArena.pushPrimitive
    (arena : OperationalExprArena)
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (kind : PrimitiveOperationKind)
    (arguments : Array OperationalExprId) : OperationalExprArena × OperationalExprId :=
  arena.push {
    matrixType
    node := .primitive {
      kind
      outputType := matrixType
      ownerScope := arena.activeScope
      ownerNode := nodeIndex
      outputPort
      parameterEnvironment := environment
    } arguments
    ownerNode := some nodeIndex
  }

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

private def OperationalExprArena.pushCheckedSchemaEnvelope
    {α : Type} [SelectionIdentityLike α]
    (arena : OperationalExprArena)
    (selection : α)
    (count representative : Nat)
    (summary : SelectedMatrixSummary)
    (fact : OperationalMatrixFact) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  if count = 0 then throw (.invalidCount 0 0)
  let selection := SelectionIdentityLike.identity selection
  let expectedOrigin := some (selectionDomainKind selection.index)
  -- The attached selection identity is authoritative. Namespace and loop-template substitution
  -- can change its analysis category without changing the selected alternatives, so recompute
  -- this derived classification instead of treating stale transport metadata as semantics.
  let summary := { summary with selectionOrigin := expectedOrigin }
  let expression ← match arena.get? representative with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef representative)
  if summary.uniformSchema != some (operationalUniformSchema fact) ||
      summary.relationFree != !matrixFactHasRelation fact ||
      summary.sharedLastPublicIdentity != boundaryLastPublicIdentity? fact ||
      summary.sharedFirstRelationPublicIdentity !=
        boundaryFirstRelationPublicIdentity? fact then
    throw (.unsupportedOperationalExpr representative)
  let (arena, domain) := arena.internSelectionDomain selection count
  let (arena, schema) := arena.internValidatedSchema summary
  pure (arena.push {
    matrixType := expression.matrixType
    node := .select domain (.shared representative schema)
  })

private def OperationalExprArena.pushSelect
    {α : Type} [SelectionIdentityLike α]
    (arena : OperationalExprArena)
    (selection : α)
    (branches : ChoiceStorage) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let domainCount? := SelectionIdentityLike.domainCount? selection
  let selection := SelectionIdentityLike.identity selection
  match branches with
  | .exact values =>
      let first ← match values[0]? with
        | some first => pure first
        | none => throw (.invalidCount 0 0)
      let matrixType ← arena.checkedType first (values.extract 1 values.size)
      if values.all (· == first) then pure (arena, first)
      else
        let (arena, domain) := arena.internSelectionDomain selection values.size
        pure (arena.push { matrixType, node := .select domain branches })
  | .shared representative schema =>
      let count ← match domainCount? with
        | some count => pure count
        | none => throw (.unsupportedOperationalExpr representative)
      let summary ← arena.validatedSchema schema
      let expression ← match arena.get? representative with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef representative)
      let summary := {
        summary with selectionOrigin := some (selectionDomainKind selection.index)
      }
      if count = 0 || summary.uniformSchema.isNone then
        throw (.unsupportedOperationalExpr representative)
      match expression.node with
      | .concrete fact =>
          if summary.uniformSchema != some (operationalUniformSchema fact) ||
              summary.relationFree != !matrixFactHasRelation fact ||
              summary.sharedLastPublicIdentity != boundaryLastPublicIdentity? fact ||
              summary.sharedFirstRelationPublicIdentity !=
                boundaryFirstRelationPublicIdentity? fact then
            throw (.unsupportedOperationalExpr representative)
      | _ => pure ()
      let (arena, domain) := arena.internSelectionDomain selection count
      let (arena, schema) := arena.internValidatedSchema summary
      pure (arena.push {
        matrixType := expression.matrixType
        node := .select domain (.shared representative schema)
      })

private def OperationalExprArena.pushSharedSelection
    (arena : OperationalExprArena)
    (selection : DynamicSelectionIdentity)
    (count representative : Nat)
    (summary : SelectedMatrixSummary) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let (arena, domain) := arena.internSelectionDomain selection count
  let (arena, schema) := arena.internValidatedSchema summary
  arena.pushSelect domain (.shared representative schema)

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

/-- An operation over an envelope may recover either the complete exact alternatives or an already
checked post-operation envelope carried by its representative expression.  Either form already
has the required selection identity and logical domain and must not be wrapped in a second,
redundant envelope. -/
private def OperationalExprArena.isMatchingSelection
    {α : Type} [SelectionIdentityLike α]
    (arena : OperationalExprArena)
    (selection : α)
    (count root : Nat) : Bool :=
  let selection := SelectionIdentityLike.identity selection
  match arena.get? root with
  | some { node := .select actual (.exact branches), .. } =>
      actual.identity == selection && branches.size == count
  | some { node := .select actual (.shared _ _), .. } =>
      actual.identity == selection && actual.count == count
  | _ => false

private def OperationalExprArena.containsSelection
    (arena : OperationalExprArena)
    (root : OperationalExprId) : Except OperationalError Bool :=
  match arena.get? root with
  | some expression => pure expression.containsSelection
  | none => throw (.invalidOperationalExprRef root)

private def ChoiceStorage.staticBranch
    (branches : ChoiceStorage)
    (index : Nat) : Except OperationalError OperationalExprId :=
  match branches with
  | .exact values => match values[index]? with
      | some value => pure value
      | none => throw (.invalidCount 0 index)
  | .shared representative _ => throw (.unsupportedOperationalExpr representative)

private def OperationalExprEvaluationState.empty
    (arena : OperationalExprArena) : OperationalExprEvaluationState := {
  totalMemo := Array.replicate arena.nodes.size none
  noiseMemo := Array.replicate arena.nodes.size none
  representativeMemo := Array.replicate arena.nodes.size none
  schemaFactMemo := Array.replicate arena.nodes.size none
  relationMemo := Array.replicate arena.nodes.size none
}

private def mergeRelationRequirements
    (left right : RelationRequirement) : RelationRequirement :=
  match left, right with
  | .none, requirement | requirement, .none => requirement
  | .uniform leftSchema, .uniform rightSchema =>
      if leftSchema == rightSchema then .uniform leftSchema else .unknown
  | .branchLocal leftDomain, .branchLocal rightDomain =>
      if leftDomain == rightDomain then .branchLocal leftDomain else .unknown
  | .unknown, _ | _, .unknown | .uniform _, .branchLocal _ | .branchLocal _, .uniform _ => .unknown

private def mergeRelationRequirementArray
    (requirements : Array RelationRequirement) : RelationRequirement :=
  requirements.foldl mergeRelationRequirements .none

private def validateSharedRelationCorrelation
    (arena : OperationalExprArena)
    (operation : PrimitiveOperation)
    (domain : SelectionDomainId)
    (arguments : Array OperationalExprId)
    (expressions : Array OperationalMatrixExpr) : Except OperationalError Unit := do
  match operation.kind with
  | .multiply (.matrixMultiplyRelation _) _ =>
      let leftExpression ← match expressions[0]? with
        | some expression => pure expression
        | none => throw (.unsupportedOutputArity operation.ownerNode expressions.size)
      let rightExpression ← match expressions[1]? with
        | some expression => pure expression
        | none => throw (.unsupportedOutputArity operation.ownerNode expressions.size)
      match leftExpression.node, rightExpression.node with
      | .select leftDomain (.shared _ leftSchemaId),
          .select rightDomain (.shared _ rightSchemaId) => do
          if leftDomain != domain || rightDomain != domain then
            throw (.incompatibleRelationDomains operation.ownerNode
              leftDomain.ordinal rightDomain.ordinal)
          let leftSchema ← arena.validatedSchema leftSchemaId
          let rightSchema ← arena.validatedSchema rightSchemaId
          let expectedOrigin := some domain.kind
          if leftSchema.selectionOrigin != expectedOrigin ||
              rightSchema.selectionOrigin != expectedOrigin then
            throw (.incompatibleRelationDomains operation.ownerNode
              leftDomain.ordinal rightDomain.ordinal)
          let leftPublic ← match leftSchema.sharedLastPublicIdentity with
            | some identity => pure identity
            | none => throw (.unknownRelationRequirement operation.ownerNode arguments[0]!)
          let rightRelation ← match rightSchema.sharedFirstRelationPublicIdentity with
            | some identity => pure identity
            | none => throw (.unknownRelationRequirement operation.ownerNode arguments[1]!)
          if !publicIdentityTemplateEqual leftPublic rightRelation then
            throw (.incompatibleRelationDomains operation.ownerNode
              leftDomain.ordinal rightDomain.ordinal)
      | _, _ => pure ()
  | _ => pure ()

/-- Memoized structural query for the one selection domain required by a relation-bearing value.
It never evaluates bounds or scans unavailable Shared alternatives. A relation-consuming multiply
removes the right requirement only after the concrete relation rewrite succeeds; any relation
carried by its left coefficient remains visible. -/
private def relationRequirementWithFuel
    (arena : OperationalExprArena)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) : Nat →
    Except OperationalError (RelationRequirement × OperationalExprEvaluationState)
  | 0 => throw (.unknownRelationRequirement 0 id)
  | fuel + 1 => match state.relationMemo[id]? with
    | none => throw (.invalidOperationalExprRef id)
    | some (some requirement) => pure (requirement, {
        state with relationStats := {
          state.relationStats with memoHits := state.relationStats.memoHits + 1 } })
    | some none => do
        let expression ← match arena.get? id with
          | some expression => pure expression
          | none => throw (.invalidOperationalExprRef id)
        let mut state := { state with relationStats := {
          evaluations := state.relationStats.evaluations + 1
          memoHits := state.relationStats.memoHits
          memoMisses := state.relationStats.memoMisses + 1
        }}
        let queryChildren
            (state : OperationalExprEvaluationState)
            (children : Array OperationalExprId) := do
          let mut state := state
          let mut requirements : Array RelationRequirement := #[]
          for child in children do
            let (requirement, nextState) ← relationRequirementWithFuel arena child state fuel
            state := nextState
            requirements := requirements.push requirement
          pure (requirements, state)
        let requirement ← match expression.node with
          | .concrete fact =>
              if matrixFactHasRelation fact then
                pure (.uniform (operationalUniformSchema fact))
              else pure .none
          | .select domain (.shared _ schemaId) => do
              let schema ← arena.validatedSchema schemaId
              pure (if schema.relationFree then .none else .branchLocal domain)
          | .select domain (.exact branches) => do
              let (requirements, nextState) ← queryChildren state branches
              state := nextState
              if requirements.all (· == .none) then pure .none
              else if requirements.any (· == .unknown) then pure .unknown
              else pure (.branchLocal domain)
          | .primitive operation arguments => do
              let (requirements, nextState) ← queryChildren state arguments
              state := nextState
              match operation.kind with
              | .transform _ => pure (requirements[0]?.getD .unknown)
              | .add _ | .tensor | .concat _ =>
                  pure (mergeRelationRequirementArray requirements)
              | .multiply (.matrixMultiplyRelation _) _ =>
                  let left := requirements[0]?.getD .unknown
                  let right := requirements[1]?.getD .unknown
                  pure (if right == .none || right == .unknown then .unknown else left)
              | .multiply _ _ =>
                  pure (if requirements.all (· == .none) then .none else .unknown)
        let relationMemo := state.relationMemo.set! id (some requirement)
        pure (requirement, { state with relationMemo })

private def relationRequirement
    (arena : OperationalExprArena)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :=
  relationRequirementWithFuel arena id state (arena.nodes.size + 1)

/-- Transfer classes, rather than broad operation names, are the closed registry keys.  In
particular relation-consuming multiplication cannot inherit the ordinary multiplication row. -/
inductive PrimitiveTransferClass where
  | addSubtract
  | multiplyOrdinary
  | multiplyRelation
  | tensor
  | concat
  | transform
  deriving BEq, DecidableEq

inductive CompositionalTransfer where
  | supported (transfer : EnvelopeSummaryTransferOperation)
  | requiresConcreteStructure
  deriving BEq

private def primitiveTransferClass (operation : PrimitiveOperation) : PrimitiveTransferClass :=
  match operation.kind with
  | .add _ => .addSubtract
  | .multiply (.matrixMultiplyRelation _) _ => .multiplyRelation
  | .multiply _ _ => .multiplyOrdinary
  | .tensor => .tensor
  | .concat _ => .concat
  | .transform _ => .transform

/-- Closed registry used by generic choice lifting.  Every transfer-class constructor has exactly
one equation, so adding a class makes this definition and its inventory fixture non-exhaustive. -/
private def compositionalTransferRegistry : PrimitiveTransferClass → CompositionalTransfer
  | .addSubtract => .supported .addSubtract
  | .multiplyOrdinary => .requiresConcreteStructure
  | .multiplyRelation => .requiresConcreteStructure
  | .tensor => .requiresConcreteStructure
  | .concat => .requiresConcreteStructure
  | .transform => .supported .transform

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

private def replaceOperationalFactorHardBound
    (bound : OperationalBoundExpr)
    (factor : OperationalFactorKey) : OperationalFactorKey :=
  let update (summary : OperationalBoundedFactorSummary) := { summary with hardBound := bound }
  { factor with
    leaf := match factor.leaf with
      | .boundedSummary origin summary => .boundedSummary origin (update summary)
      | leaf => leaf
    boundedSummary := factor.boundedSummary.map update
  }

private def abstractCarriedMaximum (slot : Nat) : OperationalFact → OperationalFact
  | .matrix fact =>
      let maximum := OperationalBoundExpr.previous (.matrixMaximum 0 slot)
      .matrix {
        fact with
        totalHardBound := maximum
        polynomial := fact.polynomial.map fun term => { term with product := {
          term.product with
          factors := term.product.factors.map (replaceOperationalFactorHardBound maximum)
        }}
      }
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
    (polynomial : OperationalPolynomial) : Except OperationalError OperationalPolynomial := do
  let rec finishTerm : Nat → OperationalTerm →
      Except OperationalError OperationalPolynomial
    | 0, term => do
        match ← rewriteOperationalTermRelation? node term with
        | none => pure [term]
        | some _ => throw (.invalidMatrixParameters node)
    | fuel + 1, term => do
        match ← rewriteOperationalTermRelation? node term with
        | none => pure [term]
        | some rewritten => do
            let mut finished : OperationalPolynomial := []
            for generated in rewritten do
              finished := finished ++ (← finishTerm fuel generated)
            pure finished
  let mut finished : OperationalPolynomial := []
  for term in polynomial do
    finished := finished ++ (← finishTerm 64 term)
  pure (normalizeOperationalTerms finished)

private def sameConcreteMatrixShape (left right : Mxx.SamplerParams) : Bool :=
  left.modulus == right.modulus &&
    left.ringDimension == right.ringDimension &&
    left.rows == right.rows &&
    left.columns == right.columns

/-- Decide matrix-product compatibility from evaluated dimensions rather than the syntax of the
dimension expressions. This accepts equivalent forms such as `2` and `1 * 2`, while remaining
fail-closed when any type expression is not closed under the current parameter environment. -/
private def concreteMatrixProductMatches
    (leftType rightType outputType : MatrixTypeExpr)
    (environment : ParamEnvironment) : Bool :=
  match leftType.evaluate environment (.constant 0),
      rightType.evaluate environment (.constant 0),
      outputType.evaluate environment (.constant 0) with
  | some left, some right, some output =>
      left.modulus == right.modulus && left.modulus == output.modulus &&
        left.ringDimension == right.ringDimension &&
        left.ringDimension == output.ringDimension &&
        left.columns == right.rows && output.rows == left.rows &&
        output.columns == right.columns
  | _, _, _ => false

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

private def erasePrimitiveSelectionFactBounds
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let zero := OperationalBoundExpr.closedInt (.constant 0)
  { fact with
    matrixParams := { fact.matrixParams with maxCoefficientBound := 0 }
    totalHardBound := zero
    polynomial := (fact.polynomial.filter operationalTermIsSignal).map fun term => { term with
      product := { term.product with
        factors := term.product.factors.map (replaceOperationalFactorHardBound zero) } }
    metadata := {}
    identity := none
    relations := [] }

private def samePrimitiveSelectionShape
    (left right : OperationalMatrixFact) : Bool :=
  operationalUniformSchema (erasePrimitiveSelectionFactBounds left) ==
    operationalUniformSchema (erasePrimitiveSelectionFactBounds right)

private def maximumPrimitiveSelectionBound
    (first : OperationalBoundExpr)
    (remaining : List OperationalBoundExpr) : OperationalBoundExpr :=
  remaining.foldl OperationalBoundExpr.maximum first

/-- Close every complete mutually-exclusive primitive result before taking the branch maximum.
This is the selection-envelope analogue of the endpoint join below, but lives next to primitive
construction so later operations never need to carry an already relation-free family exactly. -/
private def summarizePrimitiveSelectionFacts
    (environment : ParamEnvironment)
    (facts : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let first ← match facts[0]? with
    | some first => pure first
    | none => throw (.invalidCount 0 0)
  if facts.any fun fact => fact.matrixType != first.matrixType ||
      !samePrimitiveSelectionShape first fact then
    throw (.unsupportedOperationalExpr 0)
  let firstSignal := first.polynomial.filter operationalTermIsSignal
  if facts.any fun fact =>
      (!firstSignal.isEmpty &&
          fact.matrixParams.maxCoefficientBound != first.matrixParams.maxCoefficientBound) then
    -- Signal identities and values are intentionally branch-local. `samePrimitiveSelectionShape`
    -- has already proved that their complete identity-erased schemas agree, and the envelope keeps
    -- the selection identity needed to interpret the representative as the selected signal.
    throw (.unsupportedOperationalExpr 0)
  let checkedSummary := selectedMatrixSummary facts
  if facts.any matrixFactHasRelation then
    if checkedSummary.uniformSchema.isNone ||
        checkedSummary.sharedFirstRelationPublicIdentity.isNone then
      throw (.unsupportedOperationalExpr 0)
    else
      pure first
  else
    if (boundaryLastPublicIdentity? first).isSome &&
        checkedSummary.sharedLastPublicIdentity.isNone then
      -- A representative must not manufacture a branch-universal public boundary. Keep the
      -- exact selection whenever the alternatives agree structurally but carry different public
      -- identities; a later relation-sensitive operation must inspect the actual branch.
      throw (.unsupportedOperationalExpr 0)
    let noiseSummaries ← facts.mapM fun fact =>
      fact.noiseHardBound.mapError fun _ => .invalidMatrixParameters fact.subject.node
    let noiseBound ← match noiseSummaries[0]? with
      | some firstBound =>
          pure (maximumPrimitiveSelectionBound firstBound noiseSummaries.toList.tail)
      | none => throw (.invalidCount 0 0)
    let branchMetadata := facts.map (·.metadata)
    let metadata : OperationalMatrixMetadata := {
      isConstantPolynomial := branchMetadata.all (·.isConstantPolynomial)
      knownZeroRows := match branchMetadata[0]? with
        | some value =>
            if branchMetadata.all (·.knownZeroRows == value.knownZeroRows) then
              value.knownZeroRows
            else none
        | none => none
    }
    let signal := firstSignal
    let noise := if noiseBound == .closedInt (.constant 0) then [] else
      let tokens := [.sumStart, .summaryBound noiseBound, .summaryMetadata metadata, .sumEnd]
      let summary : OperationalBoundedFactorSummary := {
        matrixType := first.matrixType
        hardBound := noiseBound
        metadata
        provenance := tokens
      }
      let origin : OperationalCompressionOrigin := {
        kind := .boundedNoiseSum
        tokens
      }
      let factor : OperationalFactorKey := {
        leaf := .boundedSummary origin summary
        inputType := first.matrixType
        outputType := first.matrixType
        role := .bounded
        boundedSummary := some summary
      }
      [{ coefficient := 1, product := {
          factors := [factor], modes := [], outputType := first.matrixType } }]
    let output ← polynomialMatrixFact first.subject.node first.subject.port first.matrixType
      environment (signal ++ noise) first.canonicalRange
    match output with
    | .matrix fact => pure { fact with
        subject := first.subject
        origin := first.origin
        identity := if facts.all (·.identity == first.identity) then first.identity else none }
    | _ => throw (.operandNotMatrix first.subject.node first.subject)

/-- Build the result of pushing a primitive through an exact selection. Graph IR dimension
expressions are compared by evaluated shape.  Complete relation-free branches are joined by their
full branch maximum; relation-bearing branches use an envelope only when every branch proves the
same first relation boundary. Otherwise their exact identities remain available downstream. -/
private def OperationalExprArena.pushPrimitiveSelection
    {α : Type} [SelectionIdentityLike α]
    (arena : OperationalExprArena)
    (selection : α)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (branches : Array OperationalExprId) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let expectedParams ← match matrixType.evaluate environment (.constant 0) with
    | some params => pure params
    | none => throw (.invalidMatrixParameters 0)
  let first ← match branches[0]? with
    | some first => pure first
    | none => throw (.invalidCount 0 0)
  for branch in branches do
    let expression ← match arena.get? branch with
      | some expression => pure expression
      | none => throw (.invalidOperationalExprRef branch)
    let actualParams ← match expression.matrixType.evaluate environment (.constant 0) with
      | some params => pure params
      | none => throw (.invalidMatrixParameters 0)
    if !sameConcreteMatrixShape actualParams expectedParams then
      throw (.operationalExprTypeMismatch first branch)
  let firstExpression ← match arena.get? first with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef first)
  if branches.all (· == first) && firstExpression.matrixType == matrixType then
    pure (arena, first)
  else do
    let concreteFacts? := branches.mapM fun branch => do
      let expression ← arena.get? branch
      match expression.node with
      | .concrete fact => some fact
      | _ => none
    match concreteFacts? with
    | some concreteFacts =>
        match summarizePrimitiveSelectionFacts environment concreteFacts with
        | .ok representativeFact =>
            let summary := selectedMatrixSummary #[representativeFact]
            let (arena, representative) := arena.pushConcrete representativeFact
            arena.pushCheckedSchemaEnvelope selection branches.size representative summary
              representativeFact
        | .error _ =>
            arena.pushSelect selection (.exact branches)
    | none =>
        arena.pushSelect selection (.exact branches)

/-- Preserve a strict canonical coefficient range through ordinary matrix multiplication when
both inputs are known constant-polynomial matrices.  In that case no negacyclic convolution term
can wrap a negative coefficient to a residue near the modulus: every output coefficient is a sum
of `left.columns` nonnegative scalar products.  For general polynomial inputs the quotient-ring
signs make any sub-modulus range unsafe, so the result remains unknown. -/
private def constantPolynomialProductCanonicalRange
    (left right : OperationalMatrixFact) : CanonicalRange :=
  match left.canonicalRange, right.canonicalRange with
  | .below leftUpper, .below rightUpper =>
      if left.metadata.isConstantPolynomial && right.metadata.isConstantPolynomial &&
          left.matrixParams.modulus > 0 &&
          left.matrixParams.modulus == right.matrixParams.modulus &&
          left.matrixParams.columns == right.matrixParams.rows then
        let unreducedUpper :=
          left.matrixParams.columns * (leftUpper - 1) * (rightUpper - 1) + 1
        .below (min left.matrixParams.modulus.toNat unreducedUpper)
      else .unknown
  | _, _ => .unknown

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
  match ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
      (constantPolynomialProductCanonicalRange left right) with
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

/-- Deterministic one-domain lifting shared by all primitive operations.  It inspects only
immediate Choice arguments, chooses the first domain in argument order, and never constructs or
visits a Cartesian product.  Independent domains remain nested below one delayed Primitive. -/
private partial def liftPrimitiveOperation
    (operation : PrimitiveOperation)
    (summaryOperation : EnvelopeSummaryTransferOperation)
    (concreteTransfer : Array OperationalMatrixFact →
      Except OperationalError OperationalMatrixFact)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState)) :
    OperationalExprArena → Array OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, arguments, 0 => throw (.unsupportedOperationalExpr (arguments[0]?.getD 0))
  | arena, arguments, _fuel + 1 => do
      let expressions ← arguments.mapM fun argument => match arena.get? argument with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef argument)
      let pushImmediate
          (arena : OperationalExprArena)
          (arguments : Array OperationalExprId) := do
        let argumentExpressions ← arguments.mapM fun argument => match arena.get? argument with
          | some expression => pure expression
          | none => throw (.invalidOperationalExprRef argument)
        match operation.kind with
        | .multiply _ _ =>
            let left ← match argumentExpressions[0]? with
              | some expression => pure expression
              | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
            let right ← match argumentExpressions[1]? with
              | some expression => pure expression
              | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
            if arguments.size != 2 || !concreteMatrixProductMatches left.matrixType
                right.matrixType operation.outputType operation.parameterEnvironment then
              throw (.operationalExprTypeMismatch
                (arguments[0]?.getD 0) (arguments[1]?.getD 0))
        | _ => pure ()
        let facts? := argumentExpressions.map fun expression => match expression.node with
          | .concrete fact => some fact
          | _ => none
        if facts?.all Option.isSome then
          let facts := facts?.filterMap id
          let output ← concreteTransfer facts
          let arena := if primitiveTransferClass operation == .multiplyRelation then
            { arena with relationRewriteCount := arena.relationRewriteCount + 1 }
          else arena
          pure (arena.pushConcrete output)
        else
          pure (arena.pushPrimitive operation.ownerNode operation.outputPort operation.outputType
            operation.parameterEnvironment operation.kind arguments)
      let choices := expressions.zipIdx.filterMap fun (expression, index) =>
        match expression.node with
        | .select domain storage => some (index, domain, storage)
        | _ => none
      let firstChoice ← match choices[0]? with
        | some choice => pure choice
        | none => return ← pushImmediate arena arguments
      let (_, firstDomain, _) := firstChoice
      let relationDomain? ← match operation.kind with
        | .multiply (.matrixMultiplyRelation _) _ => do
            let right ← match arguments[1]? with
              | some right => pure right
              | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
            let (requirement, _) ← relationRequirement arena right
              (OperationalExprEvaluationState.empty arena)
            match requirement with
            | .branchLocal domain => pure (some domain)
            | .uniform _ | .none => pure none
            | .unknown => throw (.unknownRelationRequirement operation.ownerNode right)
        | _ => pure none
      let domain := relationDomain?.getD firstDomain
      if relationDomain?.isSome && !choices.any (fun (_, candidate, _) => candidate == domain) then
        return ← pushImmediate arena arguments
      let hasIndependentDomain := choices.any fun (_, candidate, _) => candidate != domain
      if hasIndependentDomain then
        match compositionalTransferRegistry (primitiveTransferClass operation) with
        | .supported _ => return ← pushImmediate arena arguments
        | .requiresConcreteStructure =>
            match operation.kind, relationDomain? with
            | .multiply (.matrixMultiplyRelation _) _, some _ => pure ()
            | _, _ => return ← pushImmediate arena arguments
      let matching := choices.filter fun (_, candidate, _) => candidate == domain
      let hasExact := matching.any fun (_, _, storage) => match storage with
        | .exact _ => true
        | .shared .. => false
      if hasExact then
        let mut arena := arena
        let mut outputs : Array OperationalExprId := #[]
        for branch in [:domain.count] do
          let branchArguments ← arguments.zip expressions |>.mapM fun (argument, expression) =>
            match expression.node with
            | .select candidate (.exact branches) =>
                if candidate == domain then match branches[branch]? with
                  | some value => pure value
                  | none => throw (.operationalExprTypeMismatch argument arguments[0]!)
                else pure argument
            | .select candidate (.shared representative _) =>
                if candidate == domain then pure representative else pure argument
            | _ => pure argument
          let (nextArena, output) ← pushImmediate arena branchArguments
          arena := nextArena
          outputs := outputs.push output
        arena.pushPrimitiveSelection domain operation.outputType operation.parameterEnvironment outputs
      else
        validateSharedRelationCorrelation arena operation domain arguments expressions
        let representativeArguments := arguments.zip expressions |>.map fun (argument, expression) =>
          match expression.node with
          | .select candidate (.shared representative _) =>
              if candidate == domain then representative else argument
          | _ => argument
        let (arena, output) ← pushImmediate arena representativeArguments
        let state := OperationalExprEvaluationState.empty arena
        let (outputFact, _) ← evaluateRepresentative arena operation.parameterEnvironment output state
        let schemaIds := matching.filterMap fun (_, _, storage) => match storage with
          | .shared _ schema => some schema
          | .exact _ => none
        let summaries ← schemaIds.mapM arena.validatedSchema
        let outputSummary ← match transferSelectedMatrixSummary summaryOperation summaries outputFact with
          | some summary => pure summary
          | none => throw (.unsupportedOperationalExpr output)
        arena.pushCheckedSchemaEnvelope domain domain.count output outputSummary outputFact

private def addOperationalExprIds
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (subtract : Bool)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let operation : PrimitiveOperation := {
    kind := .add subtract
    outputType := matrixType
    ownerScope := arena.activeScope
    ownerNode := nodeIndex
    outputPort
    parameterEnvironment := environment
  }
  let concreteTransfer (arguments : Array OperationalMatrixFact) := do
    let leftFact ← match arguments[0]? with
      | some fact => pure fact
      | none => throw (.unsupportedOutputArity nodeIndex arguments.size)
    let rightFact ← match arguments[1]? with
      | some fact => pure fact
      | none => throw (.unsupportedOutputArity nodeIndex arguments.size)
    if arguments.size != 2 then throw (.unsupportedOutputArity nodeIndex arguments.size)
    addConcreteMatrixFacts nodeIndex outputPort matrixType subtract environment leftFact rightFact
  liftPrimitiveOperation operation .addSubtract concreteTransfer evaluateRepresentative arena
    #[left, right] fuel

private def addOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (subtract : Bool)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, leftId) ← arena.pushMatrixFact left
  let (arena, rightId) ← arena.pushMatrixFact right
  let (arena, result) ← addOperationalExprIds nodeIndex outputPort matrixType subtract environment
    evaluateRepresentative arena leftId rightId (arena.nodes.size + 1)
  pure (arena, .matrixExpr result)

private def multiplyOperationalExprIds
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let operation : PrimitiveOperation := {
    kind := .multiply rule rightWire
    outputType := matrixType
    ownerScope := arena.activeScope
    ownerNode := nodeIndex
    outputPort
    parameterEnvironment := environment
  }
  let concreteTransfer (arguments : Array OperationalMatrixFact) := do
    let leftFact ← match arguments[0]? with
      | some fact => pure fact
      | none => throw (.unsupportedOutputArity nodeIndex arguments.size)
    let rightFact ← match arguments[1]? with
      | some fact => pure fact
      | none => throw (.unsupportedOutputArity nodeIndex arguments.size)
    if arguments.size != 2 then throw (.unsupportedOutputArity nodeIndex arguments.size)
    multiplyConcreteMatrixFacts nodeIndex outputPort matrixType rule rightWire environment
      leftFact rightFact
  let summaryOperation := match rule with
    | .matrixMultiplyRelation _ => EnvelopeSummaryTransferOperation.multiplyRelation
    | _ => .unregistered
  liftPrimitiveOperation operation summaryOperation concreteTransfer evaluateRepresentative arena
    #[left, right] fuel

private def multiplyOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, leftId) ← arena.pushMatrixFact left
  let (arena, rightId) ← arena.pushMatrixFact right
  let (arena, result) ← multiplyOperationalExprIds nodeIndex outputPort matrixType rule rightWire
    environment evaluateRepresentative arena leftId rightId (arena.nodes.size + 1)
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
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let operation : PrimitiveOperation := {
    kind := .tensor
    outputType := matrixType
    ownerScope := arena.activeScope
    ownerNode := nodeIndex
    outputPort
    parameterEnvironment := environment
  }
  let concreteTransfer (arguments : Array OperationalMatrixFact) := do
    let leftFact ← match arguments[0]? with
      | some fact => pure fact
      | none => throw (.unsupportedOutputArity nodeIndex arguments.size)
    let rightFact ← match arguments[1]? with
      | some fact => pure fact
      | none => throw (.unsupportedOutputArity nodeIndex arguments.size)
    if arguments.size != 2 then throw (.unsupportedOutputArity nodeIndex arguments.size)
    tensorConcreteMatrixFacts nodeIndex outputPort matrixType environment leftFact rightFact
  liftPrimitiveOperation operation .tensor concreteTransfer evaluateRepresentative arena
    #[left, right] fuel

private def tensorOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, leftId) ← arena.pushMatrixFact left
  let (arena, rightId) ← arena.pushMatrixFact right
  let (arena, result) ← tensorOperationalExprIds nodeIndex outputPort matrixType environment
    evaluateRepresentative arena leftId rightId (arena.nodes.size + 1)
  pure (arena, .matrixExpr result)

private structure OperationalExprTransformMemo where
  outputs : Std.HashMap OperationalExprId OperationalExprId := {}
  hits : Nat := 0
  misses : Nat := 0

private def mapOperationalExprWithFuelCached
    (summaryOperation : EnvelopeSummaryTransferOperation)
    (mapFact : OperationalMatrixFact → Except OperationalError OperationalMatrixFact)
    (mapSelection : DynamicSelectionIdentity → DynamicSelectionIdentity)
    (ownerFilter : Option (ScopeTemplateKey × Nat))
    (structuralSummaryMap : Option (SelectedMatrixSummary → SelectedMatrixSummary)) :
    OperationalExprArena → OperationalExprTransformMemo → OperationalExprId → Nat →
    Except OperationalError
      (OperationalExprArena × OperationalExprTransformMemo × OperationalExprId)
  | _, _, root, 0 => throw (.unsupportedOperationalExpr root)
  | arena, memo, root, fuel + 1 => do
      match memo.outputs[root]? with
      | some mapped =>
          return (arena, { memo with hits := memo.hits + 1 }, mapped)
      | none => pure ()
      let expression ← match arena.get? root with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef root)
      match ownerFilter with
      | some (ownerScope, ownerNode) =>
          if expression.ownerScope != some ownerScope || expression.ownerNode != some ownerNode then
            return (arena, { memo with outputs := memo.outputs.insert root root }, root)
      | none => pure ()
      let memo := { memo with misses := memo.misses + 1 }
      let (arena, memo, output) ← match expression.node with
      | .concrete fact =>
          let mapped ← mapFact fact
          if mapped == fact then pure (arena, memo, root)
          else
            let (arena, output) := arena.pushConcrete mapped
            pure (arena, memo, output)
      | .primitive operation arguments =>
          let mut arena := arena
          let mut memo := memo
          let mut mappedArguments : Array OperationalExprId := #[]
          for argument in arguments do
            let (nextArena, nextMemo, mapped) ← mapOperationalExprWithFuelCached
              summaryOperation mapFact mapSelection ownerFilter structuralSummaryMap
                arena memo argument fuel
            arena := nextArena
            memo := nextMemo
            mappedArguments := mappedArguments.push mapped
          if mappedArguments == arguments then pure (arena, memo, root)
          else
            let (nextArena, output) := arena.push {
              expression with node := .primitive operation mappedArguments }
            pure (nextArena, memo, output)
      | .select selection (.exact branches) =>
          let mut arena := arena
          let mut memo := memo
          let mut mappedBranches : Array OperationalExprId := #[]
          for branch in branches do
            let (nextArena, nextMemo, mapped) ← mapOperationalExprWithFuelCached
              summaryOperation mapFact mapSelection ownerFilter structuralSummaryMap
                arena memo branch fuel
            arena := nextArena
            memo := nextMemo
            mappedBranches := mappedBranches.push mapped
          let mappedSelection := mapSelection selection.identity
          if mappedSelection == selection.identity && mappedBranches == branches then
            pure (arena, memo, root)
          else
            let (nextArena, output) ←
              arena.pushSelect mappedSelection (.exact mappedBranches)
            pure (nextArena, memo, output)
      | .select selection (.shared representative summary) =>
          let summary ← arena.validatedSchema summary
          let (arena, memo, mapped) ← mapOperationalExprWithFuelCached
            summaryOperation mapFact mapSelection ownerFilter structuralSummaryMap
              arena memo representative fuel
          let mappedSummary ← match arena.get? mapped with
            | some { node := .concrete mappedFact, .. } =>
                match transferSelectedMatrixSummary summaryOperation #[summary] mappedFact with
                | some value => pure value
                | none => throw (.unsupportedOperationalExpr representative)
            | some _ => match structuralSummaryMap with
                | some mapSummary => pure (mapSummary summary)
                | none => throw (.unsupportedOperationalExpr representative)
            | none => throw (.invalidOperationalExprRef mapped)
          let mappedSelection := mapSelection selection.identity
          if mapped == representative && mappedSelection == selection.identity &&
              mappedSummary == summary then
            pure (arena, memo, root)
          else
            let (arena, output) ← arena.pushSharedSelection mappedSelection selection.count mapped
              mappedSummary
            pure (arena, memo, output)
      pure (arena, { memo with outputs := memo.outputs.insert root output }, output)

private def mapOperationalExprWithFuel
    (_cacheNamespace : String)
    (summaryOperation : EnvelopeSummaryTransferOperation)
    (mapFact : OperationalMatrixFact → Except OperationalError OperationalMatrixFact)
    (mapSelection : DynamicSelectionIdentity → DynamicSelectionIdentity)
    (ownerFilter : Option (ScopeTemplateKey × Nat))
    (structuralSummaryMap : Option (SelectedMatrixSummary → SelectedMatrixSummary))
    (arena : OperationalExprArena)
    (root : OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let memo : OperationalExprTransformMemo := {}
  let (arena, memo, output) ← mapOperationalExprWithFuelCached
    summaryOperation mapFact mapSelection ownerFilter structuralSummaryMap arena memo root fuel
  pure ({ arena with
    transformCacheHits := arena.transformCacheHits + memo.hits
    transformCacheMisses := arena.transformCacheMisses + memo.misses
  }, output)

private def mapOperationalExprM
    (cacheNamespace : String)
    (summaryOperation : EnvelopeSummaryTransferOperation)
    (arena : OperationalExprArena)
    (root : OperationalExprId)
    (mapFact : OperationalMatrixFact → Except OperationalError OperationalMatrixFact)
    (mapSelection : DynamicSelectionIdentity → DynamicSelectionIdentity := id)
    (ownerFilter : Option (ScopeTemplateKey × Nat) := none)
    (structuralSummaryMap : Option (SelectedMatrixSummary → SelectedMatrixSummary) := none) :
    Except OperationalError (OperationalExprArena × OperationalExprId) :=
  mapOperationalExprWithFuel cacheNamespace summaryOperation mapFact mapSelection ownerFilter
    structuralSummaryMap arena root (arena.nodes.size + 1)

private def mapOperationalExpr
    (cacheNamespace : String)
    (summaryOperation : EnvelopeSummaryTransferOperation)
    (arena : OperationalExprArena)
    (root : OperationalExprId)
    (mapFact : OperationalMatrixFact → OperationalMatrixFact)
    (mapSelection : DynamicSelectionIdentity → DynamicSelectionIdentity := id)
    (ownerFilter : Option (ScopeTemplateKey × Nat) := none)
    (structuralSummaryMap : Option (SelectedMatrixSummary → SelectedMatrixSummary) := none) :
    Except OperationalError (OperationalExprArena × OperationalExprId) :=
  mapOperationalExprM cacheNamespace summaryOperation arena root
    (fun fact => pure (mapFact fact)) mapSelection ownerFilter structuralSummaryMap

private def isLoopTemplateSelection
    (binder : FamilyTemplateBinder)
    (origin : OperationalValueOrigin) : Bool :=
  let base : OperationalValueOrigin :=
    .local temporaryScope { node := binder.producerNode, port := 0 }
  let namespacedBase : OperationalValueOrigin :=
    .local binder.owner { node := binder.producerNode, port := 0 }
  let rec containsBase : OperationalValueOrigin → Bool
    | candidate@(.local ..) => candidate == base || candidate == namespacedBase
    | .loopInstance _ _ source => containsBase source
    | _ => false
  containsBase origin

private def loopTemplateStaticRoot
    (arena : OperationalExprArena)
    (binder : FamilyTemplateBinder)
    (root : OperationalExprId)
    (lane : Nat) : Except OperationalError OperationalExprId := do
  let expression ← match arena.get? root with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef root)
  match expression.node with
  | .select selection (.exact branches) =>
      if isLoopTemplateSelection binder selection.index then
        match branches[lane]? with
        | some branch => pure branch
        | none => throw (.invalidCount binder.producerNode lane)
      else pure root
  | .select selection (.shared representative _) =>
      if isLoopTemplateSelection binder selection.index then
        if lane < selection.count then pure representative
        else throw (.invalidCount binder.producerNode lane)
      else pure root
  | _ => pure root

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
    (transform : OperationalFactorTransform)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (root : OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let operation : PrimitiveOperation := {
    kind := .transform transform
    outputType := matrixType
    ownerScope := arena.activeScope
    ownerNode := nodeIndex
    outputPort
    parameterEnvironment := environment
  }
  let concreteTransfer (arguments : Array OperationalMatrixFact) := do
    let input ← match arguments[0]? with
      | some fact => pure fact
      | none => throw (.unsupportedOutputArity nodeIndex arguments.size)
    if arguments.size != 1 then throw (.unsupportedOutputArity nodeIndex arguments.size)
    transformConcreteMatrixFact nodeIndex outputPort matrixType transform environment input
  liftPrimitiveOperation operation .transform concreteTransfer evaluateRepresentative arena #[root] fuel

private def transformOperationalExprFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (operation : OperationalFactorTransform)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (input : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, root) ← arena.pushMatrixFact input
  let (arena, result) ← transformOperationalExprId nodeIndex outputPort matrixType operation
    environment evaluateRepresentative arena root (arena.nodes.size + 1)
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
    (loopDomains : List OperationalParameterDomain)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState)) :
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
              scalarValues environment loopDomains evaluateRepresentative arena branch fuel
            arena := nextArena
            outputs := outputs.push output
          arena.pushPrimitiveSelection selection matrixType environment outputs
      | .select selection (.shared representative summary) =>
          let summary ← arena.validatedSchema summary
          let (arena, output) ← scaleOperationalExprId nodeIndex outputPort matrixType scalar
            scalarValues environment loopDomains evaluateRepresentative arena representative fuel
          let state := OperationalExprEvaluationState.empty arena
          let (outputFact, _) ← evaluateRepresentative arena environment output state
          let outputSummary ← match transferSelectedMatrixSummary .scale #[summary] outputFact with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr representative)
          arena.pushCheckedSchemaEnvelope selection selection.count output outputSummary outputFact
      | _ => throw (.unsupportedOperationalExpr root)

private def scaleOperationalExprFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (scalar : IntExpr)
    (scalarValues : List Int)
    (environment : ParamEnvironment)
    (loopDomains : List OperationalParameterDomain)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (input : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, root) ← arena.pushMatrixFact input
  let (arena, result) ← scaleOperationalExprId nodeIndex outputPort matrixType scalar scalarValues
    environment loopDomains evaluateRepresentative arena root (arena.nodes.size + 1)
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

private def groupBggEncodingExprIds
    (environment : ParamEnvironment) :
    OperationalExprArena → OperationalExprId → OperationalExprId → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, vector, _, _, 0 => throw (.unsupportedOperationalExpr vector)
  | arena, vector, publicKey, plaintext, fuel + 1 => do
      let vectorExpr ← match arena.get? vector with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef vector)
      let publicKeyExpr ← match arena.get? publicKey with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef publicKey)
      let plaintextExpr ← match arena.get? plaintext with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef plaintext)
      match vectorExpr.node, publicKeyExpr.node, plaintextExpr.node with
      | .concrete vectorFact, .concrete publicKeyFact, .concrete plaintextFact =>
          let grouped ← groupBggEncodingSignal vectorFact publicKeyFact plaintextFact
            |>.mapError (.flat vectorFact.subject.node)
          pure (arena.pushConcrete grouped)
      | _, _, _ =>
          let selected := [vectorExpr.node, publicKeyExpr.node, plaintextExpr.node].filterMap
            fun node => match node with
              | .select selection branches => some (selection, branches)
              | _ => none
          let (selection, branches) ← match selected.head? with
            | some selected => pure selected
            | none => throw (.unsupportedOperationalExpr vector)
          if selected.any fun candidate => candidate.1 != selection then
            throw (.operationalExprTypeMismatch vector publicKey)
          let branchFor
              (node : OperationalMatrixExprNode)
              (ordinary : OperationalExprId)
              (index : Nat) : Except OperationalError OperationalExprId :=
            match node with
            | .concrete _ => pure ordinary
            | .select candidate (.exact values) =>
                if candidate != selection then
                  throw (.operationalExprTypeMismatch ordinary vector)
                else match values[index]? with
                  | some value => pure value
                  | none => throw (.operationalExprTypeMismatch ordinary vector)
            | _ => throw (.unsupportedOperationalExpr ordinary)
          match branches with
          | .exact values =>
              if values.isEmpty then throw (.invalidCount 0 0)
              let selectedCountsAgree := selected.all fun candidate => match candidate.2 with
                | .exact candidates => candidates.size == values.size
                | .shared .. => false
              if !selectedCountsAgree then
                throw (.operationalExprTypeMismatch vector publicKey)
              let mut arena := arena
              let mut outputs : Array OperationalExprId := #[]
              for branch in [:values.size] do
                let vectorBranch ← branchFor vectorExpr.node vector branch
                let publicKeyBranch ← branchFor publicKeyExpr.node publicKey branch
                let plaintextBranch ← branchFor plaintextExpr.node plaintext branch
                let (nextArena, output) ← groupBggEncodingExprIds environment arena vectorBranch
                  publicKeyBranch plaintextBranch fuel
                arena := nextArena
                outputs := outputs.push output
              arena.pushSelect selection (.exact outputs)
          | .shared representative _ =>
              let sourceSummaryIds := selected.filterMap fun candidate => match candidate.2 with
                | .shared _ candidateSummary => some candidateSummary
                | .exact _ => none
              let sourceSummaries ← sourceSummaryIds.mapM arena.validatedSchema
              if sourceSummaries.length != selected.length then
                throw (.unsupportedOperationalExpr representative)
              let representativeFor
                  (node : OperationalMatrixExprNode)
                  (ordinary : OperationalExprId) : Except OperationalError OperationalExprId :=
                match node with
                | .concrete _ => pure ordinary
                | .select candidateSelection
                    (.shared candidateRepresentative _) =>
                    if candidateSelection != selection then
                      throw (.operationalExprTypeMismatch ordinary vector)
                    else pure candidateRepresentative
                | _ => throw (.unsupportedOperationalExpr ordinary)
              let vectorRepresentative ← representativeFor vectorExpr.node vector
              let publicKeyRepresentative ← representativeFor publicKeyExpr.node publicKey
              let plaintextRepresentative ← representativeFor plaintextExpr.node plaintext
              let (arena, output) ← groupBggEncodingExprIds environment arena vectorRepresentative
                publicKeyRepresentative plaintextRepresentative fuel
              let outputFact ← arena.concreteFact output
              let outputSummary ← match transferSelectedMatrixSummary .bggGrouping
                  sourceSummaries.toArray outputFact with
                | some outputSummary => pure outputSummary
                | none => throw (.unsupportedOperationalExpr representative)
              arena.pushCheckedSchemaEnvelope selection selection.count output outputSummary outputFact

private partial def groupBggEncodingOperationalFacts
    (environment : ParamEnvironment)
    (arena : OperationalExprArena) :
    OperationalFact → OperationalFact → OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | .familyUniform binder coordinate vector vectorCount,
      .familyUniform _ _ publicKey publicCount,
      .familyUniform _ _ plaintext plaintextCount => do
      if vectorCount != publicCount || vectorCount != plaintextCount then
        throw (.invalidDerivationAttachment "mxx-bgg" "encoding-family-pairing")
      let (arena, grouped) ← groupBggEncodingOperationalFacts environment arena
        vector publicKey plaintext
      pure (arena, .familyUniform binder coordinate grouped vectorCount)
  | vector, publicKey, plaintext => do
      let (arena, vector) ← arena.pushMatrixFact vector
      let (arena, publicKey) ← arena.pushMatrixFact publicKey
      let (arena, plaintext) ← arena.pushMatrixFact plaintext
      let (arena, grouped) ← groupBggEncodingExprIds environment arena vector publicKey plaintext
        (arena.nodes.size + 1)
      pure (arena, .matrixExpr grouped)

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
    (environment : ParamEnvironment)
    (facts : OperationalScopeFacts) : Except OperationalError OperationalScopeFacts := do
  if attachment.ownerNamespace == "mxx-bgg" &&
      attachment.ruleName == "encoding-family-pairing" then
    let vectorWire ← derivationAttachmentRole attachment "vector"
    let publicKeyWire ← derivationAttachmentRole attachment "public-key"
    let plaintextWire ← derivationAttachmentRole attachment "plaintext"
    let vector ← lookupFact node facts vectorWire
    let publicKey ← lookupFact node facts publicKeyWire
    let plaintext ← lookupFact node facts plaintextWire
    let (arena, grouped) ← groupBggEncodingOperationalFacts environment facts.arena
      vector publicKey plaintext
    replaceOperationalFact node { facts with arena } vectorWire grouped
  else if attachment.ownerNamespace == "mxx-correctness" &&
      attachment.ruleName == "protocol-boolean-signal-grouping" then
    let valueWire ← derivationAttachmentRole attachment "value"
    let value ← lookupFact node facts valueWire
    match value with
    | .matrixExpr root =>
        let mapFact (fact : OperationalMatrixFact) := do
          match ← groupProtocolBooleanSignalFact (.matrix fact) with
          | .matrix grouped => pure grouped
          | _ => throw (.invalidDerivationAttachment attachment.ownerNamespace
              attachment.ruleName)
        let cacheNamespace := s!"protocol-boolean-group:{node}:{valueWire.node}:{valueWire.port}"
        let (arena, grouped) ←
          mapOperationalExprM cacheNamespace .bggGrouping facts.arena root mapFact
        replaceOperationalFact node { facts with arena } valueWire (.matrixExpr grouped)
    | value =>
        let grouped ← groupProtocolBooleanSignalFact value
        replaceOperationalFact node facts valueWire grouped
  else
    let valueWire ← derivationAttachmentRole attachment "value"
    let value ← lookupFact node facts valueWire
    match value with
    | .matrixExpr root =>
        let mapFact (fact : OperationalMatrixFact) := do
          match ← groupPublicKeySignalFact (.matrix fact) with
          | .matrix grouped => pure grouped
          | _ => throw (.invalidDerivationAttachment attachment.ownerNamespace
              attachment.ruleName)
        let cacheNamespace := s!"public-key-group:{node}:{valueWire.node}:{valueWire.port}"
        let (arena, grouped) ←
          mapOperationalExprM cacheNamespace .bggGrouping facts.arena root mapFact
        replaceOperationalFact node { facts with arena } valueWire (.matrixExpr grouped)
    | value =>
        let grouped ← groupPublicKeySignalFact value
        replaceOperationalFact node facts valueWire grouped

private def applyPreparedDerivationAttachments
    (node : Nat)
    (attachments : Array DerivationAttachment)
    (environment : ParamEnvironment)
    (facts : OperationalScopeFacts) : Except OperationalError OperationalScopeFacts :=
  attachments.foldlM (init := facts) fun current attachment =>
    applyDerivationAttachment node attachment environment current

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
  | .familyPacked elements count summary => do
      let elements ← elements.mapM (rebindSubject subject)
      return transferPackedOperationalFamily .instantiationMap elements count summary
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

private def namespaceFreshSelectedMatrixSummary
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (summary : SelectedMatrixSummary) : SelectedMatrixSummary := {
  summary with
  sharedLastPublicIdentity := summary.sharedLastPublicIdentity.map
    (namespaceFreshPublicIdentity scope wire)
  sharedFirstRelationPublicIdentity := summary.sharedFirstRelationPublicIdentity.map
    (namespaceFreshPublicIdentity scope wire)
}

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
  | .familyPacked elements count summary =>
      transferPackedOperationalFamily .recurrenceBoundShift
        (elements.map shiftFactPreviousDepth) count summary
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
  | fact => fact

/-- Namespace the concrete leaves created by one output without rebuilding its expression DAG.
The expression arena is analysis-local and the selected owner has not yet been published in
`OperationalScopeFacts`, so updating those exact owner nodes in place preserves sharing while
leaving every imported subtree untouched. -/
private def namespaceOperationalExprInPlace
    (scope : ScopeTemplateKey)
    (wire : WireRef) :
    OperationalExprArena → Std.HashMap OperationalExprId Bool → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × Std.HashMap OperationalExprId Bool)
  | _, _visited, root, 0 => throw (.unsupportedOperationalExpr root)
  | arena, visited, root, fuel + 1 => do
      if visited.contains root then return (arena, visited)
      let visited := visited.insert root true
      let expression ← match arena.get? root with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef root)
      if expression.ownerScope != some scope || expression.ownerNode != some wire.node then
        return (arena, visited)
      match expression.node with
      | .concrete fact =>
          let mapped ← match namespaceFreshOutput scope wire (.matrix fact) with
            | .matrix mapped => pure mapped
            | _ => throw (.unsupportedOperationalExpr root)
          pure ({ arena with nodes := arena.nodes.set! root {
            expression with node := .concrete mapped
          } }, visited)
      | .primitive operation arguments =>
          let mut arena := arena
          let mut visited := visited
          for argument in arguments do
            let (nextArena, nextVisited) ←
              namespaceOperationalExprInPlace scope wire arena visited argument fuel
            arena := nextArena
            visited := nextVisited
          pure ({ arena with nodes := arena.nodes.set! root {
            expression with node := .primitive { operation with ownerScope := some scope } arguments
          } }, visited)
      | .select selection (.exact branches) =>
          let mut arena := arena
          let mut visited := visited
          for branch in branches do
            let (nextArena, nextVisited) ←
              namespaceOperationalExprInPlace scope wire arena visited branch fuel
            arena := nextArena
            visited := nextVisited
          let mappedSelection : DynamicSelectionIdentity := {
            index := namespaceFreshValueOrigin scope wire selection.index
          }
          let (domainArena, mappedDomain) :=
            arena.internSelectionDomain mappedSelection branches.size
          pure ({ domainArena with nodes := domainArena.nodes.set! root {
            expression with node := .select mappedDomain (.exact branches)
          } }, visited)
      | .select selection (.shared representative summary) =>
          let summary ← arena.validatedSchema summary
          let (arena, visited) ← namespaceOperationalExprInPlace scope wire arena visited
            representative fuel
          let mappedSelection : DynamicSelectionIdentity := {
            index := namespaceFreshValueOrigin scope wire selection.index
          }
          let mappedSummary := {
            namespaceFreshSelectedMatrixSummary scope wire summary with
            selectionOrigin := some (selectionDomainKind mappedSelection.index)
          }
          let (arena, mappedDomain) := arena.internSelectionDomain mappedSelection selection.count
          let (arena, mappedSchema) := arena.internValidatedSchema mappedSummary
          pure ({ arena with nodes := arena.nodes.set! root {
            expression with node := (.select mappedDomain
              (.shared representative mappedSchema))
          } }, visited)

partial def factHasRelation : OperationalFact → Bool
  | .matrix fact => !fact.relations.isEmpty || fact.polynomial.any fun term =>
      term.product.factors.any fun factor => !factor.relations.isEmpty
  | .familyUniform _ _ element _ => factHasRelation element
  | .familyPacked elements _ _ => elements.any factHasRelation
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
  | .familyPacked elements count summary =>
      transferPackedOperationalFamily .instantiationMap
        (elements.map (instantiateFactLoopIndex slot index)) count summary
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
  | .familyPacked elements count summary =>
      transferPackedOperationalFamily .instantiationMap
        (elements.map (selectProtocolFamilyElement index)) count summary
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
    (facts : List OperationalFact) : Except OperationalError OperationalFact := do
  match facts with
  | [] => throw (.invalidCount node 0)
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

/-- Substitute the symbolic index of a previously constructed uniform loop family with the
current consumer-loop index. This preserves correlation when a uniform family is zipped into a
later loop instead of treating the producer's template index as an independent selection. -/
private partial def substituteLoopTemplateValueOrigin
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : OperationalValueOrigin → OperationalValueOrigin
  | origin@(.local ..) => if isLoopTemplateSelection binder origin then replacement else origin
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | origin@(.loopInstance slot index source) =>
      if isLoopTemplateSelection binder origin then replacement
      else .loopInstance slot index (substituteLoopTemplateValueOrigin binder replacement source)
  | .selected selectedBinder index source =>
      .selected selectedBinder
        (substituteLoopTemplateValueOrigin binder replacement index)
        (substituteLoopTemplateValueOrigin binder replacement source)

private def substituteLoopTemplateHashIdentity
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (identity : DeterministicHashIdentity) : DeterministicHashIdentity := {
  identity with
  keyOrigin := substituteLoopTemplateValueOrigin binder replacement identity.keyOrigin
  trailingIntegerOrigins := identity.trailingIntegerOrigins.map
    (substituteLoopTemplateValueOrigin binder replacement)
}

private partial def substituteLoopTemplateMatrixOrigin
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : MatrixOriginIdentity → MatrixOriginIdentity
  | origin@(.value ..) => origin
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | .deterministicHash identity =>
      .deterministicHash (substituteLoopTemplateHashIdentity binder replacement identity)
  | .loopInstance slot index source =>
      .loopInstance slot index (substituteLoopTemplateMatrixOrigin binder replacement source)
  | .selected selectedBinder selection source =>
      .selected selectedBinder {
        index := substituteLoopTemplateValueOrigin binder replacement selection.index
      } (substituteLoopTemplateMatrixOrigin binder replacement source)

private partial def substituteLoopTemplatePublicIdentity
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : PublicMatrixIdentity → PublicMatrixIdentity
  | identity@(.sampledTrapdoor ..) => identity
  | identity@(.gadget ..) => identity
  | .loopInstance slot index source =>
      .loopInstance slot index (substituteLoopTemplatePublicIdentity binder replacement source)
  | .selected selectedBinder selection source =>
      .selected selectedBinder {
        index := substituteLoopTemplateValueOrigin binder replacement selection.index
      } (substituteLoopTemplatePublicIdentity binder replacement source)

private def substituteLoopTemplateTarget
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (target : RelationTargetSummary) : RelationTargetSummary := {
  target with
  origin := substituteLoopTemplateMatrixOrigin binder replacement target.origin
  polynomial := mapRelationSnapshotPolynomial
    (substituteLoopTemplateMatrixOrigin binder replacement)
    (substituteLoopTemplatePublicIdentity binder replacement)
    (substituteLoopTemplateValueOrigin binder replacement)
    id target.polynomial
}

private def substituteLoopTemplateRelation
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) :
    OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with
      producer := substituteLoopTemplateMatrixOrigin binder replacement relation.producer
      publicIdentity := substituteLoopTemplatePublicIdentity binder replacement
        relation.publicIdentity
      inputOrigin := substituteLoopTemplateMatrixOrigin binder replacement relation.inputOrigin
      inputSummary := substituteLoopTemplateTarget binder replacement relation.inputSummary
    }
  | .preimage relation => .preimage {
      relation with
      producer := substituteLoopTemplateMatrixOrigin binder replacement relation.producer
      publicIdentity := substituteLoopTemplatePublicIdentity binder replacement
        relation.publicIdentity
      targetOrigin := substituteLoopTemplateMatrixOrigin binder replacement relation.targetOrigin
      targetSummary := substituteLoopTemplateTarget binder replacement relation.targetSummary
    }

private def substituteLoopTemplateMatrixFact
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (fact : OperationalMatrixFact) : OperationalMatrixFact := {
  fact with
  origin := substituteLoopTemplateMatrixOrigin binder replacement fact.origin
  identity := fact.identity.map (substituteLoopTemplatePublicIdentity binder replacement)
  relations := fact.relations.map (substituteLoopTemplateRelation binder replacement)
  polynomial := mapOperationalPolynomial
    (substituteLoopTemplateMatrixOrigin binder replacement)
    (substituteLoopTemplatePublicIdentity binder replacement)
    (substituteLoopTemplateValueOrigin binder replacement)
    id
    (substituteLoopTemplateRelation binder replacement)
    fact.polynomial
}

private def substituteLoopTemplateSummary
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (summary : SelectedMatrixSummary) : SelectedMatrixSummary := {
  summary with
  sharedLastPublicIdentity := summary.sharedLastPublicIdentity.map
    (substituteLoopTemplatePublicIdentity binder replacement)
  sharedFirstRelationPublicIdentity := summary.sharedFirstRelationPublicIdentity.map
    (substituteLoopTemplatePublicIdentity binder replacement)
  selectionOrigin := some (selectionDomainKind replacement)
}

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

/-- Represent a dynamic choice from a construction-uniform matrix family by one checked schema
envelope.  The selected representative carries the unresolved index in every matrix and relation
identity, so the envelope is not an equal-value collapse. -/
private def selectDynamicUniformMatrixEnvelope
    (arena : OperationalExprArena)
    (binder : FamilyTemplateBinder)
    (selection : OperationalValueOrigin)
    (subject : WireRef)
    (count : Nat)
    (fact : OperationalMatrixFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if count = 0 then throw (.invalidCount subject.node 0)
  let selected := selectDynamicMatrixFact binder selection subject fact
  let summary := selectedMatrixSummary #[selected]
  let (arena, representative) := arena.pushConcrete selected
  let (arena, root) ← arena.pushSharedSelection
    ({ index := selection } : DynamicSelectionIdentity) count representative summary
  pure (arena, .matrixExpr root)

/-- Arena-backed parallel-loop input preparation.  Matrix families retain their exact selected
expression (or one checked schema envelope) instead of materializing an indicator polynomial or
the removed fact-level selected-family representation. -/
private def loopTemplateArgumentExpr
    (arena : OperationalExprArena)
    (node argument count : Nat)
    (mode : LoopInputMode)
    (fact : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  match mode with
  | .broadcast => pure (arena, fact)
  | .zip | .zipOffset _ =>
      match fact with
      | .familyUniform binder coordinate element familyCount =>
          let offset := match mode with | .zipOffset value => value | _ => 0
          if count + offset > familyCount.toNat then
            throw (.loopInputModeMismatch node argument)
          let baseSelection : OperationalValueOrigin :=
            .local temporaryScope { node, port := 0 }
          let consumerSelection := match mode with
            | .zip => baseSelection
            | .zipOffset value => .loopInstance argument value baseSelection
            | .broadcast => baseSelection
          let replacement := match coordinate with
            | some (.loopBinder _ _ _) => consumerSelection
            | some (.loopBinderOffset _ _ slot familyOffset) =>
                .loopInstance slot familyOffset consumerSelection
            | none => consumerSelection
          match coordinate, element with
          | some _, .matrixExpr root =>
              let mapFact := substituteLoopTemplateMatrixFact binder replacement
              let mapSelection (selection : DynamicSelectionIdentity) := {
                index := substituteLoopTemplateValueOrigin binder replacement selection.index
              }
              let mapSummary := substituteLoopTemplateSummary binder replacement
              let cacheNamespace :=
                s!"loop-uniform-zip:{node}:{argument}:{reprStr coordinate}:{reprStr replacement}"
              let (arena, mapped) ← mapOperationalExpr cacheNamespace .instantiationMap arena root
                mapFact mapSelection none (some mapSummary)
              pure (arena, .matrixExpr mapped)
          | some _, .matrix matrix =>
              pure (arena, .matrix (substituteLoopTemplateMatrixFact binder replacement matrix))
          | none, .matrix matrix =>
              selectDynamicUniformMatrixEnvelope arena binder replacement
                { node, port := argument } count matrix
          | none, .matrixExpr root => throw (.unsupportedOperationalExpr root)
          | _, .integer integer => pure (arena, .integer { integer with
              origin := substituteLoopTemplateValueOrigin binder replacement integer.origin })
          | _, .bytes bytes => pure (arena, .bytes { bytes with
              origin := substituteLoopTemplateValueOrigin binder replacement bytes.origin })
          | _, .trapdoor trapdoor => pure (arena, .trapdoor { trapdoor with
              publicIdentity := substituteLoopTemplatePublicIdentity binder replacement
                trapdoor.publicIdentity })
          | _, element => pure (arena, element)
      | .familyPacked elements familyCount matrixSummary =>
          let offset := match mode with
            | .zip => 0
            | .zipOffset value => value
            | .broadcast => 0
          if familyCount < count + offset then
            throw (.loopInputModeMismatch node argument)
          let baseSelection : OperationalValueOrigin :=
            .local temporaryScope { node, port := 0 }
          let selection : DynamicSelectionIdentity := {
            index := match mode with
              | .zip => baseSelection
              | .zipOffset value => .loopInstance argument value baseSelection
              | .broadcast => baseSelection
          }
          if count == 1 && elements.size == familyCount then
            match elements[offset]? with
            | some element =>
                let (arena, root) ← arena.pushMatrixFact element
                pure (arena, .matrixExpr root)
            | none => throw (.loopInputModeMismatch node argument)
          else match matrixSummary with
          | some summary =>
              let representativeIndex := if elements.size == familyCount then offset else 0
              let representative ← match elements[representativeIndex]? with
                | some representative => pure representative
                | none => throw (.loopInputModeMismatch node argument)
              let (arena, representative) ← arena.pushMatrixFact representative
              let (arena, root) ← arena.pushSharedSelection selection count representative summary
              pure (arena, .matrixExpr root)
          | none =>
              if elements.size < count + offset then
                throw (.loopInputModeMismatch node argument)
              let mut arena := arena
              let mut branches : Array OperationalExprId := #[]
              for element in elements.extract offset (offset + count) do
                let (nextArena, branch) ← arena.pushMatrixFact element
                arena := nextArena
                branches := branches.push branch
              let (finalArena, root) ← arena.pushSelect selection (.exact branches)
              pure (finalArena, .matrixExpr root)
      | _ => throw (.loopInputModeMismatch node argument)

/-- Re-express one construction-uniform family element in the template coordinate of a newly
constructed family.  This is a binder substitution, not a claim that two family lanes have equal
values.  A family without an explicit construction coordinate remains an unresolved selection by
the new lane identity. -/
private def reindexUniformMatrixFamilyElement
    (arena : OperationalExprArena)
    (node : Nat)
    (outputLane : OperationalValueOrigin)
    (binder : FamilyTemplateBinder)
    (coordinate : Option LoopCoordinate)
    (element : OperationalFact)
    (count : Nat) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let replacement := match coordinate with
    | some (.loopBinder _ _ _) => outputLane
    | some (.loopBinderOffset _ _ slot offset) =>
        .loopInstance slot offset outputLane
    | none => outputLane
  match coordinate, element with
  | some _, .matrixExpr root =>
      let mapFact := substituteLoopTemplateMatrixFact binder replacement
      let mapSelection (selection : DynamicSelectionIdentity) := {
        index := substituteLoopTemplateValueOrigin binder replacement selection.index
      }
      let mapSummary := substituteLoopTemplateSummary binder replacement
      let cacheNamespace :=
        s!"family-select-reindex:{node}:{reprStr binder}:{reprStr coordinate}"
      let (arena, mapped) ← mapOperationalExpr cacheNamespace .instantiationMap arena root
        mapFact mapSelection none (some mapSummary)
      pure (arena, .matrixExpr mapped)
  | some _, .matrix matrix =>
      pure (arena, .matrix (substituteLoopTemplateMatrixFact binder replacement matrix))
  | none, .matrix matrix =>
      selectDynamicUniformMatrixEnvelope arena binder replacement
        { node, port := 0 } count matrix
  | none, .matrixExpr root => throw (.unsupportedOperationalExpr root)
  | _, _ => throw (.loopInputModeMismatch node 1)

/-- Select one matrix family pointwise without materializing its lanes.  Every branch template is
first alpha-renamed to the output-family binder; the ordinary expression selection then preserves
the executable selector identity and all branch-local decomposition/preimage relations. -/
private def selectUniformMatrixFamilies
    (scopeKey : ScopeTemplateKey)
    (node : Nat)
    (selection : OperationalIntegerFact)
    (matrixType : MatrixTypeExpr)
    (expectedCount : Nat)
    (branches : List OperationalFact)
    (arena : OperationalExprArena) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if expectedCount = 0 || branches.isEmpty then
    throw (.invalidCount node expectedCount)
  let outputBinder : FamilyTemplateBinder := {
    owner := scopeKey
    producerNode := node
    binderSlot := 0
  }
  let outputCoordinate : Option LoopCoordinate :=
    some (LoopCoordinate.loopBinder scopeKey node 0)
  let outputLane : OperationalValueOrigin := .local scopeKey { node, port := 0 }
  let choiceBinder : FamilyTemplateBinder := {
    owner := scopeKey
    producerNode := node
    binderSlot := 1
  }
  let dynamicChoice := branches.length > 1
  let mut arena := arena
  let mut roots : Array OperationalExprId := #[]
  for branch in branches do
    let (binder, coordinate, element, count) ← match branch with
      | .familyUniform binder coordinate element count =>
          pure (binder, coordinate, element, count)
      | _ => throw (.loopInputModeMismatch node 1)
    if count != Int.ofNat expectedCount then
      throw (.loopInputModeMismatch node 1)
    let (nextArena, normalized) ← reindexUniformMatrixFamilyElement arena node outputLane
      binder coordinate element expectedCount
    let (nextArena, root) ← nextArena.pushMatrixFact normalized
    if dynamicChoice then
      let mapFact := selectDynamicMatrixFact choiceBinder selection.origin { node, port := 0 }
      let mapSelection (nested : DynamicSelectionIdentity) := {
        index := selectDynamicValueOrigin choiceBinder selection.origin nested.index
      }
      let cacheNamespace := s!"family-select-choice:{node}:{reprStr selection.origin}"
      let (nextArena, selected) ← mapOperationalExpr cacheNamespace .instantiationMap
        nextArena root mapFact mapSelection
      arena := nextArena
      roots := roots.push selected
    else
      arena := nextArena
      roots := roots.push root
  let (finalArena, root) ← arena.pushSelect
    ({ index := selection.origin } : DynamicSelectionIdentity) (.exact roots)
  let expression ← match finalArena.get? root with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef root)
  if expression.matrixType != matrixType then throw (.outputTypeMismatch node)
  pure (finalArena, .familyUniform outputBinder outputCoordinate (.matrixExpr root)
    (Int.ofNat expectedCount))

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
      if index.lower < 0 || index.upper >= Int.ofNat branchCount then
        throw (.invalidCount nodeIndex index.upper)
      let branches ← (node.arguments.drop 1).mapM (lookupFact nodeIndex facts)
      joinDynamicFacts nodeIndex { node := nodeIndex, port := outputPort } branches
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
      let input ← matrixFactAt nodeIndex facts inputWire
      let publicIdentity := PublicMatrixIdentity.gadget descriptor.paramsId params
        input.matrixParams.rows bound small count.toNat
      let result ← cappedMatrixFact nodeIndex outputPort matrixType environment
        (Int.ofNat (Mxx.gadgetDecompositionBound bound small))
      match result with
      | .matrix result =>
          let attachRelation (_branch : Nat) (input : OperationalMatrixFact) :=
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
          pure (.matrix (attachRelation 0 input))
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
      match leftFact, rightFact with
      | .matrix left, .matrix right => combinePair left right
      | _, _ => throw (.operandNotMatrix nodeIndex leftWire)
  | .concat axis, some matrixType =>
      let inputs ← node.arguments.mapM fun wire => matrixFactAt nodeIndex facts wire
      let polynomial ← concatOperationalPolynomials axis matrixType (inputs.map (·.polynomial))
        |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
        (joinCanonicalRanges (inputs.map (·.canonicalRange)))
  | .select, some _ => throw (.unsupportedNode nodeIndex)
  | .transpose, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← matrixFactAt nodeIndex facts inputWire
      let polynomial ← transposeOperationalPolynomial input.polynomial
        |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial input.canonicalRange
  | .slice rows columns, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← matrixFactAt nodeIndex facts inputWire
      let polynomial ← sliceOperationalPolynomial rows columns matrixType input.polynomial
        |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial input.canonicalRange
  | .reshape rows columns, some matrixType =>
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← matrixFactAt nodeIndex facts inputWire
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
      let input ← matrixFactAt nodeIndex facts inputWire
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
      let input ← matrixFactAt nodeIndex facts inputWire
      let input ← retypeMatrixFact nodeIndex matrixType input environment
      polynomialMatrixFact nodeIndex outputPort matrixType environment
        (scaleOperationalPolynomial (-1) input.polynomial) input.canonicalRange
  | .matrixScale scalar, some matrixType =>
      let scalarValues ← evaluateIntOverLoops environment loopDomains scalar
      let inputWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let input ← matrixFactAt nodeIndex facts inputWire
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
      match leftFact, rightFact with
      | .matrix left, .matrix right => multiplyPair left right
      | _, _ => throw (.operandNotMatrix nodeIndex leftWire)
  | .tensor, some matrixType =>
      let leftWire ← match node.arguments[0]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex { node := 0, port := 0 })
      let rightWire ← match node.arguments[1]? with
        | some wire => pure wire
        | none => throw (.missingOperand nodeIndex leftWire)
      let left ← matrixFactAt nodeIndex facts leftWire
      let right ← matrixFactAt nodeIndex facts rightWire
      let polynomial ← tensorOperationalPolynomials matrixType
        left.polynomial right.polynomial |>.mapError (flatErrorAt nodeIndex)
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
  | .crtRecompose plaintextModuli reconstructionCoefficients, some matrixType =>
      if node.arguments.isEmpty || node.arguments.length != plaintextModuli.length ||
          node.arguments.length != reconstructionCoefficients.length then
        throw (.unsupportedOutputArity nodeIndex node.arguments.length)
      let moduli ← plaintextModuli.mapM (evaluateIntInvariant environment loopDomains)
      let coefficients ← reconstructionCoefficients.mapM
        (evaluateIntInvariant environment loopDomains)
      let inputs ← node.arguments.mapM fun wire => matrixFactAt nodeIndex facts wire
      let modulus ← evaluateIntInvariant environment loopDomains matrixType.modulus
      if modulus <= 0 || moduli.any (fun value => value <= 1 || value >= modulus) ||
          coefficients.any (fun value => value < 0 || value >= modulus) then
        throw (.invalidMatrixParameters nodeIndex)
      let inputs ← inputs.mapM fun input => retypeMatrixFact nodeIndex matrixType input environment
      if inputs.any (·.matrixParams.rows != 1) then
        throw (.invalidMatrixParameters nodeIndex)
      let polynomial := (coefficients.zip inputs).foldl
        (fun result pair ↦ addOperationalPolynomials result
          (scaleOperationalPolynomial pair.1 pair.2.polynomial)) []
      polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
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

private def operationalExprContainsSelection
    (arena : OperationalExprArena) (root : OperationalExprId) : Except OperationalError Bool :=
  arena.containsSelection root

/-! Evaluate a single unresolved selection by streaming complete concrete alternatives into a
consumer.  A consumer that would combine two selected subexpressions rejects instead of silently
performing Cartesian-time traversal.  Selection-aware bound evaluation uses the compositional
representative evaluator below and does not call this endpoint helper. -/
private def evaluatePrimitiveConcrete
    (operation : PrimitiveOperation)
    (arguments : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let binaryArguments : Except OperationalError (OperationalMatrixFact × OperationalMatrixFact) := do
    if arguments.size != 2 then
      throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    let left ← match arguments[0]? with
      | some value => pure value
      | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    let right ← match arguments[1]? with
      | some value => pure value
      | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    pure (left, right)
  match operation.kind with
  | .add subtract =>
      let (left, right) ← binaryArguments
      return ← addConcreteMatrixFacts operation.ownerNode operation.outputPort operation.outputType
        subtract operation.parameterEnvironment left right
  | .multiply rule rightWire =>
      let (left, right) ← binaryArguments
      return ← multiplyConcreteMatrixFacts operation.ownerNode operation.outputPort
        operation.outputType rule rightWire operation.parameterEnvironment left right
  | .tensor =>
      let (left, right) ← binaryArguments
      return ← tensorConcreteMatrixFacts operation.ownerNode operation.outputPort operation.outputType
        operation.parameterEnvironment left right
  | .concat axis =>
      return ← concatConcreteMatrixFacts operation.ownerNode operation.outputPort axis
        operation.outputType operation.parameterEnvironment arguments
  | .transform transform =>
      if arguments.size != 1 then
        throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let value ← match arguments[0]? with
        | some value => pure value
        | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      return ← transformConcreteMatrixFact operation.ownerNode operation.outputPort
        operation.outputType transform operation.parameterEnvironment value

private partial def foldOperationalExprConcreteFacts
    {α : Type}
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId)
    (state : α)
    (visit : α → OperationalMatrixFact → Except OperationalError α) :
    Except OperationalError α := do
  let expression ← match arena.get? root with
    | some expression => pure expression
    | none => throw (.invalidOperationalExprRef root)
  match expression.node with
  | .concrete fact => visit state fact
  | .primitive operation arguments => do
      if arguments.countP (fun argument =>
          (arena.get? argument).any (·.containsSelection)) > 1 then
        throw (.unsupportedOperationalExpr root)
      let rec visitArguments
          (remaining : List OperationalExprId)
          (reverseFacts : List OperationalMatrixFact)
          (state : α) : Except OperationalError α := do
        match remaining with
        | [] => visit state (← evaluatePrimitiveConcrete operation reverseFacts.reverse.toArray)
        | argument :: tail =>
            foldOperationalExprConcreteFacts arena environment argument state fun state fact =>
              visitArguments tail (fact :: reverseFacts) state
      visitArguments arguments.toList [] state
  | .select _ (.exact branches) =>
      if branches.isEmpty then throw (.invalidCount 0 0)
      else
        let mut state := state
        for branch in branches do
          state ← foldOperationalExprConcreteFacts arena environment branch state visit
        pure state
  | .select selection (.shared representative summary) =>
      if selection.count = 0 then throw (.invalidCount 0 0)
      else do
        let summary ← arena.validatedSchema summary
        let fact ← arena.concreteFact representative
        if summary.uniformSchema != some (operationalUniformSchema fact) ||
            summary.relationFree != !matrixFactHasRelation fact ||
            summary.sharedLastPublicIdentity != boundaryLastPublicIdentity? fact ||
            summary.sharedFirstRelationPublicIdentity !=
              boundaryFirstRelationPublicIdentity? fact || summary.selectionOrigin.isNone then
          throw (.unsupportedOperationalExpr representative)
        foldOperationalExprConcreteFacts arena environment representative state visit

private inductive OperationalExprBoundKind where
  | total
  | noise

private def OperationalExprEvaluationState.memo
    (state : OperationalExprEvaluationState) :
    OperationalExprBoundKind → Array (Option Int)
  | .total => state.totalMemo
  | .noise => state.noiseMemo

private def OperationalExprEvaluationState.recordHit
    (state : OperationalExprEvaluationState) :
    OperationalExprBoundKind → OperationalExprEvaluationState
  | .total => { state with totalStats := {
      state.totalStats with memoHits := state.totalStats.memoHits + 1
    } }
  | .noise => { state with noiseStats := {
      state.noiseStats with memoHits := state.noiseStats.memoHits + 1
    } }

private def OperationalExprEvaluationState.recordMiss
    (state : OperationalExprEvaluationState) :
    OperationalExprBoundKind → OperationalExprEvaluationState
  | .total => { state with totalStats := {
      state.totalStats with
      evaluations := state.totalStats.evaluations + 1
      memoMisses := state.totalStats.memoMisses + 1
    } }
  | .noise => { state with noiseStats := {
      state.noiseStats with
      evaluations := state.noiseStats.evaluations + 1
      memoMisses := state.noiseStats.memoMisses + 1
    } }

private def OperationalExprEvaluationState.store
    (state : OperationalExprEvaluationState)
    (kind : OperationalExprBoundKind)
    (id : OperationalExprId)
    (value : Int) : OperationalExprEvaluationState :=
  match kind with
  | .total => { state with totalMemo := state.totalMemo.set! id (some value) }
  | .noise => { state with noiseMemo := state.noiseMemo.set! id (some value) }

private def validateOperationalEnvelope
    (representative : OperationalExprId)
    (summary : SelectedMatrixSummary)
    (fact : OperationalMatrixFact) : Except OperationalError Unit := do
  if summary.uniformSchema != some (operationalUniformSchema fact) ||
      summary.relationFree != !matrixFactHasRelation fact ||
      summary.sharedLastPublicIdentity != boundaryLastPublicIdentity? fact ||
      summary.sharedFirstRelationPublicIdentity != boundaryFirstRelationPublicIdentity? fact ||
      summary.selectionOrigin.isNone then
    throw (.unsupportedOperationalExpr representative)

private def evaluateOperationalConcreteBound
    (kind : OperationalExprBoundKind)
    (environment : ParamEnvironment)
    (fact : OperationalMatrixFact) : Except OperationalError Int :=
  match kind with
  | .total => match fact.totalHardBound with
      | .closedInt (.constant value) => pure value
      | expression => expression.evaluateWithStates environment []
  | .noise => fact.evaluateNoiseHardBound environment

private def eraseOperationalFactBounds
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let zero := OperationalBoundExpr.closedInt (.constant 0)
  { fact with
    matrixParams := { fact.matrixParams with maxCoefficientBound := 0 }
    totalHardBound := zero
    polynomial := mapOperationalPolynomial id id id (fun _ => zero) id
      (fact.polynomial.filter operationalTermIsSignal)
    metadata := {}
    identity := none
    relations := [] }

private def sameOperationalSelectionShape
    (left right : OperationalMatrixFact) : Bool :=
  operationalUniformSchema (eraseOperationalFactBounds left) ==
    operationalUniformSchema (eraseOperationalFactBounds right)

private def maximumBoundExpr
    (first : OperationalBoundExpr)
    (remaining : List OperationalBoundExpr) : OperationalBoundExpr :=
  remaining.foldl OperationalBoundExpr.maximum first

/-- Join complete mutually-exclusive alternatives into one relation-free fact for use by a parent
operation.  The join happens only after every branch has produced its complete polynomial.  Signal
shape is retained from the checked common schema, while the complete bounded-only remainder is
replaced by one summary whose bound is the maximum of the complete per-branch noise bounds.  This
prevents a later independent selection from creating a Cartesian traversal and, unlike taking a
maximum for each term, cannot combine correlated pieces from different branches. -/
private def summarizeOperationalSelectionFacts
    (environment : ParamEnvironment)
    (facts : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let first ← match facts[0]? with
    | some first => pure first
    | none => throw (.invalidCount 0 0)
  if facts.any fun fact => fact.matrixType != first.matrixType ||
      !sameOperationalSelectionShape first fact then
    throw (.unsupportedOperationalExpr 0)
  let firstSignal := first.polynomial.filter operationalTermIsSignal
  if facts.any fun fact =>
      (!firstSignal.isEmpty &&
          fact.matrixParams.maxCoefficientBound != first.matrixParams.maxCoefficientBound) ||
        fact.polynomial.filter operationalTermIsSignal != firstSignal then
    -- Keeping branch zero would under-estimate a later bounded multiplication whenever a Large
    -- alternative has a different magnitude or identity-bearing factor. Such selections require
    -- an operation-specific exact rule rather than the relation-free representative join.
    throw (.unsupportedOperationalExpr 0)
  let checkedSummary := selectedMatrixSummary facts
  if facts.any matrixFactHasRelation then
    -- A relation-bearing family may be represented once only when the complete target and public
    -- boundary templates have already been checked for every logical alternative.  The relation
    -- must then be consumed by the immediate concrete parent operation.
    if checkedSummary.uniformSchema.isNone ||
        checkedSummary.sharedFirstRelationPublicIdentity.isNone then
      throw (.unsupportedOperationalExpr 0)
    else
      pure first
  else
    let noiseSummaries ← facts.mapM fun fact =>
      fact.noiseHardBound.mapError fun _ => .invalidMatrixParameters fact.subject.node
    let noiseBound ← match noiseSummaries[0]? with
      | some firstBound => pure (maximumBoundExpr firstBound noiseSummaries.toList.tail)
      | none => throw (.invalidCount 0 0)
    let branchMetadata := facts.map (·.metadata)
    let metadata : OperationalMatrixMetadata := {
      isConstantPolynomial := branchMetadata.all (·.isConstantPolynomial)
      knownZeroRows := match branchMetadata[0]? with
        | some value =>
            if branchMetadata.all (·.knownZeroRows == value.knownZeroRows) then
              value.knownZeroRows
            else none
        | none => none
    }
    let signal := firstSignal
    let noise := if noiseBound == .closedInt (.constant 0) then [] else
      let tokens := [.sumStart, .summaryBound noiseBound, .summaryMetadata metadata, .sumEnd]
      let summary : OperationalBoundedFactorSummary := {
        matrixType := first.matrixType
        hardBound := noiseBound
        metadata
        provenance := tokens
      }
      let origin : OperationalCompressionOrigin := {
        kind := .boundedNoiseSum
        tokens
      }
      let factor : OperationalFactorKey := {
        leaf := .boundedSummary origin summary
        inputType := first.matrixType
        outputType := first.matrixType
        role := .bounded
        boundedSummary := some summary
      }
      [{ coefficient := 1, product := {
          factors := [factor], modes := [], outputType := first.matrixType } }]
    let output ← polynomialMatrixFact first.subject.node first.subject.port first.matrixType
      environment (signal ++ noise) first.canonicalRange
    match output with
    | .matrix fact => pure { fact with
        subject := first.subject
        origin := first.origin
        identity := if facts.all (·.identity == first.identity) then first.identity else none }
    | _ => throw (.operandNotMatrix first.subject.node first.subject)

/-- Return the one representative only when every unresolved choice has already been validated as
Shared. Exact alternatives have no representative: callers needing a bound must use
`evaluateCompleteBound`, which closes every branch before taking its maximum. -/
private def tryUniformRepresentativeWithFuel
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) : Nat →
    Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => match state.representativeMemo[id]? with
  | none => throw (.invalidOperationalExprRef id)
  | some (some fact) => pure (fact, state)
  | some none => do
      let expression ← match arena.get? id with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef id)
      let (fact, state) ← match expression.node with
        | .concrete fact => pure (fact, state)
        | .primitive operation arguments => do
            let mut state := state
            let mut facts : Array OperationalMatrixFact := #[]
            for argument in arguments do
              let (fact, nextState) ← tryUniformRepresentativeWithFuel
                arena environment argument state fuel
              facts := facts.push fact
              state := nextState
            pure (← evaluatePrimitiveConcrete operation facts, state)
        | .select _ (.exact branches) =>
            if branches.isEmpty then throw (.invalidCount 0 0)
            else throw (.unsupportedOperationalExpr id)
        | .select selection (.shared representative summary) => do
            if selection.count = 0 then throw (.invalidCount 0 0)
            let summary ← arena.validatedSchema summary
            let (fact, state) ← tryUniformRepresentativeWithFuel
              arena environment representative state fuel
            validateOperationalEnvelope representative summary fact
            pure (fact, state)
      pure (fact, { state with representativeMemo :=
        state.representativeMemo.set! id (some fact) })

private def tryUniformRepresentative
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :
    Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState) :=
  tryUniformRepresentativeWithFuel
    arena environment id state (arena.nodes.size + 1)

/-- Derive the one fact needed only to validate or transfer a uniform schema. Unlike a value
representative, this operation may close a relation-free Exact choice by summarizing all complete
branches. It is never used for relation rewriting or executable identity checks. -/
private def deriveOperationalSchemaFactWithFuel
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) : Nat →
    Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => match state.schemaFactMemo[id]? with
    | none => throw (.invalidOperationalExprRef id)
    | some (some fact) => pure (fact, state)
    | some none => do
        let expression ← match arena.get? id with
          | some expression => pure expression
          | none => throw (.invalidOperationalExprRef id)
        let (fact, state) ← match expression.node with
          | .concrete fact => pure (fact, state)
          | .select _ (.exact branches) => do
              if branches.isEmpty then throw (.invalidCount 0 0)
              let mut state := state
              let mut facts : Array OperationalMatrixFact := #[]
              for branch in branches do
                let (fact, nextState) ← deriveOperationalSchemaFactWithFuel
                  arena environment branch state fuel
                if matrixFactHasRelation fact then
                  throw (.unknownRelationRequirement fact.subject.node branch)
                facts := facts.push fact
                state := nextState
              pure (← summarizeOperationalSelectionFacts environment facts, state)
          | .select selection (.shared representative summaryId) => do
              if selection.count = 0 then throw (.invalidCount 0 0)
              let summary ← arena.validatedSchema summaryId
              let (fact, state) ← deriveOperationalSchemaFactWithFuel
                arena environment representative state fuel
              validateOperationalEnvelope representative summary fact
              pure (fact, state)
          | .primitive operation arguments =>
              match compositionalTransferRegistry (primitiveTransferClass operation) with
              | .requiresConcreteStructure =>
                  tryUniformRepresentativeWithFuel arena environment id state (fuel + 1)
              | .supported _ => do
                  let mut state := state
                  let mut facts : Array OperationalMatrixFact := #[]
                  for argument in arguments do
                    let (fact, nextState) ← deriveOperationalSchemaFactWithFuel
                      arena environment argument state fuel
                    if matrixFactHasRelation fact then
                      throw (.unknownRelationRequirement operation.ownerNode argument)
                    facts := facts.push fact
                    state := nextState
                  pure (← evaluatePrimitiveConcrete operation facts, state)
        let schemaFactMemo := state.schemaFactMemo.set! id (some fact)
        pure (fact, { state with schemaFactMemo })

private def deriveOperationalSchemaFact
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :=
  deriveOperationalSchemaFactWithFuel arena environment id state (arena.nodes.size + 1)

private def concatOperationalExprIds
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (roots : Array OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  if roots.isEmpty then throw (.invalidCount nodeIndex 0)
  let operation : PrimitiveOperation := {
    kind := .concat axis
    outputType := matrixType
    ownerScope := arena.activeScope
    ownerNode := nodeIndex
    outputPort
    parameterEnvironment := environment
  }
  let concreteTransfer (arguments : Array OperationalMatrixFact) :=
    concatConcreteMatrixFacts nodeIndex outputPort axis matrixType environment arguments
  liftPrimitiveOperation operation .concat concreteTransfer deriveOperationalSchemaFact
    arena roots fuel

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

private def evaluateCompleteBoundWithFuel
    (kind : OperationalExprBoundKind)
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) : Nat →
    Except OperationalError (Int × OperationalExprEvaluationState)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => match (state.memo kind)[id]? with
  | none => throw (.invalidOperationalExprRef id)
  | some (some value) => pure (value, state.recordHit kind)
  | some none => do
      let expression ← match arena.get? id with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef id)
      let state := state.recordMiss kind
      let evaluateChildren
          (children : Array OperationalExprId)
          (state : OperationalExprEvaluationState) := do
        let mut state := state
        for child in children do
          let (_, nextState) ← evaluateCompleteBoundWithFuel
            kind arena environment child state fuel
          state := nextState
        pure state
      let (value, state) ← match expression.node with
        | .concrete fact => pure (← evaluateOperationalConcreteBound kind environment fact, state)
        | .primitive operation arguments => do
            match compositionalTransferRegistry (primitiveTransferClass operation) with
            | .supported .addSubtract =>
                if arguments.size != 2 then
                  throw (.unsupportedOutputArity operation.ownerNode arguments.size)
                let left ← match arguments[0]? with
                  | some value => pure value
                  | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
                let right ← match arguments[1]? with
                  | some value => pure value
                  | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
                let (leftBound, state) ← evaluateCompleteBoundWithFuel
                  kind arena environment left state fuel
                let (rightBound, state) ← evaluateCompleteBoundWithFuel
                  kind arena environment right state fuel
                pure (leftBound + rightBound, state)
            | .supported _ | .requiresConcreteStructure =>
                let state := ← evaluateChildren arguments state
                match tryUniformRepresentative arena environment id state with
                | .ok (fact, state) =>
                    pure (← evaluateOperationalConcreteBound kind environment fact, state)
                | .error (.unsupportedOperationalExpr _) =>
                    throw (.unresolvedConcreteStructure operation.ownerNode id)
                | .error error => throw error
        | .select _ (.exact branches) => do
            let first ← match branches[0]? with
              | some first => pure first
              | none => throw (.invalidCount 0 0)
            let (firstBound, state) ← evaluateCompleteBoundWithFuel
              kind arena environment first state fuel
            let mut maximum := firstBound
            let mut state := state
            for branch in branches.extract 1 branches.size do
              let (bound, nextState) ← evaluateCompleteBoundWithFuel
                kind arena environment branch state fuel
              maximum := max maximum bound
              state := nextState
            pure (maximum, state)
        | .select selection (.shared representative summary) => do
            if selection.count = 0 then throw (.invalidCount 0 0)
            let summary ← arena.validatedSchema summary
            let (fact, state) ← deriveOperationalSchemaFact
              arena environment representative state
            validateOperationalEnvelope representative summary fact
            evaluateCompleteBoundWithFuel kind arena environment representative state fuel
      pure (value, state.store kind id value)

private def evaluateCompleteBound
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :
    Except OperationalError (Int × OperationalExprEvaluationState) :=
  evaluateCompleteBoundWithFuel .total arena environment id state (arena.nodes.size + 1)

private def evaluateOperationalExprNoiseBoundWithState
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (id : OperationalExprId)
    (state : OperationalExprEvaluationState) :
    Except OperationalError (Int × OperationalExprEvaluationState) :=
  evaluateCompleteBoundWithFuel .noise arena environment id state (arena.nodes.size + 1)

def matrixMaximum
    (node : Nat)
    (wire : WireRef)
    (facts : OperationalScopeFacts) : Except OperationalError Int := do
  match ← lookupFact node facts wire with
  | .matrix fact => fact.totalHardBound.evaluate [] #[]
  | .matrixExpr root => do
      let (maximum, _) ← evaluateCompleteBound facts.arena [] root
        (OperationalExprEvaluationState.empty facts.arena)
      pure maximum
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

private def operationalExprHasRelation
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId) : Except OperationalError Bool :=
  foldOperationalExprConcreteFacts arena environment root false fun found fact =>
    pure (found || matrixFactHasRelation fact)

private def matrixBoundaryPublicIdentityMatches
    (expected : PublicMatrixIdentity)
    (fact : OperationalMatrixFact) : Bool :=
  match boundaryLastPublicIdentity? fact with
  | some actual => actual == expected || publicIdentityTemplateEqual actual expected
  | none => false

/-- Require the registered public-matrix boundary on every complete alternative.  Exact
selections are checked branch-by-branch.  Compact envelopes are accepted only through their
validated representative and complete shared boundary template; one mismatching branch therefore
rejects the endpoint rather than being hidden by a numerical maximum. -/
partial def requireOperationalBoundaryPublicIdentity
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (node : Nat)
    (expected : PublicMatrixIdentity) : OperationalFact → Except OperationalError Unit
  | .matrix fact =>
      if matrixBoundaryPublicIdentityMatches expected fact then pure ()
      else throw (.publicIdentityMismatch node)
  | .matrixExpr root => do
      let allMatch ← foldOperationalExprConcreteFacts arena environment root true
        fun allMatch fact => pure (allMatch && matrixBoundaryPublicIdentityMatches expected fact)
      if allMatch then pure () else throw (.publicIdentityMismatch node)
  | .familyUniform _ _ element count =>
      if count <= 0 then throw (.invalidCount node count)
      else requireOperationalBoundaryPublicIdentity arena environment node expected element
  | .familyPacked elements count summary => do
      if elements.isEmpty || !packedMatrixEnvelopeIsComplete elements count summary then
        throw (.invalidCount node (Int.ofNat count))
      if elements.size != count then
        match summary >>= (·.sharedLastPublicIdentity) with
        | some identity =>
            if !(identity == expected || publicIdentityTemplateEqual identity expected) then
              throw (.publicIdentityMismatch node)
        | none => throw (.publicIdentityMismatch node)
      for element in elements do
        requireOperationalBoundaryPublicIdentity arena environment node expected element
  | _ => throw (.operandNotMatrix node { node, port := 0 })

private def summarizeSequentialOperationalExpr
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId) : Except OperationalError OperationalMatrixFact := do
  let summary ← foldOperationalExprConcreteFacts arena environment root none fun summary fact => do
    if matrixFactHasRelation fact then throw (.relationBearingCarriedValue temporaryScope 0 0)
    match summary with
    | none => pure (some fact)
    | some first =>
        if !sameCarriedSchema (.matrix first) (.matrix fact) then
          throw (.sequentialSchemaMismatch temporaryScope 0 0
            (first.polynomial.map operationalLargeFactorCount)
            (fact.polynomial.map operationalLargeFactorCount))
        pure (some { first with
          totalHardBound := .maximum first.totalHardBound fact.totalHardBound })
  match summary with
  | some fact => pure fact
  | none => throw (.invalidCount 0 0)

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
          (inputs : List OperationalFact)
          (arena : OperationalExprArena) :
          Except OperationalError (OperationalExprArena × List OperationalFact) := do
        match modes, inputs with
        | [], [] => pure (arena, [])
        | mode :: modeTail, input :: inputTail =>
            let (arena, head) ←
              loopTemplateArgumentExpr arena nodeIndex argumentIndex count mode input
            let (arena, tail) ← prepareParallelInputs nodeIndex count (argumentIndex + 1)
              modeTail inputTail arena
            pure (arena, head :: tail)
        | _, _ => throw (.loopInputModeMismatch nodeIndex argumentIndex)
      let mut facts : OperationalScopeFacts := {
        arena := { initialArena with activeScope := some scopeKey, activeNode := none }
      }
      for node in scope.nodes do
            let index := facts.values.size
            facts := { facts with arena := {
              facts.arena with activeScope := some scopeKey, activeNode := some index
            } }
            if scope.nodes.size > 1000 && index % 4096 = 0 then
              dbg_trace s!"operational-progress node={index}/{scope.nodes.size} expr={facts.arena.nodes.size} relation_rewrites={facts.arena.relationRewriteCount} transform_hits={facts.arena.transformCacheHits} transform_misses={facts.arena.transformCacheMisses}"
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
                  | .familyUniform binder coordinate element count =>
                      if requested < 0 || requested >= count then
                        throw (.invalidCount index requested)
                      else
                        let subject : WireRef := { node := index, port := 0 }
                        match element with
                        | .matrixExpr root =>
                            let root ← match coordinate with
                              | some (.loopBinder ..) | some (.loopBinderOffset ..) =>
                                  loopTemplateStaticRoot facts.arena binder root requested.toNat
                              | none => pure root
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
                            let cacheNamespace :=
                              s!"family-static:{index}:{requested}:{reprStr coordinate}"
                            let (arena, mapped) ← mapOperationalExpr cacheNamespace
                              .instantiationMap facts.arena root mapFact mapSelection
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
                            let _ ← match summary.uniformSchema with
                              | some schema => pure schema
                              | none => throw (.selectedFamilyOperationUnsupported index)
                            pure [← rebindSubject { node := index, port := 0 }
                              (selectProtocolFamilyElement requested.toNat element)]
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
                  let family ← lookupFact index facts familyWire
                  match selectionFact.lower == selectionFact.upper, family with
                  | true, .familyUniform binder coordinate element count =>
                      let requested := selectionFact.lower
                      if requested < 0 || requested >= count then
                        throw (.invalidCount index requested)
                      else
                        let subject : WireRef := { node := index, port := 0 }
                        match element with
                        | .matrixExpr root =>
                            let root ← match coordinate with
                              | some (.loopBinder ..) | some (.loopBinderOffset ..) =>
                                  loopTemplateStaticRoot facts.arena binder root requested.toNat
                              | none => pure root
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
                            let mapSelection (nested : DynamicSelectionIdentity) := {
                              index := match coordinate with
                                | some (.loopBinder _ _ slot) =>
                                    instantiateValueOriginLoopIndex slot requested.toNat nested.index
                                | some (.loopBinderOffset _ _ slot offset) =>
                                    instantiateValueOriginLoopIndex slot (requested.toNat + offset)
                                      nested.index
                                | none => selectProtocolValueOrigin requested.toNat nested.index
                            }
                            let cacheNamespace :=
                              s!"family-dynamic-exact:{index}:{requested}:{reprStr coordinate}"
                            let (arena, mapped) ← mapOperationalExpr cacheNamespace
                              .instantiationMap facts.arena root mapFact mapSelection
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
                  | true, .familyPacked elements count summary =>
                      let requested := selectionFact.lower
                      if requested < 0 || requested >= Int.ofNat count then
                        throw (.invalidCount index requested)
                      else if elements.size == count then
                        match elements[requested.toNat]? with
                        | some element => pure [← rebindSubject { node := index, port := 0 } element]
                        | none => throw (.invalidCount index requested)
                      else match summary, elements[0]? with
                        | some summary, some element =>
                            let _ ← match summary.uniformSchema with
                              | some schema => pure schema
                              | none => throw (.selectedFamilyOperationUnsupported index)
                            pure [← rebindSubject { node := index, port := 0 }
                              (selectProtocolFamilyElement requested.toNat element)]
                        | _, _ => throw (.selectedFamilyOperationUnsupported index)
                  | true, _ => throw (.loopInputModeMismatch index 0)
                  | false, .familyUniform binder coordinate element count =>
                      if count <= 0 || selectionFact.lower < 0 || selectionFact.upper >= count then
                        throw (.invalidCount index selectionFact.upper)
                      else
                        let subject : WireRef := { node := index, port := 0 }
                        match element with
                        | .matrixExpr root =>
                            let mapFact := selectDynamicMatrixFact binder selection subject
                            let mapSelection (nested : DynamicSelectionIdentity) := {
                              index := if coordinate.isSome &&
                                  isLoopTemplateSelection binder nested.index then
                                selection
                              else selectDynamicValueOrigin binder selection nested.index
                            }
                            let cacheNamespace :=
                              s!"family-dynamic:{index}:{reprStr selection}:{reprStr coordinate}"
                            let (arena, mapped) ← mapOperationalExpr cacheNamespace
                              .instantiationMap facts.arena root mapFact mapSelection
                            facts := { facts with arena }
                            pure [.matrixExpr mapped]
                        | .matrix matrix =>
                            let (arena, selected) ← selectDynamicUniformMatrixEnvelope facts.arena
                              binder selection subject count.toNat matrix
                            facts := { facts with arena }
                            pure [selected]
                        | element => pure [← selectDynamicUniformFact binder selection subject element]
                  | false, .familyPacked elements count summary =>
                      if count == 0 || selectionFact.lower < 0 ||
                          selectionFact.upper >= Int.ofNat count then
                        throw (.invalidCount index selectionFact.upper)
                      else match summary, elements[0]? with
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
                          let transferred ← match transferSelectedMatrixSummary
                              .instantiationMap #[summary] selected with
                            | some transferred => pure transferred
                            | none => throw (.selectedFamilyOperationUnsupported index)
                          let (arena, representativeId) := facts.arena.pushConcrete selected
                          let (arena, root) ← arena.pushSharedSelection
                            ({ index := selection } : DynamicSelectionIdentity) count representativeId
                              transferred
                          facts := { facts with arena }
                          pure [.matrixExpr root]
                        | _ => throw (.loopInputModeMismatch index 0)
                      | _, _ =>
                        if elements.size == count then
                          let binder : FamilyTemplateBinder := {
                            owner := dynamicSelectionScope selection
                            producerNode := index
                            binderSlot := 0
                          }
                          let subject : WireRef := { node := index, port := 0 }
                          let mut arena := facts.arena
                          let mut selectedBranches : Array OperationalExprId := #[]
                          for branch in elements do
                            match branch with
                            | .matrix branch =>
                                let selected :=
                                  selectDynamicMatrixFact binder selection subject branch
                                let (nextArena, root) := arena.pushConcrete selected
                                arena := nextArena
                                selectedBranches := selectedBranches.push root
                            | .matrixExpr root =>
                                let mapFact := selectDynamicMatrixFact binder selection subject
                                let mapSelection (nested : DynamicSelectionIdentity) := {
                                  index := selectDynamicValueOrigin binder selection nested.index
                                }
                                let cacheNamespace :=
                                  s!"packed-dynamic:{index}:{reprStr selection}:{reprStr binder}"
                                let (nextArena, selected) ← mapOperationalExpr cacheNamespace
                                  .instantiationMap arena root mapFact mapSelection
                                arena := nextArena
                                selectedBranches := selectedBranches.push selected
                            | _ => throw (.loopInputModeMismatch index 0)
                          let (finalArena, root) ← arena.pushSelect
                            ({ index := selection } : DynamicSelectionIdentity)
                            (.exact selectedBranches)
                          facts := { facts with arena := finalArena }
                          pure [.matrixExpr root]
                        else throw (.selectedFamilyOperationUnsupported index)
                  | false, _ => throw (.loopInputModeMismatch index 0)
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
                  let (arena, templateInputs) ←
                    prepareParallelInputs index evaluatedCount.toNat 0 modes actualInputs facts.arena
                  facts := { facts with arena }
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
                    let relationBearing ← match fact with
                      | .matrixExpr root => operationalExprHasRelation facts.arena environment root
                      | fact => pure (factHasRelation fact)
                    if relationBearing then
                      throw (.relationBearingCarriedValue scopeKey index slot)
                  let mut abstractCarried : List OperationalFact := []
                  for (fact, slot) in carriedFacts.zipIdx do
                    match fact with
                    | .matrixExpr root =>
                        let mapFact (fact : OperationalMatrixFact) :=
                          match abstractCarriedMaximum slot (.matrix fact) with
                          | .matrix mapped => mapped
                          | _ => fact
                        let cacheNamespace := s!"sequential-carried:{index}:{slot}"
                        let (arena, mapped) ← mapOperationalExpr cacheNamespace
                          .instantiationMap facts.arena root mapFact
                        facts := { facts with arena }
                        abstractCarried := abstractCarried ++ [.matrixExpr mapped]
                    | fact =>
                        abstractCarried := abstractCarried ++ [abstractCarriedMaximum slot fact]
                  let mut shiftedInvariantFacts : List OperationalFact := []
                  for fact in invariantFacts do
                    match fact with
                    | .matrixExpr root =>
                        let mapFact (fact : OperationalMatrixFact) :=
                          match shiftFactPreviousDepth (.matrix fact) with
                          | .matrix mapped => mapped
                          | _ => fact
                        let cacheNamespace := s!"sequential-invariant:{index}"
                        let (arena, mapped) ← mapOperationalExpr cacheNamespace
                          .instantiationMap facts.arena root mapFact
                        facts := { facts with arena }
                        shiftedInvariantFacts := shiftedInvariantFacts ++ [.matrixExpr mapped]
                    | fact =>
                        shiftedInvariantFacts := shiftedInvariantFacts ++
                          [shiftFactPreviousDepth fact]
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
                  let rawOutputTemplates ← scopeOutputFacts index child.scope.outputs childFacts
                  if rawOutputTemplates.length != carriedCount then
                    throw (.childInputMismatch index carriedCount rawOutputTemplates.length)
                  let mut initialTemplates : List OperationalFact := []
                  for carried in carriedFacts do
                    match carried with
                    | .matrixExpr root =>
                        initialTemplates := initialTemplates ++
                          [.matrix (← summarizeSequentialOperationalExpr facts.arena environment root)]
                    | carried => initialTemplates := initialTemplates ++ [carried]
                  let mut outputTemplates : List OperationalFact := []
                  for output in rawOutputTemplates do
                    match output with
                    | .matrixExpr root =>
                        outputTemplates := outputTemplates ++
                          [.matrix (← summarizeSequentialOperationalExpr facts.arena environment root)]
                    | output => outputTemplates := outputTemplates ++ [output]
                  for slot in List.range carriedCount do
                    match initialTemplates[slot]?, outputTemplates[slot]? with
                    | some initial, some output =>
                        if !sameCarriedSchema initial output || factHasRelation output then
                          if factHasRelation output then
                            throw (.relationBearingCarriedValue scopeKey index slot)
                          else throw (.sequentialSchemaMismatch scopeKey index slot
                            (carriedLargeFactorCounts initial) (carriedLargeFactorCounts output))
                    | _, _ => throw (.childInputMismatch index carriedCount outputTemplates.length)
                  let initialComponents := initialTemplates.zipIdx.flatMap fun (carried, slot) =>
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
              | .thresholdDecodeBool ciphertextModulus plaintextModulus length |
                  .thresholdDecodeInt ciphertextModulus plaintextModulus length =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  let input ← lookupFact index facts inputWire
                  match input with
                  | .matrixExpr root =>
                      let ciphertext ← evaluateIntInvariant environment loopDomains
                        ciphertextModulus
                      let plaintext ← evaluateIntInvariant environment loopDomains
                        plaintextModulus
                      let count ← evaluateIntInvariant environment loopDomains length
                      let allValid ← foldOperationalExprConcreteFacts facts.arena environment root
                        true fun valid branch => pure (valid &&
                          branch.matrixParams.rows == 1 && branch.matrixParams.columns == 1 &&
                          ciphertext == branch.matrixParams.modulus && plaintext > 1 && count > 0 &&
                          count <= Int.ofNat branch.matrixParams.ringDimension &&
                          node.outputCount == count.toNat)
                      if !allValid then throw (.invalidMatrixParameters index)
                      node.outputTypes.zipIdx.mapM fun (outputType, port) =>
                        match node.kind, outputType with
                        | .thresholdDecodeBool .., .boolean => pure .boolean
                        | .thresholdDecodeInt .., .integer =>
                            integerFact index port 0 (plaintext - 1)
                        | _, _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                  | _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
              | .extractCoefficient position =>
                  let inputWire ← match node.arguments with
                    | [wire] => pure wire
                    | _ => throw (.unsupportedOutputArity index node.arguments.length)
                  let input ← lookupFact index facts inputWire
                  match input with
                  | .matrixExpr root =>
                      let minimum ← evaluateIntMinimum environment loopDomains position
                      let maximum ← evaluateIntMaximum environment loopDomains position
                      let exclusiveUpper? ← foldOperationalExprConcreteFacts facts.arena environment
                        root none fun current branch => do
                          if minimum < 0 ||
                              maximum >= Int.ofNat branch.matrixParams.ringDimension then
                            throw (.invalidCount index maximum)
                          let branchUpper := match branch.canonicalRange with
                            | .below upper => Int.ofNat upper
                            | .unknown => branch.matrixParams.modulus
                          if branchUpper <= 0 then throw (.invalidMatrixParameters index)
                          pure (some (match current with
                            | some previous => max previous branchUpper
                            | none => branchUpper))
                      let exclusiveUpper ← match exclusiveUpper? with
                        | some value => pure value
                        | none => throw (.invalidCount index 0)
                      pure [← integerFact index 0 0 (exclusiveUpper - 1)]
                  | _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
              | .select =>
                  let indexWire ← match node.arguments[0]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let selection ← match ← lookupFact index facts indexWire with
                    | .integer fact => pure fact
                    | _ => throw (.loopInputModeMismatch index 0)
                  let branchWires := node.arguments.drop 1
                  if branchWires.isEmpty || selection.lower < 0 ||
                      selection.upper >= Int.ofNat branchWires.length then
                    throw (.invalidCount index selection.upper)
                  let branches ← branchWires.mapM (lookupFact index facts)
                  match node.outputTypes with
                  | [.indexedFamily (.matrix matrixType) count]
                  | [.indexedFamily (.preimage matrixType) count] =>
                      let expectedCount ← match count.evaluate environment with
                        | some value => pure value
                        | none => throw .nonClosedExpression
                      if expectedCount <= 0 then throw (.invalidCount index expectedCount)
                      let selectedBranches ←
                        if selection.lower == selection.upper then
                          match branches[selection.lower.toNat]? with
                          | some selected => pure [selected]
                          | none => throw (.invalidCount index selection.lower)
                        else pure branches
                      let (arena, output) ← selectUniformMatrixFamilies scopeKey index selection
                        matrixType expectedCount.toNat selectedBranches facts.arena
                      facts := { facts with arena }
                      pure [output]
                  | [.matrix _] | [.preimage _] =>
                      if selection.lower == selection.upper then
                        let selected ← match branches[selection.lower.toNat]? with
                          | some selected => pure selected
                          | none => throw (.invalidCount index selection.lower)
                        pure [← rebindSubject { node := index, port := 0 } selected]
                      else
                        let mut arena := facts.arena
                        let mut roots : Array OperationalExprId := #[]
                        for branch in branches do
                          let (nextArena, root) ← arena.pushMatrixFact branch
                          arena := nextArena
                          roots := roots.push root
                        let (finalArena, root) ← arena.pushSelect
                          ({ index := selection.origin } : DynamicSelectionIdentity)
                          (.exact roots)
                        facts := { facts with arena := finalArena }
                        pure [.matrixExpr root]
                  | _ =>
                      deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                        layouts facts 0 node.outputTypes
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
                        [coefficient] environment loopDomains
                          deriveOperationalSchemaFact arena input
                      arena := nextArena
                      scaled := scaled.push output
                    let mut output ← match scaled[0]? with
                      | some output => pure output
                      | none => throw (.invalidCount index 0)
                    for next in scaled.extract 1 scaled.size do
                      let (nextArena, sum) ← addOperationalExprFacts index 0 matrixType false
                        environment deriveOperationalSchemaFact arena output next
                      arena := nextArena
                      output := sum
                    facts := { facts with arena }
                    pure [output]
                  else
                    deriveOrdinaryOutputs scopeKey index node step.rule environment loopDomains
                      layouts facts 0 node.outputTypes
              | .preimageSample .. =>
                  let targetWire ← match node.arguments[2]? with
                    | some wire => pure wire
                    | none => throw (.missingOperand index { node := 0, port := 0 })
                  let target ← lookupFact index facts targetWire
                  match target with
                  | .matrixExpr root =>
                      let matrixType ← match node.outputTypes with
                        | [.matrix matrixType] | [.preimage matrixType] => pure matrixType
                        | _ => throw (.unsupportedOutputArity index node.outputTypes.length)
                      let rec mapPreimageExpression : OperationalExprArena → OperationalExprId → Nat →
                          Except OperationalError (OperationalExprArena × OperationalExprId)
                        | _, current, 0 => throw (.unsupportedOperationalExpr current)
                        | arena, current, remaining + 1 => do
                            let expression ← match arena.get? current with
                              | some expression => pure expression
                              | none => throw (.invalidOperationalExprRef current)
                            match expression.node with
                            | .concrete branch =>
                                let branchFacts ← replaceOperationalFact index facts targetWire
                                  (.matrix branch)
                                let output ← genericNodeFact scopeKey index node step.rule 0
                                  (.preimage matrixType) branchFacts environment loopDomains layouts
                                let output := namespaceFreshOutput scopeKey
                                  { node := index, port := 0 } output
                                let output ← match output with
                                  | .matrix output => pure output
                                  | _ => throw (.operandNotMatrix index targetWire)
                                pure (arena.pushConcrete output)
                            | .select selection (.exact branches) =>
                                let mut arena := arena
                                let mut outputs : Array OperationalExprId := #[]
                                for branch in branches do
                                  let (nextArena, output) ←
                                    mapPreimageExpression arena branch remaining
                                  arena := nextArena
                                  outputs := outputs.push output
                                arena.pushPrimitiveSelection selection matrixType environment outputs
                            | .select selection
                                (.shared representative summary) =>
                                let summary ← arena.validatedSchema summary
                                let (arena, output) ←
                                  mapPreimageExpression arena representative remaining
                                let state := OperationalExprEvaluationState.empty arena
                                let (outputFact, _) ← tryUniformRepresentative
                                  arena environment output state
                                let outputSummary ← match transferSelectedMatrixSummary
                                    .preimage #[summary] outputFact with
                                  | some value => pure value
                                  | none => throw (.unsupportedOperationalExpr representative)
                                arena.pushCheckedSchemaEnvelope selection selection.count output
                                  outputSummary outputFact
                            | _ => throw (.unsupportedOperationalExpr current)
                      let (arena, output) ← mapPreimageExpression facts.arena root
                        (facts.arena.nodes.size + 1)
                      facts := { facts with arena }
                      pure [.matrixExpr output]
                  | _ =>
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
                                arena.pushPrimitiveSelection selection matrixType environment outputs
                            | .select selection
                                (.shared representative summary) =>
                                let summary ← arena.validatedSchema summary
                                let (arena, output) ←
                                  mapExpression arena representative remaining
                                let state := OperationalExprEvaluationState.empty arena
                                let (outputFact, _) ← tryUniformRepresentative
                                  arena environment output state
                                let outputSummary ← match transferSelectedMatrixSummary
                                    .decomposition #[summary] outputFact with
                                  | some value => pure value
                                  | none => throw (.unsupportedOperationalExpr representative)
                                arena.pushCheckedSchemaEnvelope selection selection.count output
                                  outputSummary outputFact
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
                        scalarValues environment loopDomains deriveOperationalSchemaFact
                          facts.arena input
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
                          matrixType operation environment deriveOperationalSchemaFact
                            arena output
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
                        environment deriveOperationalSchemaFact facts.arena left right
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
                        step.rule rightWire environment deriveOperationalSchemaFact
                          facts.arena left right
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
                        environment deriveOperationalSchemaFact facts.arena left right
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
                  let (arena, _) ← namespaceOperationalExprInPlace scopeKey wire facts.arena {}
                    root (facts.arena.nodes.size + 1)
                  facts := { facts with arena }
                  namespacedOutputs := namespacedOutputs.push (.matrixExpr root)
              | output => namespacedOutputs := namespacedOutputs.push output
            let outputs := namespacedOutputs
            facts := { facts with values := facts.values.push outputs }
            let attachments := prepared.attachmentBuckets[index]?.getD #[]
            facts := ← applyPreparedDerivationAttachments index attachments environment facts
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
  | .matrixExact contractType canonicalUpper isConstantPolynomial, .matrix wireMatrixType =>
      if contractType != wireMatrixType then throw (.inputContractMismatch "matrix")
      let cap ← match matrixCap wireMatrixType environment with
        | some value => pure value
        | none => throw (.invalidMatrixParameters subject.node)
      let canonicalRange ← match canonicalUpper with
        | none => pure CanonicalRange.unknown
        | some upper =>
            let upper ← match upper.evaluate environment with
              | some value => pure value
              | none => throw .nonClosedExpression
            if upper <= 0 then throw (.inputContractMismatch "matrix canonical range")
            pure (.below upper.toNat)
      return setMatrixOrigin (← classifiedMatrixFact subject.node subject.port wireMatrixType
        environment cap true canonicalRange { isConstantPolynomial })
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

structure OperationalAnalysisDiagnostics where
  expressionNodeCount : Nat := 0
  memoEvaluations : Nat := 0
  memoHits : Nat := 0
  memoMisses : Nat := 0
  peakMemoEntries : Nat := 0
  envelopeLogicalBranchCount : Nat := 0
  envelopeStoredBranchCount : Nat := 0
  relationRewriteCount : Nat := 0
  transformCacheHits : Nat := 0
  transformCacheMisses : Nat := 0
  cartesianPairVisits : Nat := 0
  maximumPolynomialTerms : Nat := 0
  deriving BEq, DecidableEq, Repr

private def operationalAnalysisDiagnostics
    (arena : OperationalExprArena)
    (stats : OperationalExprEvaluationStats := {}) : OperationalAnalysisDiagnostics := Id.run do
  let mut logicalBranches := 0
  let mut storedBranches := 0
  let mut maximumPolynomialTerms := 0
  for expression in arena.nodes do
    match expression.node with
    | .concrete fact =>
        maximumPolynomialTerms := max maximumPolynomialTerms fact.polynomial.length
    | .select _ (.exact branches) =>
        logicalBranches := logicalBranches + branches.size
        storedBranches := storedBranches + branches.size
    | .select selection (.shared _ _) =>
        logicalBranches := logicalBranches + selection.count
        storedBranches := storedBranches + 1
    | _ => pure ()
  return {
    expressionNodeCount := arena.nodes.size
    memoEvaluations := stats.evaluations
    memoHits := stats.memoHits
    memoMisses := stats.memoMisses
    peakMemoEntries := arena.nodes.size
    envelopeLogicalBranchCount := logicalBranches
    envelopeStoredBranchCount := storedBranches
    relationRewriteCount := arena.relationRewriteCount
    transformCacheHits := arena.transformCacheHits
    transformCacheMisses := arena.transformCacheMisses
    -- Independent selections are summarized compositionally. There is deliberately no Cartesian
    -- traversal or fallback whose visits could increment this counter.
    cartesianPairVisits := 0
    maximumPolynomialTerms
  }

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
  diagnostics : OperationalAnalysisDiagnostics := {}

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

private def evaluateOperationalExprNoiseBoundWithStats
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId) :
    Except OperationalError (Int × OperationalExprEvaluationStats) := do
  let (maximum, state) ← evaluateOperationalExprNoiseBoundWithState arena environment root
    (OperationalExprEvaluationState.empty arena)
  pure (maximum, state.noiseStats)

private def evaluateOperationalExprNoiseBound
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (root : OperationalExprId) : Except OperationalError Int := do
  let (maximum, _) ← evaluateOperationalExprNoiseBoundWithStats arena environment root
  pure maximum

private def addOperationalExprEvaluationStats
    (left right : OperationalExprEvaluationStats) : OperationalExprEvaluationStats := {
  evaluations := left.evaluations + right.evaluations
  memoHits := left.memoHits + right.memoHits
  memoMisses := left.memoMisses + right.memoMisses
}

private partial def collectDecoderResidualBounds
    (arena : OperationalExprArena)
    (environment : ParamEnvironment) : OperationalFact →
    Except OperationalError (List Int × OperationalExprEvaluationStats)
  | .matrix residual => return ([← residual.evaluateNoiseHardBound environment], {})
  | .matrixExpr root => do
      let (bound, stats) ← evaluateOperationalExprNoiseBoundWithStats arena environment root
      pure ([bound], stats)
  | .familyUniform _ _ element count => do
      if count <= 0 then
        throw (.invalidCount 0 count)
      collectDecoderResidualBounds arena environment element
  | .familyPacked elements count summary => do
      if elements.isEmpty || !packedMatrixEnvelopeIsComplete elements count summary then
        throw (.invalidCount 0 (Int.ofNat count))
      let rows ← elements.toList.mapM (collectDecoderResidualBounds arena environment)
      pure (rows.flatMap (fun row => row.1), rows.foldl
        (fun stats row => addOperationalExprEvaluationStats stats row.2) {})
  | _ => throw (.operandNotMatrix 0 { node := 0, port := 0 })

/-- Evaluate every matrix-like port produced at `node` across all workflow stages.  This helper is
used only by the external performance harness to time former hot nodes; it does not affect the
accepted bound or executable graph.  Missing node indices are skipped because stage scopes have
different sizes, while a present unsupported port still fails closed. -/
def operationalNodeNoiseBounds
    (outputs : List OperationalStageResult)
    (node : Nat)
    (environment : ParamEnvironment) : Except OperationalError (List Int) := do
  let mut result : List Int := []
  for stage in outputs do
    match stage.facts.values[node]? with
    | none => pure ()
    | some ports =>
        for fact in ports do
          match fact with
          | .matrix _ | .matrixExpr _ | .familyUniform .. | .familyPacked .. =>
              let (bounds, _) ←
                collectDecoderResidualBounds stage.facts.arena environment fact
              result := result ++ bounds
          | _ => pure ()
  pure result

/-- Evaluates the graph-derived structural bound for a matrix residual or residual family once.
The result is independent of the decoder threshold and can therefore be reused by compatible
parameter requests. Packed families are checked member-by-member and use their maximum bound. -/
def operationalNoiseBoundForFact
    (arena : OperationalExprArena)
    (residual : OperationalFact)
    (environment : ParamEnvironment) :
    Except OperationalError (Int × OperationalAnalysisDiagnostics) := do
  let (bounds, evaluationStats) ← collectDecoderResidualBounds arena environment residual
  let noiseBound ← match bounds with
    | head :: tail => pure (tail.foldl max head)
    | [] => throw (OperationalError.invalidCount 0 0)
  pure (noiseBound, operationalAnalysisDiagnostics arena evaluationStats)

/-- Applies a cheap decoder threshold to an already evaluated structural bound. -/
def decoderNoiseCheckReportFromBound
    (outputs : List OperationalStageResult)
    (noiseBound : Int)
    (diagnostics : OperationalAnalysisDiagnostics)
    (plaintextModulus ciphertextModulus : Int) : OperationalNoiseCheckReport :=
  let obligation := OperationalNoiseObligation.decoderThreshold
    plaintextModulus ciphertextModulus noiseBound
  let (accepted, rejection) :=
    checkDecoderThreshold plaintextModulus ciphertextModulus noiseBound
  {
    outputs := outputs
    obligations := [obligation]
    accepted := accepted
    rejection := rejection
    diagnostics }

/-- Builds one decoder obligation from a residual. Prefer `operationalNoiseBoundForFact` followed
by `decoderNoiseCheckReportFromBound` when several threshold requests share the same residual and
numeric environment. -/
def decoderNoiseCheckReportForFact
    (outputs : List OperationalStageResult)
    (arena : OperationalExprArena)
    (residual : OperationalFact)
    (environment : ParamEnvironment)
    (plaintextModulus ciphertextModulus : Int) :
    Except OperationalError OperationalNoiseCheckReport := do
  let (noiseBound, diagnostics) ← operationalNoiseBoundForFact arena residual environment
  pure (decoderNoiseCheckReportFromBound outputs noiseBound diagnostics
    plaintextModulus ciphertextModulus)

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
example : (show Except OperationalError Bool from do
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

private def interningFixtureFactor (node : Nat) : OperationalFactorKey := {
  leaf := .primitive (.matrix (.value temporaryScope { node, port := 0 }))
  inputType := fixtureType
  outputType := fixtureType
  role := .large
}

private def interningFixtureProduct (factors : List OperationalFactorKey) : OperationalProductKey := {
  factors
  modes := List.replicate (factors.length - 1) .ordinaryMatrixProduct
  outputType := fixtureType
}

/-- Factor fingerprints are candidate indices only: equal factors reuse an ID, while two unequal
factors deliberately sharing this coarse fingerprint remain distinct. -/
example :
    let first := interningFixtureFactor 10
    let different := interningFixtureFactor 11
    let (arena, firstId) := internOperationalFactor {} first
    let (arena, repeatedId) := internOperationalFactor arena first
    let (arena, differentId) := internOperationalFactor arena different
    firstId = repeatedId ∧ firstId ≠ differentId ∧
      arena.factorHits = 1 ∧ arena.factorMisses = 2 := by
  native_decide

/-- Ordered products receive stable request-local IDs and cancellation retains deterministic
first-occurrence order. -/
example :
    let first := interningFixtureFactor 20
    let second := interningFixtureFactor 21
    let forward := interningFixtureProduct [first, second]
    let reverse := interningFixtureProduct [second, first]
    let (arena, forwardId) := internOperationalProduct {} forward
    let (arena, repeatedId) := internOperationalProduct arena forward
    let (_, reverseId) := internOperationalProduct arena reverse
    let normalized := normalizeOperationalTerms [
      { coefficient := 3, product := forward },
      { coefficient := 5, product := reverse },
      { coefficient := -3, product := forward }]
    (forwardId == repeatedId) = true ∧ (forwardId == reverseId) = false ∧
      (normalized == [{ coefficient := 5, product := reverse }]) = true := by
  native_decide

/-- An exact external matrix is not a zero matrix. Without an explicit bounded contract it keeps
the conservative centered-residue cap and a Large primitive factor. -/
example : (do
    let fact ← contractFact (.root (.workflowStage ⟨"stage"⟩)) { node := 0, port := 0 }
      ⟨"matrix"⟩ (.matrix fixtureType) (.matrixExact fixtureType none false) []
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

/-- Relation-consuming multiplication preserves the established concrete bound and recomputes an
envelope summary from its output.  The consumed relation boundary cannot survive, and an operation
absent from the transfer registry has no permissive fallback. -/
example : (show Except OperationalError (Int × Bool × Bool) from do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    let maximum ← matrixMaximum 3 { node := 3, port := 0 } facts
    let relationBearing ← matrixFactAt 3 facts { node := 2, port := 0 }
    let rewritten ← matrixFactAt 3 facts { node := 3, port := 0 }
    let source := selectedMatrixSummary #[relationBearing]
    let output ← match transferSelectedMatrixSummary .multiplyRelation #[source] rewritten with
      | some output => pure output
      | none => throw (OperationalError.unsupportedOperationalExpr 0)
    pure (maximum,
      source.sharedFirstRelationPublicIdentity.isSome &&
        output.sharedFirstRelationPublicIdentity.isNone && output.relationFree,
      (transferSelectedMatrixSummary .unregistered #[source] rewritten).isNone)) =
    Except.ok (3, true, true) := by
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
    | .matrix fact => pure (!fact.relations.isEmpty)
    | .matrixExpr root => match facts.arena.get? root with
        | some { node := .select _ (.exact branches), .. } => match branches[0]? with
            | some firstId => match facts.arena.get? firstId with
                | some { node := .concrete first, .. } =>
                    pure (branches.size == 2 && !first.relations.isEmpty)
                | _ => pure false
            | none => pure false
        | _ => pure false
    | _ => pure false
    let rewritten ← lookupFact 11 facts { node := 11, port := 0 }
    let rewrittenBounds ← match rewritten with
    | .matrix fact => pure [← fact.totalHardBound.evaluateWithStates [] []]
    | .matrixExpr root => match facts.arena.get? root with
        | some { node := .select _ (.exact branches), .. } => do
            let bounds ← branches.toList.mapM fun branch => match facts.arena.get? branch with
              | some { node := .concrete fact, .. } =>
                  fact.totalHardBound.evaluateWithStates [] []
              | _ => throw (OperationalError.unsupportedOperationalExpr branch)
            pure bounds
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
    let (envelopeArena, envelopeRoot) ← envelopeArena.pushSharedSelection envelopeSelection
      30720 representativeId summary
    let (envelopeBound, _) ← evaluateCompleteBound envelopeArena [] envelopeRoot
      (OperationalExprEvaluationState.empty envelopeArena)
    let staleRepresentative := { representative with
      totalHardBound := OperationalBoundExpr.closedInt (.constant 8) }
    let (staleArena, staleId) := ({} : OperationalExprArena).pushConcrete staleRepresentative
    let staleRejected := match staleArena.pushSharedSelection envelopeSelection 2 staleId summary with
      | .error (.unsupportedOperationalExpr _) => true
      | _ => false
    let report ← decoderNoiseCheckReportForFact [] facts.arena rewritten [] 2 25
    pure (dynamicOk && rewrittenBounds == [3] && envelopeArena.nodes.size == 2 &&
      envelopeBound == 7 && staleRejected &&
      report.obligations == [.decoderThreshold 2 25 3])

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

private def packedSelectionLoopBody : Scope := {
  nodes := #[{
    kind := .input "value"
    arguments := []
    outputTypes := [.matrix fixtureType]
  }]
  outputs := [("result", { node := 0, port := 0 })]
  inputNames := ["value"]
}

private def packedSelectionLoopProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .familyPack,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .parallelLoop "body" (.constant 2) 0 [] [.zip],
        arguments := [{ node := 2, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .familyGetStatic (.constant 1), arguments := [{ node := 3, port := 0 }],
        outputTypes := [.matrix fixtureType] },
      { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
      { kind := .familyGetDynamic,
        arguments := [{ node := 3, port := 0 }, { node := 5, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("family", { node := 3, port := 0 }),
      ("static", { node := 4, port := 0 }), ("dynamic", { node := 6, port := 0 })]
    inputNames := []
  }
  definitions := [("body", packedSelectionLoopBody)]
}

private def packedSelectionLoopDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 2, rule := .familyPack,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
    { sourceNode := 3, rule := .parallelLoop, arguments := [{ node := 2, port := 0 }] },
    { sourceNode := 4, rule := .familyGetStatic, arguments := [{ node := 3, port := 0 }] },
    { sourceNode := 5, rule := .constantInt, arguments := [] },
    { sourceNode := 6, rule := .familyGetDynamic,
      arguments := [{ node := 3, port := 0 }, { node := 5, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] }
  ] })]
}

/-- A packed matrix family crosses a one-evaluation parallel body as an arena selection. Static
extraction and a dynamically addressed but exactly known index both reduce to the corresponding
lane without restoring the fact-level selected-family path. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts packedSelectionLoopProgram
      packedSelectionLoopDerivation [] []
    let staticMaximum ← matrixMaximum 7 { node := 4, port := 0 } facts
    let dynamic ← lookupFact 7 facts { node := 6, port := 0 }
    let report ← decoderNoiseCheckReportForFact [] facts.arena dynamic [] 2 25
    let familyIsExpression ← match ← lookupFact 7 facts { node := 3, port := 0 } with
      | .familyUniform _ _ (.matrixExpr root) 2 => match facts.arena.get? root with
          | some { node := .select _ (.exact branches), .. } => pure (branches.size == 2)
          | _ => pure false
      | _ => pure false
    pure (staticMaximum, report.obligations, familyIsExpression)) =
    .ok (5, [.decoderThreshold 2 25 3], true) := by
  native_decide

private def selectedSequentialBody : Scope := {
  nodes := #[
    { kind := .input "state", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .input "selector", arguments := [], outputTypes := [.integer] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .select,
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 },
        { node := 3, port := 0 }], outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 4, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("state", { node := 5, port := 0 })]
  inputNames := ["state", "selector"]
}

private def selectedSequentialProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 1), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
      { kind := .sequentialLoop "body" (.constant 2) 0 [] 1,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("result", { node := 2, port := 0 })]
    inputNames := []
  }
  definitions := [("body", selectedSequentialBody)]
}

private def selectedSequentialDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .constantInt, arguments := [] },
    { sourceNode := 2, rule := .sequentialLoop,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .input, arguments := [] },
    { sourceNode := 2, rule := .gaussianSample, arguments := [] },
    { sourceNode := 3, rule := .gaussianSample, arguments := [] },
    { sourceNode := 4, rule := .select,
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 },
        { node := 3, port := 0 }] },
    { sourceNode := 5, rule := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 4, port := 0 }] }
  ] })]
}

/-- A sequential body may contain a selection.  Here its executable selector is statically zero,
so concrete-index reduction selects the bound-three branch before constructing the numeric
recurrence.  Two iterations therefore evaluate `min(q/2, previous + 3)` without retaining a
spurious dynamic alternative. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts selectedSequentialProgram
      selectedSequentialDerivation [] []
    matrixMaximum 3 { node := 2, port := 0 } facts) = .ok 7 := by
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

/-- A one-lane ZipOffset is a statically exact branch and therefore reduces to the corresponding
relation-bearing packed element. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts sharedPreimageBaseScope
      sharedPreimageBaseDerivation [] [fixtureLayout]
    let first ← lookupFact 5 facts { node := 3, port := 0 }
    let second ← lookupFact 5 facts { node := 4, port := 0 }
    let (arena, selected) ← loopTemplateArgumentExpr {} 20 0 1 (.zipOffset 1)
      (packedFacts [first, second])
    match selected, second with
    | .matrixExpr root, .matrix expected => match arena.get? root with
        | some { node := .concrete actual, .. } => pure (actual.origin == expected.origin)
        | _ => pure false
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

/-- A reused expression ID hits its O(1) memo entry without structural comparison. Total and noise
evaluations use separate memo tables and statistics even when they share one evaluation state. The
selection fixtures below separately exercise max-over-complete-branch evaluation. -/
private def operationalExprMemoFixture : Bool :=
  match (do
    let first := operationalExprFixtureFact 0 3
    let arena : OperationalExprArena := { nodes := #[
      { matrixType := fixtureType, node := .concrete first }
    ] }
    let state := ((OperationalExprEvaluationState.empty arena).recordMiss .total).store .total 0 3
    let (secondBound, state) ← evaluateCompleteBound arena [] 0 state
    let state := state.store .noise 0 0
    let (noise, state) ←
      evaluateOperationalExprNoiseBoundWithState arena [] 0 state
    pure (arena.nodes.size, secondBound, noise,
      state.totalStats, state.noiseStats)) with
  | .ok (nodeCount, secondBound, noise, totalStats, noiseStats) =>
      nodeCount == 1 && secondBound == 3 && noise == 0 &&
        totalStats == { evaluations := 1, memoHits := 1, memoMisses := 1 } &&
        noiseStats == { memoHits := 1 }
  | .error _ => false

example : operationalExprMemoFixture = true := by
  simp [operationalExprMemoFixture, operationalExprFixtureFact, fixtureType,
    OperationalExprArena.get?, OperationalExprEvaluationState.empty,
    evaluateCompleteBound, evaluateOperationalExprNoiseBoundWithState,
    evaluateCompleteBoundWithFuel, evaluateOperationalConcreteBound,
    OperationalExprEvaluationState.memo, OperationalExprEvaluationState.recordHit,
    OperationalExprEvaluationState.recordMiss, OperationalExprEvaluationState.store]
  rfl

/-- Different selections remain one binary expression node.  Endpoint evaluation streams complete
branch pairs through the existing addition rule and takes the maximum only after each full sum;
it does not allocate the four-element Cartesian product in the arena. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let first ← matrixFactAt 2 facts { node := 0, port := 0 }
    let second ← matrixFactAt 2 facts { node := 1, port := 0 }
    let (arena, leftFirst) := ({} : OperationalExprArena).pushConcrete first
    let (arena, leftSecond) := arena.pushConcrete second
    let (arena, rightFirst) := arena.pushConcrete first
    let (arena, rightSecond) := arena.pushConcrete second
    let leftSelection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 9, port := 0 }
    }
    let rightSelection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 10, port := 0 }
    }
    let (arena, left) ← arena.pushSelect leftSelection (.exact #[leftFirst, leftSecond])
    let (arena, right) ← arena.pushSelect rightSelection (.exact #[rightFirst, rightSecond])
    let (arena, result) ← addOperationalExprIds 11 0 fixtureType false []
      deriveOperationalSchemaFact arena left right
      (arena.nodes.size + 1)
    let bound ← evaluateOperationalExprNoiseBound arena [] result
    pure (arena.nodes.size, bound)) = .ok (7, 12) := by
  native_decide

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
example : (ChoiceStorage.exact #[4, 7]).staticBranch 2 =
    .error (.invalidCount 0 2) := by
  decide

private def boundedOperationalExprFixtureFact
    (node : Nat)
    (bound : Int) : OperationalMatrixFact :=
  (operationalExprFixtureFact node bound).initializePrimitivePolynomial .bounded

/-- A schema envelope added to an expression that contains an independent selection remains a
binary DAG node. The checker neither distributes the two domains nor evaluates the nested
selection to a stale representative; addition bounds compose directly from the two child bounds. -/
private def envelopePlusNestedSelectionFixture : Bool :=
  match (do
    let envelopeFact := boundedOperationalExprFixtureFact 12 2
    let firstBranch := boundedOperationalExprFixtureFact 13 3
    let secondBranch := boundedOperationalExprFixtureFact 14 5
    let zeroBranch := boundedOperationalExprFixtureFact 15 0
    let (arena, envelopeRepresentative) :=
      ({} : OperationalExprArena).pushConcrete envelopeFact
    let envelopeSelection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 16, port := 0 }
    }
    let envelopeSummary := {
      selectedMatrixSummary #[envelopeFact] with
      selectionOrigin := some (selectionDomainKind envelopeSelection.index)
    }
    let (arena, envelope) ← arena.pushCheckedSchemaEnvelope envelopeSelection 8
      envelopeRepresentative envelopeSummary envelopeFact
    let (arena, firstId) := arena.pushConcrete firstBranch
    let (arena, secondId) := arena.pushConcrete secondBranch
    let nestedSelection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 17, port := 0 }
    }
    let (arena, selected) ← arena.pushSelect nestedSelection (.exact #[firstId, secondId])
    let (arena, zeroId) := arena.pushConcrete zeroBranch
    let (arena, nested) := arena.pushPrimitive 17 0 fixtureType [] (.add false)
      #[selected, zeroId]
    let (arena, result) ← addOperationalExprIds 18 0 fixtureType false []
      deriveOperationalSchemaFact arena envelope nested (arena.nodes.size + 1)
    let resultIsDelayed := match arena.get? result with
      | some { node := .select domain (.shared representative _), .. } =>
          domain.identity == envelopeSelection &&
            (match arena.get? representative with
            | some { node := .primitive operation arguments, .. } =>
                operation.kind == PrimitiveOperationKind.add false &&
                  arguments == #[envelopeRepresentative, nested]
            | _ => false)
      | _ => false
    let bound ← evaluateOperationalExprNoiseBound arena [] result
    pure (resultIsDelayed && bound == 7)) with
  | .ok value => value
  | .error _ => false

/-- Constant-polynomial matrix multiplication preserves the strict nonnegative range needed by a
following coefficient extraction and LUT selection. General polynomial inputs remain unknown
because negacyclic reduction can map a negative coefficient close to the modulus. -/
private def constantPolynomialProductCanonicalRangeFixture : Bool :=
  let constant (node upper : Nat) := {
    operationalExprFixtureFact node (Int.ofNat upper) with
    metadata := { isConstantPolynomial := true }
    canonicalRange := .below upper
  }
  constantPolynomialProductCanonicalRange (constant 20 4) (constant 21 5) == .below 13 &&
    constantPolynomialProductCanonicalRange
      { constant 20 4 with metadata := {} } (constant 21 5) == .unknown

/-- Matrix-product compatibility uses evaluated dimensions, accepts equivalent symbolic syntax,
retains the declared canonical output type, and still rejects a genuinely incompatible product. -/
private def equivalentProductDimensionFixture : Bool :=
  let leftType : MatrixTypeExpr := {
    modulus := .constant 17, ringDimension := .constant 1,
    rows := .constant 1, columns := .constant 2
  }
  let rightType : MatrixTypeExpr := {
    modulus := .constant 17, ringDimension := .constant 1,
    rows := .multiply (.constant 1) (.constant 2), columns := .constant 3
  }
  let incompatibleRightType : MatrixTypeExpr := {
    rightType with rows := .constant 3
  }
  let outputType : MatrixTypeExpr := {
    modulus := .constant 17, ringDimension := .constant 1,
    rows := .constant 1, columns := .constant 3
  }
  match (do
    let base := boundedOperationalExprFixtureFact 22 2
    let (arena, baseId) := ({} : OperationalExprArena).pushConcrete base
    let (arena, left) := arena.pushPrimitive 23 0 leftType [] (.add false) #[baseId, baseId]
    let (arena, right) := arena.pushPrimitive 24 0 rightType [] (.add false) #[baseId, baseId]
    let (arena, accepted) ← multiplyOperationalExprIds 25 0 outputType .matrixMultiplyBound
      { node := 24, port := 0 } [] deriveOperationalSchemaFact arena left right
      (arena.nodes.size + 1)
    let acceptedType ← match arena.get? accepted with
      | some expression => pure expression.matrixType
      | none => throw (OperationalError.invalidOperationalExprRef accepted)
    let (arena, incompatible) := arena.pushPrimitive 26 0 incompatibleRightType [] (.add false)
      #[baseId, baseId]
    let rejected := match multiplyOperationalExprIds 27 0 outputType .matrixMultiplyBound
        { node := 26, port := 0 } [] deriveOperationalSchemaFact arena left incompatible
        (arena.nodes.size + 1) with
      | .error (.operationalExprTypeMismatch _ _) => true
      | _ => false
    pure (acceptedType == outputType && rejected)) with
  | .ok value => value
  | .error _ => false

/-- Exact relation-free alternatives for the same selection are combined branch-wise and may be
recompressed only after the complete post-operation branches prove one uniform schema. The
selection identity and logical branch count remain visible; no Exact value is used as a
representative. -/
private def exactSelectionRecoveredFromEnvelopeFixture : Bool :=
  match (do
    let first := boundedOperationalExprFixtureFact 28 2
    let second := { boundedOperationalExprFixtureFact 29 2 with
      origin := .value temporaryScope { node := 29, port := 0 }
    }
    let (arena, firstId) := ({} : OperationalExprArena).pushConcrete first
    let (arena, secondId) := arena.pushConcrete second
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 30, port := 0 }
    }
    let (arena, exact) ← arena.pushSelect selection (.exact #[firstId, secondId])
    let (arena, output) ← addOperationalExprIds 31 0 fixtureType false []
      deriveOperationalSchemaFact arena exact exact (arena.nodes.size + 1)
    match arena.get? output with
    | some { node := .select actual (.shared representative summary), .. } =>
        let summary ← arena.validatedSchema summary
        let state := OperationalExprEvaluationState.empty arena
        let (fact, _) ← tryUniformRepresentative arena [] representative state
        pure (actual.identity == selection && actual.count == 2 && summary.relationFree &&
          summary.uniformSchema == some (operationalUniformSchema fact))
    | _ => pure false) with
  | .ok value => value
  | .error _ => false

/-- Tensor accepts a checked schema envelope whose representative is an expression DAG, evaluates
that representative before transferring metadata, and preserves the exact selection identity. -/
private def tensorSchemaEnvelopeRepresentativeFixture : Bool :=
  match (do
    let first := boundedOperationalExprFixtureFact 24 2
    let second := boundedOperationalExprFixtureFact 25 3
    let tensorRight := boundedOperationalExprFixtureFact 26 4
    let (arena, firstId) := ({} : OperationalExprArena).pushConcrete first
    let (arena, secondId) := arena.pushConcrete second
    let (arena, representative) := arena.pushPrimitive 27 0 fixtureType [] (.add false)
      #[firstId, secondId]
    let state := OperationalExprEvaluationState.empty arena
    let (representativeFact, _) ←
      tryUniformRepresentative arena [] representative state
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 28, port := 0 }
    }
    let summary := selectedMatrixSummary #[representativeFact]
    let (arena, selected) ← arena.pushCheckedSchemaEnvelope selection 2 representative summary
      representativeFact
    let (arena, result) ← tensorOperationalExprFacts 29 0 fixtureType []
      deriveOperationalSchemaFact arena (.matrixExpr selected) (.matrix tensorRight)
    let root ← match result with
      | .matrixExpr root => pure root
      | _ => throw (OperationalError.unsupportedOperationalExpr arena.nodes.size)
    match arena.get? root with
    | some { node := (.select actualSelection
        (.shared output outputSummary)), .. } => do
        let outputSummary ← arena.validatedSchema outputSummary
        let outputExpression ← match arena.get? output with
          | some expression => pure expression
          | none => throw (OperationalError.invalidOperationalExprRef output)
        let state := OperationalExprEvaluationState.empty arena
        let (outputFact, _) ← tryUniformRepresentative arena [] output state
        pure (actualSelection.identity == selection && actualSelection.count == 2 &&
          (match outputExpression.node with
            | .primitive operation _ => operation.kind == PrimitiveOperationKind.tensor
            | _ => false) &&
          outputSummary.uniformSchema == some (operationalUniformSchema outputFact) &&
          outputSummary.relationFree == !matrixFactHasRelation outputFact &&
          outputSummary.selectionOrigin == some (selectionDomainKind selection.index))
    | _ => throw (OperationalError.unsupportedOperationalExpr root)) with
  | .ok value => value
  | .error _ => false

/-- Two operands controlled by the same selection identity zip branch-wise. Every complete branch
agrees with the explicitly unrolled reference, and the post-operation envelope stores their
maximum without retaining the two concrete identities. -/
private def sameSelectionZipMatchesUnrolledFixture : Bool :=
  match (do
    let leftFirst := boundedOperationalExprFixtureFact 30 1
    let leftSecond := boundedOperationalExprFixtureFact 31 4
    let rightFirst := boundedOperationalExprFixtureFact 32 2
    let rightSecond := boundedOperationalExprFixtureFact 33 1
    let explicitFirst ← addConcreteMatrixFacts 40 0 fixtureType false [] leftFirst rightFirst
    let explicitSecond ← addConcreteMatrixFacts 41 0 fixtureType false [] leftSecond rightSecond
    let explicitBounds ← [explicitFirst, explicitSecond].mapM
      (fun (fact : OperationalMatrixFact) => fact.evaluateNoiseHardBound [])
    let (arena, leftFirstId) := ({} : OperationalExprArena).pushConcrete leftFirst
    let (arena, leftSecondId) := arena.pushConcrete leftSecond
    let (arena, rightFirstId) := arena.pushConcrete rightFirst
    let (arena, rightSecondId) := arena.pushConcrete rightSecond
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 42, port := 0 }
    }
    let (arena, left) ← arena.pushSelect selection (.exact #[leftFirstId, leftSecondId])
    let (arena, right) ← arena.pushSelect selection (.exact #[rightFirstId, rightSecondId])
    let (arena, result) ← addOperationalExprIds 43 0 fixtureType false []
      deriveOperationalSchemaFact arena left right
      (arena.nodes.size + 1)
    let expression ← match arena.get? result with
      | some expression => pure expression
      | none => throw (OperationalError.invalidOperationalExprRef result)
    let selectedMaximum ← match expression.node with
      | .select actualSelection (.shared representative _) => do
          if actualSelection.identity != selection || actualSelection.count != 2 then
            throw (OperationalError.unsupportedOperationalExpr result)
          let fact ← arena.concreteFact representative
          fact.evaluateNoiseHardBound []
      | _ => throw (OperationalError.unsupportedOperationalExpr result)
    pure (selectedMaximum, explicitBounds)) with
  | .ok (selected, explicit) => explicit == [3, 5] && selected == 5
  | .error _ => false

/-- A packed zipped input and a checked uniform zipped input share one loop body without equating
their identities.  The packed input remains the two-way selection, while the uniform operand is
broadcast through each complete branch and the resulting maximum is seven. -/
private def mixedPackedUniformZipFixture : Bool :=
  match (do
    let packed := packedFacts [
      .matrix (boundedOperationalExprFixtureFact 44 3),
      .matrix (boundedOperationalExprFixtureFact 45 5)]
    let uniform : OperationalFact := .familyUniform fixtureFamilyBinder
      (some (.loopBinder temporaryScope 46 1))
      (.matrix (boundedOperationalExprFixtureFact 46 2)) 2
    let (arena, packedInput) ← loopTemplateArgumentExpr {} 47 0 2 .zip packed
    let (arena, uniformInput) ← loopTemplateArgumentExpr arena 47 1 2 .zip uniform
    let (arena, result) ← addOperationalExprFacts 48 0 fixtureType false []
      deriveOperationalSchemaFact arena packedInput uniformInput
    let root ← match result with
      | .matrixExpr root => pure root
      | _ => throw (OperationalError.unsupportedOperationalExpr arena.nodes.size)
    let bound ← evaluateOperationalExprNoiseBound arena [] root
    let branchCount ← match arena.get? root with
      | some { node := .select domain (.shared _ _), .. } => pure domain.count
      | _ => throw (OperationalError.unsupportedOperationalExpr root)
    pure (branchCount, bound)) with
  | .ok (2, 7) => true
  | _ => false

/-- Equal hard bounds or equal schemas do not collapse distinct expression identities.  Only the
complete repeated expression ID used by the earlier fixture is eligible for equal-branch reduction. -/
private def equalBoundDistinctBranchesRemainSelectedFixture : Bool :=
  match (do
    let first := boundedOperationalExprFixtureFact 50 3
    let second := boundedOperationalExprFixtureFact 51 3
    let (arena, firstId) := ({} : OperationalExprArena).pushConcrete first
    let (arena, secondId) := arena.pushConcrete second
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 52, port := 0 }
    }
    let (arena, root) ← arena.pushSelect selection (.exact #[firstId, secondId])
    pure (arena.nodes.size, root, arena.get? root)) with
  | .ok (3, 2, some { node := .select _ (.exact branches), .. }) => branches.size == 2
  | _ => false

/-- Selecting a family constructs one pointwise expression template.  The four logical lanes are
not materialized, the two family alternatives remain exact, and their complete noise bound is the
branch maximum.  The output-lane binder and the family-choice binder are deliberately disjoint. -/
private def uniformMatrixFamilySelectFixture : Bool :=
  match (do
    let selection : OperationalIntegerFact := {
      subject := { node := 70, port := 0 }
      origin := .local temporaryScope { node := 70, port := 0 }
      lower := 0
      upper := 1
      lowerExpression := .closedInt (.constant 0)
      upperExpression := .closedInt (.constant 1)
    }
    let leftBinder : FamilyTemplateBinder := {
      owner := temporaryScope, producerNode := 71, binderSlot := 3
    }
    let rightBinder : FamilyTemplateBinder := {
      owner := temporaryScope, producerNode := 72, binderSlot := 4
    }
    let left : OperationalFact := .familyUniform leftBinder
      (some (.loopBinder temporaryScope 71 3))
      (.matrix (boundedOperationalExprFixtureFact 71 3)) 4
    let right : OperationalFact := .familyUniform rightBinder
      (some (.loopBinder temporaryScope 72 4))
      (.matrix (boundedOperationalExprFixtureFact 72 5)) 4
    let (arena, output) ← selectUniformMatrixFamilies temporaryScope 73 selection
      fixtureType 4 [left, right] {}
    let (binder, coordinate, root, count) ← match output with
      | .familyUniform binder coordinate (.matrixExpr root) count =>
          pure (binder, coordinate, root, count)
      | _ => throw (OperationalError.loopInputModeMismatch 73 0)
    let alternatives ← match arena.get? root with
      | some { node := .select actual (.exact branches), .. } =>
          if actual.index == selection.origin then pure branches.size
          else throw (OperationalError.unsupportedOperationalExpr root)
      | _ => throw (OperationalError.unsupportedOperationalExpr root)
    let bound ← evaluateOperationalExprNoiseBound arena [] root
    pure (binder.binderSlot == 0 &&
      coordinate == some (LoopCoordinate.loopBinder temporaryScope 73 0) &&
      count == 4 && alternatives == 2 && bound == 5 && arena.nodes.size <= 5)) with
  | .ok value => value
  | .error _ => false

/-- An incomplete envelope summary cannot be promoted from one representative. -/
private def incompleteEnvelopeRejectedFixture : Bool :=
  let representative := boundedOperationalExprFixtureFact 60 3
  let source := selectedMatrixSummary #[representative]
  let incomplete := { source with uniformSchema := none }
  let (arena, representativeId) :=
    ({} : OperationalExprArena).pushConcrete representative
  let selection : DynamicSelectionIdentity := {
    index := .local temporaryScope { node := 61, port := 0 }
  }
  match arena.pushSharedSelection selection 2 representativeId incomplete with
  | .error (.unsupportedOperationalExpr 0) => true
  | _ => false

private def endpointIdentityFixtureFact
    (node : Nat)
    (identity : PublicMatrixIdentity) : OperationalMatrixFact :=
  ({ operationalExprFixtureFact node 8 with identity := some identity })
    |>.initializePrimitivePolynomial .large

/-- Structurally equal, relation-free signal alternatives with different public boundaries stay
exact. Compressing them through the first representative would incorrectly turn that branch-local
identity into a selection-wide relation boundary. -/
private def distinctPublicBoundariesRemainExactFixture : Bool :=
  match (do
    let first := endpointIdentityFixtureFact 68 fixtureSampledIdentity
    let secondIdentity : PublicMatrixIdentity :=
      .sampledTrapdoor (.root (.standalone 8)) { node := 0, port := 0 }
    let second := endpointIdentityFixtureFact 69 secondIdentity
    let (arena, firstId) := ({} : OperationalExprArena).pushConcrete first
    let (arena, secondId) := arena.pushConcrete second
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 70, port := 0 }
    }
    let (arena, root) ← arena.pushPrimitiveSelection selection fixtureType [] #[firstId, secondId]
    match arena.get? root with
    | some { node := .select actual (.exact branches), .. } =>
        pure (actual == selection && branches == #[firstId, secondId])
    | _ => pure false) with
  | .ok true => true
  | _ => false

/-- Endpoint identity validation is universal over complete selected alternatives: two matching
branches pass, while replacing only one branch with a different public identity rejects the whole
endpoint. -/
private def oneBadEndpointIdentityRejectsFixture : Bool :=
  match (do
    let expected := fixtureSampledIdentity
    let matchingFirst := endpointIdentityFixtureFact 70 expected
    let matchingSecond := endpointIdentityFixtureFact 71 expected
    let different : PublicMatrixIdentity :=
      .sampledTrapdoor (.root (.standalone 8)) { node := 0, port := 0 }
    let mismatching := endpointIdentityFixtureFact 72 different
    let (goodArena, goodFirst) := ({} : OperationalExprArena).pushConcrete matchingFirst
    let (goodArena, goodSecond) := goodArena.pushConcrete matchingSecond
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 73, port := 0 }
    }
    let (goodArena, goodRoot) ←
      goodArena.pushSelect selection (.exact #[goodFirst, goodSecond])
    requireOperationalBoundaryPublicIdentity goodArena [] 74 expected (.matrixExpr goodRoot)
    let (badArena, badFirst) := ({} : OperationalExprArena).pushConcrete matchingFirst
    let (badArena, badSecond) := badArena.pushConcrete mismatching
    let (badArena, badRoot) ← badArena.pushSelect selection (.exact #[badFirst, badSecond])
    let rejected := match requireOperationalBoundaryPublicIdentity badArena [] 74 expected
        (.matrixExpr badRoot) with
      | .error (.publicIdentityMismatch 74) => true
      | _ => false
    pure rejected) with
  | .ok result => result
  | .error _ => false

private def buildTwoWayScanExpression :
    Nat → OperationalExprArena → OperationalExprId →
      Except OperationalError (OperationalExprArena × OperationalExprId)
  | 0, arena, root => pure (arena, root)
  | remaining + 1, arena, root => do
      let step := remaining
      let first := boundedOperationalExprFixtureFact (100 + 2 * step) 1
      let second := boundedOperationalExprFixtureFact (101 + 2 * step) 2
      let (arena, firstId) := arena.pushConcrete first
      let (arena, secondId) := arena.pushConcrete second
      let selection : DynamicSelectionIdentity := {
        index := .local temporaryScope { node := 200 + step, port := 0 }
      }
      let (arena, selected) ← arena.pushSelect selection (.exact #[firstId, secondId])
      let (arena, next) ← addOperationalExprIds (300 + step) 0 fixtureType false []
        deriveOperationalSchemaFact arena root selected (arena.nodes.size + 1)
      buildTwoWayScanExpression remaining arena next

/-- Eight independent two-way scan steps retain linear arena size and linear bound evaluation.
Each complete two-way result is summarized before the next independent selection is introduced;
the complete result is one plus eight times the larger branch bound, namely seventeen. -/
private def twoWayScanExpressionFixtureResult := do
    let initial := boundedOperationalExprFixtureFact 99 1
    let (arena, root) := ({} : OperationalExprArena).pushConcrete initial
    let (arena, root) ← buildTwoWayScanExpression 8 arena root
    let rootContainsChoice ← arena.containsSelection root
    let (bound, state) ← evaluateOperationalExprNoiseBoundWithState arena [] root
      (OperationalExprEvaluationState.empty arena)
    pure (arena.nodes.size, rootContainsChoice, bound, state.noiseStats.evaluations)

private def twoWayScanExpressionIsLinearFixture : Bool :=
  match twoWayScanExpressionFixtureResult with
  | .ok (size, true, 17, evaluations) => size <= 48 && evaluations <= size
  | _ => false

/-- A relation selected under one executable identity cannot be consumed by a public matrix
selected under another identity, even when their underlying unselected gadget identities match. -/
private def crossSelectionRelationMismatchFixtureResult := do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    let publicMatrix ← matrixFactAt 3 facts { node := 1, port := 0 }
    let preimage ← matrixFactAt 3 facts { node := 2, port := 0 }
    let binder : FamilyTemplateBinder := {
      owner := temporaryScope, producerNode := 80, binderSlot := 0
    }
    let leftSelection : OperationalValueOrigin := .local temporaryScope { node := 81, port := 0 }
    let rightSelection : OperationalValueOrigin := .local temporaryScope { node := 82, port := 0 }
    let selectedPublic := selectDynamicMatrixFact binder leftSelection { node := 83, port := 0 }
      publicMatrix
    let matchingPreimage := selectDynamicMatrixFact binder leftSelection { node := 84, port := 0 }
      preimage
    let mismatchingPreimage := selectDynamicMatrixFact binder rightSelection
      { node := 85, port := 0 } preimage
    let _ ← multiplyConcreteMatrixFacts 86 0 fixtureType
      (.matrixMultiplyRelation { node := 84, port := 0 }) { node := 84, port := 0 } []
      selectedPublic matchingPreimage
    let rejected := match multiplyConcreteMatrixFacts 87 0 fixtureType
        (.matrixMultiplyRelation { node := 85, port := 0 }) { node := 85, port := 0 } []
        selectedPublic mismatchingPreimage with
      | .error (.missingRelation 87 { node := 85, port := 0 }) => true
      | _ => false
    pure (true, rejected)

private def crossSelectionRelationMismatchFixture : Bool :=
  match crossSelectionRelationMismatchFixtureResult with
  | .ok (positive, rejected) => positive && rejected
  | .error _ => false

/-- The Tall-size LUT family is checked once, remains logically 30,720 distinct relation-bearing
lane instances, and stores only one representative after uniform-schema validation. -/
private def tallLutEnvelopeFixtureResult :
    Except OperationalError (Nat × Nat × Bool × Bool × Bool × Bool) := do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    let preimage ← matrixFactAt 3 facts { node := 2, port := 0 }
    let binder : FamilyTemplateBinder := {
      owner := temporaryScope, producerNode := 88, binderSlot := 0
    }
    let selection : OperationalValueOrigin := .local temporaryScope { node := 89, port := 0 }
    let first := selectDynamicMatrixFact binder selection { node := 90, port := 0 } preimage
    let second := selectDynamicMatrixFact binder selection { node := 91, port := 0 } preimage
    let checked := selectedMatrixSummary #[first, second]
    if first == second || checked.uniformSchema.isNone ||
        checked.sharedFirstRelationPublicIdentity.isNone then
      throw (OperationalError.unsupportedOperationalExpr 0)
    let (arena, representative) := ({} : OperationalExprArena).pushConcrete first
    let (arena, root) ← arena.pushSharedSelection
      ({ index := selection } : DynamicSelectionIdentity) 30720 representative checked
    match arena.get? root with
    | some { node := .select actual (.shared representative summary), .. } =>
        let summary ← arena.validatedSchema summary
        let relationBearing ← arena.concreteFact representative
        pure (arena.nodes.size, actual.count, actual.index == selection, !summary.relationFree,
          matrixFactHasRelation relationBearing, first != second)
    | _ => pure (0, 0, false, false, false, false)

private def tallLutEnvelopeFixture : Bool :=
  match tallLutEnvelopeFixtureResult with
  | .ok (2, 30720, true, true, true, true) => true
  | _ => false

/-- Correlated subterms are combined inside each complete branch before the mutually exclusive
maximum.  Independently maximizing the two sides would incorrectly return twenty. -/
private def completeBranchMaximumFixture : Bool :=
  match (do
    let leftFirst := boundedOperationalExprFixtureFact 90 10
    let leftSecond := boundedOperationalExprFixtureFact 91 0
    let rightFirst := boundedOperationalExprFixtureFact 92 0
    let rightSecond := boundedOperationalExprFixtureFact 93 10
    let (arena, leftFirstId) := ({} : OperationalExprArena).pushConcrete leftFirst
    let (arena, leftSecondId) := arena.pushConcrete leftSecond
    let (arena, rightFirstId) := arena.pushConcrete rightFirst
    let (arena, rightSecondId) := arena.pushConcrete rightSecond
    let selection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 94, port := 0 }
    }
    let (arena, left) ← arena.pushSelect selection (.exact #[leftFirstId, leftSecondId])
    let (arena, right) ← arena.pushSelect selection (.exact #[rightFirstId, rightSecondId])
    let (arena, result) ← addOperationalExprIds 95 0 fixtureType false []
      deriveOperationalSchemaFact arena left right (arena.nodes.size + 1)
    evaluateOperationalExprNoiseBound arena [] result) with
  | .ok 10 => true
  | _ => false

/-- The summary-transfer registry has an explicit fail-closed row for every operation category
used by the Tall inventory; no registered category falls through to the unregistered behavior. -/
private def summaryTransferRegistryCoverageFixture : Bool :=
  let representative := boundedOperationalExprFixtureFact 96 3
  let source := selectedMatrixSummary #[representative]
  let registered := #[
    EnvelopeSummaryTransferOperation.instantiationMap,
    .recurrenceBoundShift, .addSubtract, .multiplyRelation, .tensor, .concat, .transform,
    .scale, .bggGrouping, .preimage, .decomposition]
  registered.all (fun operation =>
    (transferSelectedMatrixSummary operation #[source] representative).isSome) &&
    (transferSelectedMatrixSummary .unregistered #[source] representative).isNone

/-- One transform invocation visits a shared child once. Separate lane invocations use isolated
sparse memos, so an earlier lane's mapped value cannot be reused by a later lane. -/
private def transformMemoInvocationIsolationFixture : Bool :=
  match (do
    let source := boundedOperationalExprFixtureFact 110 3
    let (arena, child) := ({} : OperationalExprArena).pushConcrete source
    let (arena, root) := arena.pushPrimitive 110 0 fixtureType [] (.add false) #[child, child]
    let mapLane (lane : Nat) (arena : OperationalExprArena) :=
      mapOperationalExprM s!"fixture-lane:{lane}" .instantiationMap arena root (fun fact => pure {
        fact with
        subject := { node := lane, port := 0 }
        origin := .value temporaryScope { node := lane, port := 0 }
      })
    let (arena, firstRoot) ← mapLane 111 arena
    let firstHits := arena.transformCacheHits
    let (arena, secondRoot) ← mapLane 112 arena
    let secondHits := arena.transformCacheHits
    let (arena, repeatedFirstRoot) ← mapLane 111 arena
    let childOrigin (root : OperationalExprId) := do
      let expression ← match arena.get? root with
        | some expression => pure expression
        | none => throw (OperationalError.invalidOperationalExprRef root)
      let child : OperationalExprId ← match expression.node with
        | .primitive operation arguments =>
            if operation.kind == PrimitiveOperationKind.add false then
              match arguments[0]? with
              | some left => pure left
              | none => throw (OperationalError.unsupportedOperationalExpr root)
            else throw (OperationalError.unsupportedOperationalExpr root)
        | _ => throw (OperationalError.unsupportedOperationalExpr root)
      return (← arena.concreteFact child).origin
    let firstCacheWorked := firstHits > 0
    let lanesDiffer := (← childOrigin firstRoot) != (← childOrigin secondRoot)
    let secondCacheWorked := secondHits > firstHits
    let repeatedLaneMatches :=
      (← childOrigin repeatedFirstRoot) == (← childOrigin firstRoot)
    pure (firstCacheWorked && lanesDiffer && secondCacheWorked && repeatedLaneMatches)) with
  | .ok true => true
  | _ => false

/-- Generic endpoint traversal never forms the Cartesian product of two independent selections;
identity-sensitive consumers must use a dedicated selection rule or fail closed. -/
private def independentSelectionCartesianRejectsFixture : Bool :=
  match (do
    let first := boundedOperationalExprFixtureFact 120 1
    let second := boundedOperationalExprFixtureFact 121 2
    let third := boundedOperationalExprFixtureFact 122 3
    let (arena, firstId) := ({} : OperationalExprArena).pushConcrete first
    let (arena, secondId) := arena.pushConcrete second
    let (arena, thirdId) := arena.pushConcrete third
    let leftSelection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 123, port := 0 }
    }
    let rightSelection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 124, port := 0 }
    }
    let (arena, left) ← arena.pushSelect leftSelection (.exact #[firstId, secondId])
    let (arena, right) ← arena.pushSelect rightSelection (.exact #[firstId, secondId, thirdId])
    let (arena, root) := arena.pushPrimitive 125 0 fixtureType [] (.add false) #[left, right]
    foldOperationalExprConcreteFacts arena [] root 0 fun count _ => pure (count + 1)) with
  | .error (.unsupportedOperationalExpr _) => true
  | _ => false

/-- Schema-uniform families retain exact branch identities beyond the envelope threshold. A
statically exact lane can consume its matching preimage relation, while a neighboring lane cannot
be substituted merely because the schemas and bounds are equal. -/
private def largeUniformFamilyExactRelationFixture : Bool :=
  match (do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    let publicMatrix ← matrixFactAt 3 facts { node := 1, port := 0 }
    let preimage ← matrixFactAt 3 facts { node := 2, port := 0 }
    let binder : FamilyTemplateBinder := {
      owner := temporaryScope, producerNode := 130, binderSlot := 0
    }
    let lanes := Array.range 65
    let publicBranches := lanes.map fun lane =>
      let selection : OperationalValueOrigin :=
        .protocolFamilyElement { name := "fixture-lane" } lane
      OperationalFact.matrix (selectDynamicMatrixFact binder selection { node := 140 + lane, port := 0 }
        publicMatrix)
    let preimageBranches := lanes.map fun lane =>
      let selection : OperationalValueOrigin :=
        .protocolFamilyElement { name := "fixture-lane" } lane
      OperationalFact.matrix (selectDynamicMatrixFact binder selection { node := 240 + lane, port := 0 }
        preimage)
    let publicFamily := packedOperationalFamily publicBranches
    let preimageFamily := packedOperationalFamily preimageBranches
    let (arena, selectedPublic) ← loopTemplateArgumentExpr {} 340 0 1 (.zipOffset 64)
      publicFamily
    let (arena, selectedPreimage) ← loopTemplateArgumentExpr arena 341 0 1 (.zipOffset 64)
      preimageFamily
    let (arena, wrongPreimage) ← loopTemplateArgumentExpr arena 342 0 1 (.zipOffset 63)
      preimageFamily
    let concrete (fact : OperationalFact) := match fact with
      | .matrixExpr root => arena.concreteFact root
      | _ => throw (OperationalError.operandNotMatrix 0 { node := 0, port := 0 })
    let selectedPublic ← concrete selectedPublic
    let selectedPreimage ← concrete selectedPreimage
    let wrongPreimage ← concrete wrongPreimage
    let _ ← multiplyConcreteMatrixFacts 343 0 fixtureType
      (.matrixMultiplyRelation selectedPreimage.subject) selectedPreimage.subject []
      selectedPublic selectedPreimage
    let rejected := match multiplyConcreteMatrixFacts 344 0 fixtureType
        (.matrixMultiplyRelation wrongPreimage.subject) wrongPreimage.subject []
        selectedPublic wrongPreimage with
      | .error (.missingRelation 344 _) => true
      | _ => false
    pure (publicBranches.size == 65 && preimageBranches.size == 65 && rejected)) with
  | .ok true => true
  | _ => false

/-! Reuse one pre-existing native fixture gate for the computationally heavy operational
fixtures.  This keeps the trusted-evaluation surface unchanged while checking the production
functions rather than duplicating their behavior in proof-only reference code. -/
example : exactRelationSelectionFixtureResult = .ok true ∧
    constantPolynomialProductCanonicalRangeFixture = true ∧
    equivalentProductDimensionFixture = true ∧
    exactSelectionRecoveredFromEnvelopeFixture = true ∧
    tensorSchemaEnvelopeRepresentativeFixture = true ∧
    sameSelectionZipMatchesUnrolledFixture = true ∧
    mixedPackedUniformZipFixture = true ∧
    equalBoundDistinctBranchesRemainSelectedFixture = true ∧
    uniformMatrixFamilySelectFixture = true ∧
    incompleteEnvelopeRejectedFixture = true ∧
    distinctPublicBoundariesRemainExactFixture = true ∧
    oneBadEndpointIdentityRejectsFixture = true ∧
    twoWayScanExpressionIsLinearFixture = true ∧
    crossSelectionRelationMismatchFixture = true ∧
    tallLutEnvelopeFixture = true ∧
    completeBranchMaximumFixture = true ∧
    summaryTransferRegistryCoverageFixture = true ∧
    transformMemoInvocationIsolationFixture = true ∧
    envelopePlusNestedSelectionFixture = true ∧
    independentSelectionCartesianRejectsFixture = true ∧
    largeUniformFamilyExactRelationFixture = true := by
  native_decide

end Mxx.Certificate
