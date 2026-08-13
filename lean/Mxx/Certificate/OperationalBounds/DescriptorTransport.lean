import Mxx.Certificate.OperationalBounds.DirectCarrier

namespace Mxx.Certificate

open Mxx.Ir

/-- The capture-free identity transport for one complete owner-aware context.  This is the
starting point for a final-reduction map stack: every binder is retained by its full owner,
slot, and declared domain, never reconstructed from its numeric slot alone. -/
def identityIndexMap (context : IndexContext) : IndexMap := {
  source := context
  destination := context
  assignments := context.binders.map (fun binder => IndexExpr.variable binder)
}

/-- Normalize one ordered lazy transport stack by composing adjacent maps exactly when the
intermediate owner-aware contexts agree.  A non-composable pair remains ordered; callers must
apply it inner-to-outer at the fixed-leaf boundary rather than weakening its provenance. -/
def normalizeIndexMapStack : List IndexMap → List IndexMap
  | [] => []
  | map :: tail =>
      match normalizeIndexMapStack tail with
      | next :: remaining =>
          match composeIndexMap map next with
          | some composed => composed :: remaining
          | none => map :: next :: remaining
      | [] => [map]

/-- Apply an ordered pending transport stack at a fixed-leaf boundary.  The first map is nearest
the leaf and is therefore applied first; this preserves nested gather positions and owner-aware
domain checks exactly as repeated eager transport did. -/
def reindexIndexExprStack
    (maps : List IndexMap) (expression : IndexExpr) : Option IndexExpr :=
  maps.foldlM (fun expression map => reindex map expression) expression

/-- A stack is admissible only when every map is a checked capture-free transport and consecutive
contexts join.  Non-composable maps are retained (not rewritten) so callers can reject them at
the descriptor boundary rather than guessing an owner correspondence. -/
def indexMapStackValid : List IndexMap → Bool
  | [] => true
  | [map] => map.transportValid
  | first :: second :: tail =>
      first.transportValid && second.transportValid && first.destination == second.source &&
        indexMapStackValid (second :: tail)

/-- Lift an ordinary Graph-IR parameter expression into the direct-carrier descriptor language.
This is kept beside the transport functions because every delayed descriptor must cross the same
owner-aware index boundary before it may be stored or reduced. -/
def indexedParameterOfIr (value : IntExpr) : IndexedParameterExpr := .ir value

def indexedMatrixTypeOfIr (value : MatrixTypeExpr) : IndexedMatrixTypeExpr := {
  modulus := .ir value.modulus
  ringDimension := .ir value.ringDimension
  rows := .ir value.rows
  columns := .ir value.columns
}

/-- Transport every owner-bearing dimension of a delayed matrix descriptor. -/
def reindexIndexedMatrixTypeExpr
    (map : IndexMap) (value : IndexedMatrixTypeExpr) : Option IndexedMatrixTypeExpr := do
  pure {
    modulus := ← value.modulus.reindex map
    ringDimension := ← value.ringDimension.reindex map
    rows := ← value.rows.reindex map
    columns := ← value.columns.reindex map
  }

/-- Transport contextual bound domains without projecting binders back to numeric loop slots. -/
def reindexIndexedOperationalParameterDomains
    (map : IndexMap) : List IndexedOperationalParameterDomain → Option (List IndexedOperationalParameterDomain)
  | [] => some []
  | .loopIndex binder :: tail => do
      let tail ← reindexIndexedOperationalParameterDomains map tail
      let expression ← map.assignmentFor binder
      match expression with
      | .variable destination =>
          if tail.any (fun candidate => candidate == .loopIndex destination) then some tail
          else some (.loopIndex destination :: tail)
      | .constant _ => some tail
      | _ => none
  | .parameter name environment domains expression :: tail => do
      let tail ← reindexIndexedOperationalParameterDomains map tail
      let domains ← reindexIndexedOperationalParameterDomains map domains
      let expression ← expression.reindex map
      let candidate := IndexedOperationalParameterDomain.parameter name environment domains expression
      match tail.filter (fun existing => match existing with
        | .parameter existingName _ _ _ => existingName == name
        | .loopIndex _ => false) with
      | [] => some (candidate :: tail)
      | [existing] => if existing == candidate then some tail else none
      | _ => none

/-- Transport the owner-aware parameter frame embedded in a delayed deterministic-hash query.
The key is an `IndexExpr`, not a numeric slot, so gather substitutions remain representable. -/
def reindexIndexedParamEnvironment
    (map : IndexMap) : IndexedParamEnvironment → Option IndexedParamEnvironment
  | [] => some []
  | (.parameter name, value) :: remaining => do
      let remaining ← reindexIndexedParamEnvironment map remaining
      insertIndexedParamBinding (.parameter name, value) remaining
  | (.index expression, value) :: remaining => do
      let remaining ← reindexIndexedParamEnvironment map remaining
      insertIndexedParamBinding (.index (← reindex map expression), value) remaining

def reindexIndexedOperationalBoundExpr
    (map : IndexMap) : IndexedOperationalBoundExpr → Option IndexedOperationalBoundExpr
  | .closedInt value => return .closedInt (← value.reindex map)
  | .contextual kind environment domains value =>
      return .contextual kind environment (← reindexIndexedOperationalParameterDomains map domains)
        (← value.reindex map)
  | .closedOperational value => some (.closedOperational value)

def sameReindexedContextualDomainKey : OperationalParameterDomain → OperationalParameterDomain → Bool
  | .loopIndex left _, .loopIndex right _ => left == right
  | .parameter left _ _ _, .parameter right _ _ _ => left == right
  | _, _ => false

private def indexedParameterFreeVariables : IndexedParameterExpr → List IndexVariable
  | .ir _ => []
  | .index value => value.freeVariables
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => indexedParameterFreeVariables left ++ indexedParameterFreeVariables right
  | .log2Ceil value => indexedParameterFreeVariables value

private def indexedMatrixTypeFreeVariables (value : IndexedMatrixTypeExpr) : List IndexVariable :=
  indexedParameterFreeVariables value.modulus ++ indexedParameterFreeVariables value.ringDimension ++
    indexedParameterFreeVariables value.rows ++ indexedParameterFreeVariables value.columns

private def indexedParameterEnvironmentFreeVariables : IndexedParamEnvironment → List IndexVariable
  | [] => []
  | (.parameter _, _) :: tail => indexedParameterEnvironmentFreeVariables tail
  | (.index expression, _) :: tail =>
      expression.freeVariables ++ indexedParameterEnvironmentFreeVariables tail

private def indexedOperationalParameterDomainFreeVariables :
    List IndexedOperationalParameterDomain → List IndexVariable
  | [] => []
  | .loopIndex binder :: tail => binder :: indexedOperationalParameterDomainFreeVariables tail
  | .parameter _ _ domains expression :: tail =>
      indexedOperationalParameterDomainFreeVariables domains ++ indexedParameterFreeVariables expression ++
        indexedOperationalParameterDomainFreeVariables tail

private def mapOwnsAllOrNoFreeVariables
    (map : IndexMap) (freeVariables : List IndexVariable) : Option Bool :=
  let owned := freeVariables.filter map.source.binders.contains
  if owned.isEmpty then some false
  else if owned.length == freeVariables.length then some true
  else none

private def reindexOperationalValueOriginUnchecked
    (map : IndexMap) : OperationalValueOrigin → Option OperationalValueOrigin
  | .local scope wire => some (.local scope wire)
  | .protocolInput input => some (.protocolInput input)
  | .protocolFamilyElement input index =>
      return .protocolFamilyElement input (← reindex map index)
  | .loopInstance slot index source =>
      return .loopInstance slot (← reindex map index) (← reindexOperationalValueOriginUnchecked map source)
  | .indexed binder expression source =>
      return .indexed binder (← reindex map expression)
        (← reindexOperationalValueOriginUnchecked map source)

/-- The contextual binders that occur in a value origin.  A mapped carrier may transport the
origin only when it owns all of these binders; a disjoint enclosing map must leave the semantic
source untouched, while a partial overlap is malformed rather than silently dropping a frame. -/
def operationalValueOriginFreeVariables : OperationalValueOrigin → List IndexVariable
  | .local _ _ | .protocolInput _ => []
  | .protocolFamilyElement _ index => index.freeVariables
  | .loopInstance _ index source => index.freeVariables ++ operationalValueOriginFreeVariables source
  | .indexed _ expression source => expression.freeVariables ++ operationalValueOriginFreeVariables source

/-- Transport one complete value identity only when the map owns all of its index atoms.  A
disjoint enclosing carrier map preserves the identity; a partial overlap is malformed because it
would detach one nested selector from the rest of that exact executable identity. -/
def reindexOperationalValueOrigin
    (map : IndexMap) (origin : OperationalValueOrigin) : Option OperationalValueOrigin := do
  match mapOwnsAllOrNoFreeVariables map (operationalValueOriginFreeVariables origin) with
  | false => some origin
  | true => reindexOperationalValueOriginUnchecked map origin
  | none => none

def reindexDynamicSelectionIdentity
    (map : IndexMap)
    (selection : DynamicSelectionIdentity) : Option DynamicSelectionIdentity := do
  pure {
    selection with
    index := ← reindexOperationalValueOrigin map selection.index
    expression := ← reindex map selection.expression
  }

def indexMapSourceBinderForSlot (map : IndexMap) (slot : Nat) : Option IndexVariable :=
  match (map.source.binders.filter fun binder => binder.slot == slot).toList with
  | [binder] => some binder
  | _ => none

def indexExprAsIntExpr : IndexExpr → Option IntExpr
  | .constant value => some (.constant (Int.ofNat value))
  | .variable binder => some (.loopIndex binder.slot)
  | .offset base amount => do
      let base ← indexExprAsIntExpr base
      if amount < 0 then some (.subtract base (.constant (-amount)))
      else some (.add base (.constant amount))
  | .gather _ _ _ => none

/-- Transport loop-index arithmetic through the same capture-free map as identities.  Gather is
not representable by `IntExpr` and therefore remains fail-closed until bounds use `IndexExpr`
directly. -/
def reindexIntExpr (map : IndexMap) : IntExpr → Option IntExpr
  | .constant value => some (.constant value)
  | .parameter name => some (.parameter name)
  | .loopIndex slot =>
      /- `IntExpr` carries a numeric slot but not an owner.  A capture-free map can rewrite it
      only when that slot occurs exactly once in the map source; otherwise it is not free in
      this transport and must remain intact.  Rejecting it would make an unrelated outer map
      erase loop-local hash and bound descriptors. -/
      match indexMapSourceBinderForSlot map slot with
      | some binder => do
          let expression ← reindex map (.variable binder)
          indexExprAsIntExpr expression
      | none => some (.loopIndex slot)
  | .add left right => return .add (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .subtract left right => return .subtract (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .multiply left right => return .multiply (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .divide left right => return .divide (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .roundDivide left right =>
      return .roundDivide (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .log2Ceil value => return .log2Ceil (← reindexIntExpr map value)

def reindexRealExpr (map : IndexMap) : RealExpr → Option RealExpr
  | .rational value => some (.rational value)
  | .parameter name => some (.parameter name)
  | .fromInt value => .fromInt <$> reindexIntExpr map value
  | .add left right => return .add (← reindexRealExpr map left) (← reindexRealExpr map right)
  | .subtract left right => return .subtract (← reindexRealExpr map left) (← reindexRealExpr map right)
  | .multiply left right => return .multiply (← reindexRealExpr map left) (← reindexRealExpr map right)
  | .divide left right => return .divide (← reindexRealExpr map left) (← reindexRealExpr map right)
  | .sqrt value => .sqrt <$> reindexRealExpr map value

def reindexMatrixTypeExpr (map : IndexMap)
    (matrixType : MatrixTypeExpr) : Option MatrixTypeExpr := do
  pure {
    modulus := ← reindexIntExpr map matrixType.modulus
    ringDimension := ← reindexIntExpr map matrixType.ringDimension
    rows := ← reindexIntExpr map matrixType.rows
    columns := ← reindexIntExpr map matrixType.columns
  }

def reindexedEnvironmentValue
    (key : ParamKey) : ParamEnvironment → Option ParamValue
  | [] => none
  | (candidate, value) :: tail => if candidate == key then some value
      else reindexedEnvironmentValue key tail

def reindexParamEnvironment (map : IndexMap) : ParamEnvironment → Option ParamEnvironment
  | [] => some []
  | (.parameter name, value) :: remaining =>
      return (.parameter name, value) :: (← reindexParamEnvironment map remaining)
  | (.loopIndex slot, value) :: remaining => do
      let tail ← reindexParamEnvironment map remaining
      let binder ← indexMapSourceBinderForSlot map slot
      let mapped ← reindex map (.variable binder)
      match mapped.freeVariables with
      | [] => pure tail
      | [destination] =>
          match reindexedEnvironmentValue (.loopIndex destination.slot) tail with
          | none => pure ((.loopIndex destination.slot, value) :: tail)
          | some existing => if existing == value then pure tail else none
      | _ => none

/-- Insert one transported contextual domain exactly once.  A capture-free substitution may map
multiple source binders to one destination; retaining both would make enumeration order depend on
the source order, while retaining unequal definitions would silently change its meaning. -/
def insertReindexedParameterDomain
    (candidate : OperationalParameterDomain)
    (domains : List OperationalParameterDomain) : Option (List OperationalParameterDomain) :=
  match domains with
  | [] => some [candidate]
  | head :: tail =>
      if sameReindexedContextualDomainKey head candidate then
        if head == candidate then some domains else none
      else
        return head :: (← insertReindexedParameterDomain candidate tail)

def reindexParameterDomains
    (transportEnvironment : ParamEnvironment)
    (map : IndexMap) : List OperationalParameterDomain → Option (List OperationalParameterDomain)
  | [] => some []
  | domain :: remaining => do
      let tail ← reindexParameterDomains transportEnvironment map remaining
      match domain with
      | .parameter name sourceEnvironment domains expression =>
          insertReindexedParameterDomain (.parameter name (← reindexParamEnvironment map sourceEnvironment)
            (← reindexParameterDomains transportEnvironment map domains)
            (← reindexIntExpr map expression)) tail
      | .loopIndex slot count =>
          let binder ← indexMapSourceBinderForSlot map slot
          let sourceCount ← binder.count.evaluate transportEnvironment
          if sourceCount <= 0 || sourceCount.toNat != count then none
          let mapped ← reindex map (.variable binder)
          match mapped.freeVariables with
          | [] => pure tail
          | [destination] => do
              let destinationCount ← destination.count.evaluate transportEnvironment
              if destinationCount <= 0 then none
              insertReindexedParameterDomain (.loopIndex destination.slot destinationCount.toNat) tail
          | _ => none

def reindexOperationalBoundExpr
    (transportEnvironment : ParamEnvironment) (map : IndexMap) : OperationalBoundExpr → Option OperationalBoundExpr
  | .closedInt value => return .closedInt (← reindexIntExpr map value)
  | .contextual kind environment domains value =>
      return .contextual kind (← reindexParamEnvironment map environment)
        (← reindexParameterDomains transportEnvironment map domains) (← reindexIntExpr map value)
  | .previous path => some (.previous path)
  | .negate value => return .negate (← reindexOperationalBoundExpr transportEnvironment map value)
  | .add left right =>
      return .add (← reindexOperationalBoundExpr transportEnvironment map left)
        (← reindexOperationalBoundExpr transportEnvironment map right)
  | .subtract left right =>
      return .subtract (← reindexOperationalBoundExpr transportEnvironment map left)
        (← reindexOperationalBoundExpr transportEnvironment map right)
  | .multiply left right =>
      return .multiply (← reindexOperationalBoundExpr transportEnvironment map left)
        (← reindexOperationalBoundExpr transportEnvironment map right)
  | .divide left right =>
      return .divide (← reindexOperationalBoundExpr transportEnvironment map left)
        (← reindexOperationalBoundExpr transportEnvironment map right)
  | .minimum left right =>
      return .minimum (← reindexOperationalBoundExpr transportEnvironment map left)
        (← reindexOperationalBoundExpr transportEnvironment map right)
  | .maximum left right =>
      return .maximum (← reindexOperationalBoundExpr transportEnvironment map left)
        (← reindexOperationalBoundExpr transportEnvironment map right)
  | .centeredCap modulus value =>
      return .centeredCap (← reindexOperationalBoundExpr transportEnvironment map modulus)
        (← reindexOperationalBoundExpr transportEnvironment map value)
  | .matrixProduct ringDimension innerDimension left right =>
      return .matrixProduct (← reindexOperationalBoundExpr transportEnvironment map ringDimension)
        (← reindexOperationalBoundExpr transportEnvironment map innerDimension)
        (← reindexOperationalBoundExpr transportEnvironment map left)
        (← reindexOperationalBoundExpr transportEnvironment map right)
  | .recurrence count initial transition slot =>
      return .recurrence count (← initial.mapM (reindexOperationalBoundExpr transportEnvironment map))
        (← transition.mapM (reindexOperationalBoundExpr transportEnvironment map)) slot
  | .recurrenceState count paths initial transition output =>
      return .recurrenceState count paths (← initial.mapM (reindexOperationalBoundExpr transportEnvironment map))
        (← transition.mapM (reindexOperationalBoundExpr transportEnvironment map)) output

private def deterministicHashFreeVariables (query : DeterministicHashIdentity) : List IndexVariable :=
  operationalValueOriginFreeVariables query.keyOrigin ++
    indexedMatrixTypeFreeVariables query.matrixType ++
    indexedParameterEnvironmentFreeVariables query.parameterEnvironment ++
    indexedOperationalParameterDomainFreeVariables query.parameterDomains ++
    query.tagExpressions.flatMap indexedParameterFreeVariables ++
    query.tagDecimalExpressions.flatMap indexedParameterFreeVariables ++
    query.tagU64LeExpressions.flatMap indexedParameterFreeVariables ++
    query.trailingIntegerOrigins.flatMap operationalValueOriginFreeVariables

def matrixOriginFreeVariables : MatrixOriginIdentity → List IndexVariable
  | .value _ _ | .protocolInput _ => []
  | .protocolFamilyElement _ index => index.freeVariables
  | .deterministicHash query => deterministicHashFreeVariables query
  | .loopInstance _ index source => index.freeVariables ++ matrixOriginFreeVariables source
  | .indexed _ expression source => expression.freeVariables ++ matrixOriginFreeVariables source

private def reindexMatrixOriginIdentityUnchecked
    (environment : ParamEnvironment) (map : IndexMap) : MatrixOriginIdentity → Option MatrixOriginIdentity
  | .value scope wire => some (.value scope wire)
  | .protocolInput input => some (.protocolInput input)
  | .protocolFamilyElement input index =>
      return .protocolFamilyElement input (← reindex map index)
  | .deterministicHash query =>
      return .deterministicHash {
        query with
        keyOrigin := ← reindexOperationalValueOrigin map query.keyOrigin
        matrixType := ← reindexIndexedMatrixTypeExpr map query.matrixType
        parameterEnvironment := ← reindexIndexedParamEnvironment map query.parameterEnvironment
        parameterDomains := ← reindexIndexedOperationalParameterDomains map query.parameterDomains
        tagExpressions := ← query.tagExpressions.mapM (IndexedParameterExpr.reindex map)
        tagDecimalExpressions := ← query.tagDecimalExpressions.mapM (IndexedParameterExpr.reindex map)
        tagU64LeExpressions := ← query.tagU64LeExpressions.mapM (IndexedParameterExpr.reindex map)
        trailingIntegerOrigins := ← query.trailingIntegerOrigins.mapM
          (reindexOperationalValueOriginUnchecked map)
      }
  | .loopInstance slot index source =>
      return .loopInstance slot (← reindex map index)
        (← reindexMatrixOriginIdentityUnchecked environment map source)
  | .indexed binder expression source =>
      return .indexed binder (← reindex map expression)
        (← reindexMatrixOriginIdentityUnchecked environment map source)

def reindexMatrixOriginIdentity
    (environment : ParamEnvironment) (map : IndexMap) (origin : MatrixOriginIdentity) :
    Option MatrixOriginIdentity := do
  match mapOwnsAllOrNoFreeVariables map (matrixOriginFreeVariables origin) with
  | false => some origin
  | true => reindexMatrixOriginIdentityUnchecked environment map origin
  | none => none

def publicMatrixIdentityFreeVariables : PublicMatrixIdentity → List IndexVariable
  | .sampledTrapdoor _ _ | .gadget .. => []
  | .indexed _ expression source => expression.freeVariables ++ publicMatrixIdentityFreeVariables source
  | .loopInstance _ index source => index.freeVariables ++ publicMatrixIdentityFreeVariables source

private def reindexPublicMatrixIdentityUnchecked
    (map : IndexMap) : PublicMatrixIdentity → Option PublicMatrixIdentity
  | .sampledTrapdoor scope wire => some (.sampledTrapdoor scope wire)
  | .gadget paramsId params inputRows base small digitCount =>
      some (.gadget paramsId params inputRows base small digitCount)
  | .indexed binder expression source =>
      return .indexed binder (← reindex map expression)
        (← reindexPublicMatrixIdentityUnchecked map source)
  | .loopInstance slot index source =>
      return .loopInstance slot (← reindex map index) (← reindexPublicMatrixIdentityUnchecked map source)

def reindexPublicMatrixIdentity
    (map : IndexMap) (identity : PublicMatrixIdentity) : Option PublicMatrixIdentity := do
  match mapOwnsAllOrNoFreeVariables map (publicMatrixIdentityFreeVariables identity) with
  | false => some identity
  | true => reindexPublicMatrixIdentityUnchecked map identity
  | none => none

def reindexOperationalPrimitiveIdentityFully
    (transportEnvironment : ParamEnvironment) (map : IndexMap) :
    OperationalPrimitiveIdentity → Option OperationalPrimitiveIdentity
  | .matrix identity => return .matrix (← reindexMatrixOriginIdentity transportEnvironment map identity)
  | .publicMatrix identity => return .publicMatrix (← reindexPublicMatrixIdentity map identity)
  | .value identity => return .value (← reindexOperationalValueOrigin map identity)
  | .parameterScalar sourceEnvironment domains value =>
      return .parameterScalar (← reindexParamEnvironment map sourceEnvironment)
        (← reindexParameterDomains transportEnvironment map domains) (← reindexIntExpr map value)
  | .identityMatrix type => return .identityMatrix (← reindexMatrixTypeExpr map type)
  | .indexedArtifact input index => return .indexedArtifact input (← reindex map index)
  | .recurrenceResult scope node path => some (.recurrenceResult scope node path)
  | .carriedInput path => some (.carriedInput path)

def reindexOperationalCompressionToken
    (environment : ParamEnvironment) (map : IndexMap) : OperationalCompressionToken → Option OperationalCompressionToken
  | .primitive identity =>
      return .primitive (← reindexOperationalPrimitiveIdentityFully environment map identity)
  | .transform value => some (.transform value)
  | .productMode value => some (.productMode value)
  | .intermediateType value => return .intermediateType (← reindexMatrixTypeExpr map value)
  | .productStart => some .productStart
  | .productEnd => some .productEnd
  | .groupStart => some .groupStart
  | .groupEnd => some .groupEnd
  | .sumStart => some .sumStart
  | .sumEnd => some .sumEnd
  | .termStart coefficient => some (.termStart coefficient)
  | .termEnd => some .termEnd
  | .summaryBound bound =>
      return .summaryBound (← reindexOperationalBoundExpr environment map bound)
  | .summaryMetadata metadata => some (.summaryMetadata metadata)
  | .segmentStart kind length => some (.segmentStart kind length)
  | .segmentEnd => some .segmentEnd

def reindexOperationalBoundedSummary
    (environment : ParamEnvironment) (map : IndexMap)
    (summary : OperationalBoundedFactorSummary) : Option OperationalBoundedFactorSummary := do
  pure { summary with
    matrixType := ← reindexMatrixTypeExpr map summary.matrixType
    hardBound := ← reindexOperationalBoundExpr environment map summary.hardBound
    provenance := ← summary.provenance.mapM (reindexOperationalCompressionToken environment map)
  }

def reindexRelationSnapshotPolynomial
    (environment : ParamEnvironment) (map : IndexMap)
    (polynomial : RelationSnapshotPolynomial) : Option RelationSnapshotPolynomial :=
  polynomial.mapM fun term => do
    let factors ← term.product.factors.mapM fun factor => do
      let leaf : RelationSnapshotFactorLeaf ← match factor.leaf with
        | .primitive identity =>
            pure (.primitive (← reindexOperationalPrimitiveIdentityFully environment map identity))
        | .boundedSummary origin summary =>
            pure (.boundedSummary
              { origin with tokens := ← (origin.tokens.mapM
                (reindexOperationalCompressionToken environment map)) }
              (← reindexOperationalBoundedSummary environment map summary))
        | .exactTransform tokens type =>
            pure (.exactTransform (← tokens.mapM (reindexOperationalCompressionToken environment map))
              (← reindexMatrixTypeExpr map type))
      pure { factor with
        leaf
        inputType := ← reindexMatrixTypeExpr map factor.inputType
        outputType := ← reindexMatrixTypeExpr map factor.outputType
        boundedSummary := ← factor.boundedSummary.mapM (reindexOperationalBoundedSummary environment map)
      }
    pure { term with product := { term.product with
      factors
      outputType := ← reindexMatrixTypeExpr map term.product.outputType
    } }

def reindexRelationTargetSummary
    (environment : ParamEnvironment) (map : IndexMap)
    (summary : RelationTargetSummary) : Option RelationTargetSummary := do
  pure { summary with
    origin := ← reindexMatrixOriginIdentity environment map summary.origin
    matrixType := ← reindexMatrixTypeExpr map summary.matrixType
    totalHardBound := ← reindexOperationalBoundExpr environment map summary.totalHardBound
    polynomial := ← reindexRelationSnapshotPolynomial environment map summary.polynomial
  }

def reindexOperationalMatrixRelation
    (environment : ParamEnvironment) (map : IndexMap) : OperationalMatrixRelation → Option OperationalMatrixRelation
  | .decomposition relation =>
      return .decomposition {
        relation with
        producer := ← reindexMatrixOriginIdentity environment map relation.producer
        publicIdentity := ← reindexPublicMatrixIdentity map relation.publicIdentity
        inputOrigin := ← reindexMatrixOriginIdentity environment map relation.inputOrigin
        inputSummary := ← reindexRelationTargetSummary environment map relation.inputSummary
      }
  | .preimage relation =>
      return .preimage {
        relation with
        producer := ← reindexMatrixOriginIdentity environment map relation.producer
        publicIdentity := ← reindexPublicMatrixIdentity map relation.publicIdentity
        targetOrigin := ← reindexMatrixOriginIdentity environment map relation.targetOrigin
        targetSummary := ← reindexRelationTargetSummary environment map relation.targetSummary
      }

def reindexOperationalPolynomial
    (environment : ParamEnvironment) (map : IndexMap)
    (polynomial : OperationalPolynomial) : Option OperationalPolynomial :=
  polynomial.mapM fun term => do
    let factors ← term.product.factors.mapM fun factor => do
      let leaf : OperationalFactorLeaf ← match factor.leaf with
        | .primitive identity =>
            pure (.primitive (← reindexOperationalPrimitiveIdentityFully environment map identity))
        | .boundedSummary origin summary =>
            pure (.boundedSummary
              { origin with tokens := ← (origin.tokens.mapM
                (reindexOperationalCompressionToken environment map)) }
              (← reindexOperationalBoundedSummary environment map summary))
        | .exactTransform tokens type =>
            pure (.exactTransform (← tokens.mapM (reindexOperationalCompressionToken environment map))
              (← reindexMatrixTypeExpr map type))
      pure { factor with
        leaf
        inputType := ← reindexMatrixTypeExpr map factor.inputType
        outputType := ← reindexMatrixTypeExpr map factor.outputType
        boundedSummary := ← factor.boundedSummary.mapM (reindexOperationalBoundedSummary environment map)
        relations := ← factor.relations.mapM (reindexOperationalMatrixRelation environment map)
      }
    pure { term with product := { term.product with
      factors
      outputType := ← reindexMatrixTypeExpr map term.product.outputType
    } }

def reindexOperationalBlockPartition
    (environment : ParamEnvironment) (map : IndexMap)
    (partition : OperationalBlockPartition) : Option OperationalBlockPartition := do
  pure {
    matrixType := ← reindexMatrixTypeExpr map partition.matrixType
    polynomial := ← reindexOperationalPolynomial environment map partition.polynomial
  }

def reindexOperationalBlockLayout
    (environment : ParamEnvironment) (map : IndexMap)
    (layout : OperationalBlockLayout) : Option OperationalBlockLayout := do
  pure { layout with partitions := ← (layout.partitions.mapM
    (reindexOperationalBlockPartition environment map)) }

/-- Exhaustive transport for a matrix payload.  Old selected identities return `none` above, so
callers cannot accidentally retain a pre-indexed selector in an otherwise reindexed fact. -/
def reindexOperationalMatrixFact
    (environment : ParamEnvironment) (map : IndexMap)
    (fact : OperationalMatrixFact) : Option OperationalMatrixFact :=
  if map.isDirectCarrierContextLift then some fact else do
  let origin ← reindexMatrixOriginIdentity environment map fact.origin
  let matrixType ← reindexMatrixTypeExpr map fact.matrixType
  let totalHardBound ← reindexOperationalBoundExpr environment map fact.totalHardBound
  let identity ← fact.identity.mapM (reindexPublicMatrixIdentity map)
  let relations ← fact.relations.mapM (reindexOperationalMatrixRelation environment map)
  let polynomial ← reindexOperationalPolynomial environment map fact.polynomial
  let blockLayout ← fact.blockLayout.mapM (reindexOperationalBlockLayout environment map)
  pure { fact with
    origin, matrixType, totalHardBound, identity, relations, polynomial, blockLayout
  }

/-- Reindex scalar identity/bound fields when a direct carrier map specializes its selector. -/
def reindexOperationalScalarFact
    (_environment : ParamEnvironment) (map : IndexMap) : OperationalScalarFact → Option OperationalScalarFact :=
  fun fact => if map.isDirectCarrierContextLift then some fact else match fact with
  | .integer fact => do
      pure (.integer { fact with
        origin := ← reindexOperationalValueOrigin map fact.origin
        lowerExpression := ← reindexIndexedOperationalBoundExpr map fact.lowerExpression
        upperExpression := ← reindexIndexedOperationalBoundExpr map fact.upperExpression })
  | .trapdoor fact => do
      let preimageCutoff ← match fact.preimageCutoff with
        | none => pure none | some cutoff => reindexIndexedOperationalBoundExpr map cutoff
      pure (.trapdoor { fact with
        matrixType := ← reindexMatrixTypeExpr map fact.matrixType
        sigma := ← reindexRealExpr map fact.sigma
        gadgetBase := ← reindexIntExpr map fact.gadgetBase
        digitCount := ← reindexIntExpr map fact.digitCount
        preimageMaxCoefficientBound := ← reindexIntExpr map fact.preimageMaxCoefficientBound
        maximum := ← reindexIndexedOperationalBoundExpr map fact.maximum
        preimageCutoff
        publicIdentity := ← reindexPublicMatrixIdentity map fact.publicIdentity })
  | .bytes fact => do
      pure (.bytes { fact with origin := ← reindexOperationalValueOrigin map fact.origin })
  | .boolean => some .boolean
  | .real => some .real
  | .typedBlob typeName schemaHash => some (.typedBlob typeName schemaHash)
  | .unknown wireType => some (.unknown wireType)

/-- Transport the schema carried by a delayed direct value.  This is deliberately separate from
the fixed-payload transport: pointwise producers retain their declared result schema even when
all of their leaves are delayed. -/
def reindexWireTypeExpr (map : IndexMap) : WireTypeExpr → Option WireTypeExpr
  | .constantInt => some .constantInt
  | .constantReal => some .constantReal
  | .constantBool => some .constantBool
  | .integer => some .integer
  | .real => some .real
  | .boolean => some .boolean
  | .bytes length => return .bytes (← reindexIntExpr map length)
  | .typedBlob typeName schemaHash => some (.typedBlob typeName schemaHash)
  | .matrix matrixType => return .matrix (← reindexMatrixTypeExpr map matrixType)
  | .trapdoor matrixType sigma gadgetBase digitCount maximum =>
      return .trapdoor (← reindexMatrixTypeExpr map matrixType) (← reindexRealExpr map sigma)
        (← reindexIntExpr map gadgetBase) (← reindexIntExpr map digitCount)
        (← reindexIntExpr map maximum)
  | .preimage matrixType => return .preimage (← reindexMatrixTypeExpr map matrixType)
  | .indexedFamily element count =>
      return .indexedFamily (← reindexWireTypeExpr map element) (← reindexIntExpr map count)

def reindexOperationalFixedScalarSchema
    (map : IndexMap) : OperationalFixedScalarSchema → Option OperationalFixedScalarSchema
  | .integer => some .integer
  | .boolean => some .boolean
  | .real => some .real
  | .trapdoor matrixType sigma gadgetBase digitCount maximum =>
      return .trapdoor (← reindexMatrixTypeExpr map matrixType) (← reindexRealExpr map sigma)
        (← reindexIntExpr map gadgetBase) (← reindexIntExpr map digitCount)
        (← reindexIntExpr map maximum)
  | .bytes length => some (.bytes length)
  | .typedBlob typeName schemaHash => some (.typedBlob typeName schemaHash)
  | .unknown wireType => return .unknown (← reindexWireTypeExpr map wireType)

def reindexOperationalIndexedPayloadSchema
    (map : IndexMap) : OperationalIndexedPayloadSchema → Option OperationalIndexedPayloadSchema
  | .matrix matrixType => return .matrix (← reindexMatrixTypeExpr map matrixType)
  | .scalar schema => return .scalar (← reindexOperationalFixedScalarSchema map schema)

def reindexOperationalFactorTransform
    (map : IndexMap) : OperationalFactorTransform → Option OperationalFactorTransform
  | .negate => some .negate
  | .transpose => some .transpose
  | .rowSlice start stop => return .rowSlice (← reindexIntExpr map start) (← reindexIntExpr map stop)
  | .columnSlice start stop =>
      return .columnSlice (← reindexIntExpr map start) (← reindexIntExpr map stop)
  | .rowEmbed axis part => some (.rowEmbed axis part)
  | .columnEmbed axis part => some (.columnEmbed axis part)

private def reindexPrimitiveOperationKind
    (map : IndexMap) : PrimitiveOperationKind → Option PrimitiveOperationKind
  | .add subtract => some (.add subtract)
  | .multiply rule rightWire => some (.multiply rule rightWire)
  | .tensor => some .tensor
  | .concat axis => some (.concat axis)
  | .transform transform =>
      return .transform (← reindexOperationalFactorTransform map transform)
  | .slice rows columns => do
      let rows ← rows.mapM fun (start, stop) => do
        pure (← reindexIntExpr map start, ← reindexIntExpr map stop)
      let columns ← columns.mapM fun (start, stop) => do
        pure (← reindexIntExpr map start, ← reindexIntExpr map stop)
      pure (.slice rows columns)
  | .scale scalar loopDomains =>
      return .scale (← scalar.reindex map) (← reindexIndexedOperationalParameterDomains map loopDomains)
  | .bggGrouping => some .bggGrouping

private def reindexDirectRelationOperationKind
    (map : IndexMap) : DirectRelationOperationKind → Option DirectRelationOperationKind
  | .preimage maximum loopDomains =>
      return .preimage (← maximum.reindex map)
        (← reindexIndexedOperationalParameterDomains map loopDomains)
  | .decomposition declaredType base small digitCount loopDomains layouts =>
      return .decomposition (← reindexIndexedMatrixTypeExpr map declaredType) (← base.reindex map)
        small (← digitCount.reindex map) (← reindexIndexedOperationalParameterDomains map loopDomains)
        layouts

private def reindexDirectValueScalarOperationKind
    (map : IndexMap) : DirectValueScalarOperationKind → Option DirectValueScalarOperationKind
  | .extractCoefficient position => return .extractCoefficient (← reindexIntExpr map position)
  | .thresholdDecodeBool ciphertext plaintext length =>
      return .thresholdDecodeBool (← reindexIntExpr map ciphertext) (← reindexIntExpr map plaintext)
        (← reindexIntExpr map length)
  | .thresholdDecodeInt ciphertext plaintext length =>
      return .thresholdDecodeInt (← reindexIntExpr map ciphertext) (← reindexIntExpr map plaintext)
        (← reindexIntExpr map length)

private def reindexDirectValueMatrixOperationKind
    (map : IndexMap) : DirectValueMatrixOperationKind → Option DirectValueMatrixOperationKind
  | .liftIntegerToConstantPolynomial matrixType =>
      return .liftIntegerToConstantPolynomial (← reindexMatrixTypeExpr map matrixType)
  | .trapdoorPublic matrixType => return .trapdoorPublic (← reindexMatrixTypeExpr map matrixType)

/-- Reindex every expression-bearing field of one delayed pointwise descriptor.  Producer
ownership is an exact executable identity, not an index expression, and is therefore retained
unchanged. -/
def reindexOperationalIndexedPointwiseOperation
    (_environment : ParamEnvironment) (map : IndexMap) :
    OperationalIndexedPointwiseOperation → Option OperationalIndexedPointwiseOperation
  | .matrix operation => do
      let kind ← reindexPrimitiveOperationKind map operation.kind
      let outputType ← reindexIndexedMatrixTypeExpr map operation.outputType
      let outputSchema ← reindexMatrixTypeExpr map operation.outputSchema
      let parameterEnvironment ← reindexParamEnvironment map operation.parameterEnvironment
      pure (.matrix { operation with kind, outputType, outputSchema, parameterEnvironment })
  | .relation operation => do
      let kind ← reindexDirectRelationOperationKind map operation.kind
      let outputType ← reindexIndexedMatrixTypeExpr map operation.outputType
      let outputSchema ← reindexMatrixTypeExpr map operation.outputSchema
      let parameterEnvironment ← reindexParamEnvironment map operation.parameterEnvironment
      pure (.relation { operation with kind, outputType, outputSchema, parameterEnvironment })
  | .scalar operation => some (.scalar operation)
  | .matrixToScalar operation => do
      let kind ← reindexDirectValueScalarOperationKind map operation.kind
      let parameterEnvironment ← reindexParamEnvironment map operation.parameterEnvironment
      pure (.matrixToScalar { operation with kind, parameterEnvironment })
  | .matrixFromScalar operation => do
      let kind ← reindexDirectValueMatrixOperationKind map operation.kind
      let parameterEnvironment ← reindexParamEnvironment map operation.parameterEnvironment
      pure (.matrixFromScalar { operation with kind, parameterEnvironment })

/-- Name the first top-level matrix-fact field that rejects a reindex map.  This retraces work
only after the production transport has already failed, preserving its success-path cost. -/
def operationalMatrixFactReindexFailureField
    (environment : ParamEnvironment) (map : IndexMap) (fact : OperationalMatrixFact) : String :=
  if map.isDirectCarrierContextLift then "unexpected_direct_carrier_context_lift"
  else if (reindexMatrixOriginIdentity environment map fact.origin).isNone then "origin"
  else if (reindexMatrixTypeExpr map fact.matrixType).isNone then "matrix_type"
  else if (reindexOperationalBoundExpr environment map fact.totalHardBound).isNone then "total_bound"
  else if (fact.identity.mapM (reindexPublicMatrixIdentity map)).isNone then "identity"
  else if (fact.relations.mapM (reindexOperationalMatrixRelation environment map)).isNone then "relations"
  else if (reindexOperationalPolynomial environment map fact.polynomial).isNone then "polynomial"
  else if (fact.blockLayout.mapM (reindexOperationalBlockLayout environment map)).isNone then "block_layout"
  else "unknown"

/-- Name the descriptor field that rejects a reindex map.  It is intentionally top-level: nested
field traces can be added later without making failure diagnostics protocol-specific. -/
def operationalPointwiseOperationReindexFailureField
    (map : IndexMap) (operation : OperationalIndexedPointwiseOperation) : String :=
  match operation with
  | .matrix operation =>
      if (reindexPrimitiveOperationKind map operation.kind).isNone then "kind"
      else if (reindexIndexedMatrixTypeExpr map operation.outputType).isNone then "output_type"
      else if (reindexMatrixTypeExpr map operation.outputSchema).isNone then "output_schema"
      else if (reindexParamEnvironment map operation.parameterEnvironment).isNone then "parameter_environment"
      else "unknown"
  | .relation operation =>
      if (reindexDirectRelationOperationKind map operation.kind).isNone then "kind"
      else if (reindexIndexedMatrixTypeExpr map operation.outputType).isNone then "output_type"
      else if (reindexMatrixTypeExpr map operation.outputSchema).isNone then "output_schema"
      else if (reindexParamEnvironment map operation.parameterEnvironment).isNone then "parameter_environment"
      else "unknown"
  | .scalar _ => "scalar_descriptor_unexpected_failure"
  | .matrixToScalar operation =>
      if (reindexDirectValueScalarOperationKind map operation.kind).isNone then "kind"
      else if (reindexParamEnvironment map operation.parameterEnvironment).isNone then "parameter_environment"
      else "unknown"
  | .matrixFromScalar operation =>
      if (reindexDirectValueMatrixOperationKind map operation.kind).isNone then "kind"
      else if (reindexParamEnvironment map operation.parameterEnvironment).isNone then "parameter_environment"
      else "unknown"

/-- Stable descriptor tags keep failure logs filterable by operation family without exposing
producer node numbers or protocol names. -/
def operationalPointwiseOperationDescriptorKind : OperationalIndexedPointwiseOperation → String
  | .matrix _ => "matrix"
  | .relation _ => "relation"
  | .scalar _ => "scalar"
  | .matrixToScalar _ => "matrix_to_scalar"
  | .matrixFromScalar _ => "matrix_from_scalar"

end Mxx.Certificate
