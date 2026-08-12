import Mxx.Certificate.OperationalBounds.DirectCarrier

namespace Mxx.Certificate

open Mxx.Ir

inductive ChoiceStorage where
  | exact (branches : Array OperationalExprId)
  | shared
      (representative : OperationalExprId)
      (schema : ValidatedSchemaId)
  deriving BEq

structure SelectionDomainInterner where
  keys : Array SelectionDomainKey := #[]
  buckets : Std.HashMap UInt64 (Array Nat) := {}
  deriving BEq

structure ValidatedSchemaInterner where
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

def primitiveOperationDiagnosticName : PrimitiveOperationKind → String
  | .add false => "add"
  | .add true => "subtract"
  | .multiply (.matrixMultiplyRelation _) _ => "multiply-relation"
  | .multiply _ _ => "multiply-ordinary"
  | .tensor => "tensor"
  | .concat .rows => "concat-rows"
  | .concat .columns => "concat-columns"
  | .concat .diagonal => "concat-diagonal"
  | .transform _ => "transform"
  | .slice _ _ => "slice"
  | .scale .. => "scale"
  | .bggGrouping => "bgg-grouping"

structure OperationalMatrixExpr where
  matrixType : MatrixTypeExpr
  node : OperationalMatrixExprNode
  containsSelection : Bool := false
  ownerScope : Option ScopeTemplateKey := none
  ownerNode : Option Nat := none
  deriving BEq

structure OperationalExprArena where
  nodes : Array OperationalMatrixExpr := #[]
  /-- Request-owned direct indexed payloads. Wire facts that use `directValue` refer only to this
  carrier; the legacy expression DAG is never consulted for their evaluation. -/
  direct : DirectOperationalIndexedArena := {}
  /-- Indexed metadata is stored by expression ID alongside the arena DAG.  This retains the
  pointwise domain selected for a primitive result without changing wire-level fact storage. -/
  indexedFacts : Array (Option (IndexedFact OperationalExprId)) := #[]
  /-- Nonrecursive scalar atoms and their indexed selection nodes.  These IDs share the request
  lifetime with matrix-expression IDs but are interpreted only through `scalarNodes`. -/
  scalarNodes : Array OperationalScalarExprNode := #[]
  indexedScalars : Array (Option (IndexedFact Nat)) := #[]
  selectionDomains : SelectionDomainInterner := {}
  validatedSchemas : ValidatedSchemaInterner := {}
  activeScope : Option ScopeTemplateKey := none
  activeNode : Option Nat := none
  choiceJoinCount : Nat := 0
  domainComparisonCount : Nat := 0
  exactBranchVisitCount : Nat := 0
  /-- Shared alternatives are represented by one representative/schema pair.  This counter is
  intentionally never incremented; exposing it makes accidental logical-domain traversal visible
  in diagnostics and acceptance fixtures. -/
  sharedLogicalBranchVisitCount : Nat := 0
  cartesianPairVisitCount : Nat := 0
  transformCacheHits : Nat := 0
  transformCacheMisses : Nat := 0
  evaluationState : OperationalExprEvaluationState := {}
  deriving BEq

def liftIndexedOperationalFact
    (arena : OperationalExprArena)
    (left right : IndexedOperationalFact)
    (kernel : OperationalExprArena → OperationalExprId → OperationalExprId →
      Except OperationalError (OperationalExprArena × OperationalExprId)) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  let merged ← match liftPointwise (fun left right => some (left, right)) left right with
    | some value => pure value
    | none => throw (.unsupportedOperationalExpr left.payload)
  let (arena, root) ← kernel arena left.payload right.payload
  pure (arena, { merged with payload := .matrix root })

def liftIndexedOperationalFacts
    (arena : OperationalExprArena)
    (inputs : Array IndexedOperationalFact)
    (kernel : OperationalExprArena → Array OperationalExprId →
      Except OperationalError (OperationalExprArena × OperationalExprId)) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  let first ← match inputs[0]? with
    | some value => pure value
    | none => throw (.unsupportedOperationalExpr arena.nodes.size)
  let initial : IndexedFact (Array OperationalExprId) := {
    context := first.context
    payload := #[first.payload]
    storage := first.storage
  }
  let merged ← inputs.extract 1 inputs.size |>.foldlM (fun accumulated next =>
    match liftPointwise (fun roots root => some (roots.push root)) accumulated next with
    | some value => pure value
    | none => throw (.unsupportedOperationalExpr first.payload)) initial
  let (arena, root) ← kernel arena merged.payload
  pure (arena, { merged with payload := .matrix root })

def mapIndexedOperationalFact
    (arena : OperationalExprArena)
    (input : IndexedOperationalFact)
    (kernel : OperationalExprArena → OperationalExprId →
      Except OperationalError (OperationalExprArena × OperationalExprId)) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  let (arena, root) ← kernel arena input.payload
  pure (arena, { input with payload := .matrix root })

def selectionDomainFingerprint (key : SelectionDomainKey) : UInt64 :=
  let kind := match key.kind with | .loopLane => 1 | .protocolSelection => 2
  mixOperationalFingerprint
    (mixOperationalFingerprint kind (operationalSelectionFingerprint key.identity))
    (UInt64.ofNat key.count)

def DynamicSelectionIdentity.fromDeclaredCount
    (origin : OperationalValueOrigin) (count : IntExpr) : DynamicSelectionIdentity := {
  index := origin
  expression := .variable {
    owner := {
      stage := ⟨s!"operational-selection:{reprStr origin}"⟩
      scope := ⟨[]⟩
      node := ⟨0⟩
    }
    slot := 0
    count
  }
}

def DynamicSelectionIdentity.fromOrigin
    (origin : OperationalValueOrigin) (count : Nat) : DynamicSelectionIdentity :=
  .fromDeclaredCount origin (.constant count)

def OperationalExprArena.internSelectionDomain
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

def validatedSchemaFingerprint (schema : SelectedMatrixSummary) : UInt64 :=
  let seed := mixOperationalFingerprint 127 (if schema.relationFree then 1 else 0)
  let seed := mixOperationalFingerprint seed (if schema.uniformSchema.isSome then 1 else 0)
  let seed := mixOperationalFingerprint seed (if schema.conservativeFact.isSome then 1 else 0)
  let seed := match schema.sharedLastPublicIdentity with
    | some identity => mixOperationalFingerprint seed (operationalPublicMatrixFingerprint identity)
    | none => mixOperationalFingerprint seed 131
  match schema.selectionOrigin with
  | some .loopLane => mixOperationalFingerprint seed 139
  | some .protocolSelection => mixOperationalFingerprint seed 149
  | none => mixOperationalFingerprint seed 151

def OperationalExprArena.internValidatedSchema
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

def OperationalExprArena.validatedSchema
    (arena : OperationalExprArena)
    (id : ValidatedSchemaId) : Except OperationalError SelectedMatrixSummary :=
  match arena.validatedSchemas.schemas[id.ordinal]? with
  | some schema => pure schema
  | none => throw (.unsupportedOperationalExpr id.ordinal)

/-- Facts for one frozen scope and the append-only expression arena shared by its request.  Wire
facts remain indexed by exact `(node, port)` locations; expression IDs are valid only in `arena`. -/
structure OperationalScopeFacts where
  values : Array (Array OperationalFact) := #[]
  arena : OperationalExprArena := {}

def OperationalExprArena.get?
    (arena : OperationalExprArena) (id : OperationalExprId) : Option OperationalMatrixExpr :=
  arena.nodes[id]?

/-- Recover the indexed pointwise domain represented by a matrix-expression DAG.  Each selection
domain carries its materialized selector expression; a missing materialization is rejected rather
than treating the branch as independent of its selector. -/
def OperationalExprArena.indexContextFor
    (arena : OperationalExprArena) (root : OperationalExprId) : Except OperationalError IndexContext := do
  let rec extendVariables : IndexContext → List IndexVariable → Except OperationalError IndexContext
    | context, [] => pure context
    | context, binder :: remaining => do
        let next ←
          if context.binders.contains binder then pure context
          else match extendContext context binder with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr root)
        extendVariables next remaining
  let rec visit : Nat → OperationalExprId → IndexContext → Except OperationalError IndexContext
    | 0, _, _ => throw (.unsupportedOperationalExpr root)
    | fuel + 1, id, context => do
        match arena.get? id with
        | none => throw (.unsupportedOperationalExpr id)
        | some expression => match expression.node with
          | .concrete _ => pure context
          | .primitive _ arguments =>
              arguments.foldlM (fun accumulated child => visit fuel child accumulated) context
          | .select domain branches =>
              let context ← match branches with
                | .exact roots =>
                    roots.foldlM (fun accumulated child => visit fuel child accumulated) context
                | .shared representative _ => visit fuel representative context
              extendVariables context domain.identity.expression.freeVariables
  let context ← visit (arena.nodes.size + 1) root emptyContext
  if validateContext context then pure context else throw (.unsupportedOperationalExpr root)

def OperationalExprArena.indexedExpr
    (arena : OperationalExprArena) (root : OperationalExprId) : Except OperationalError IndexedOperationalFact := do
  match arena.indexedFacts[root]? with
  | some (some fact) =>
      if fact.payload == root && validateContext fact.context then
        pure { fact with payload := .matrix root }
      else throw (.unsupportedOperationalExpr root)
  | some none | none =>
      let context ← arena.indexContextFor root
      let storage ← match arena.get? root with
        | some expression => pure <| if expression.containsSelection then
            .sharedTemplate else .explicitTable
        | none => throw (.unsupportedOperationalExpr root)
      pure { context, payload := .matrix root, storage }

def OperationalExprArena.rememberIndexedExpr
    (arena : OperationalExprArena) (expression : IndexedOperationalFact) :
    Except OperationalError OperationalExprArena := do
  let root ← match expression.payload with
    | .matrix root => pure root
    | .directValue root => throw (.unsupportedOperationalExpr root)
    | .scalar root => throw (.unsupportedOperationalExpr root)
  if !validateContext expression.context then throw (.unsupportedOperationalExpr root)
  let derivedContext ← arena.indexContextFor root
  if derivedContext != expression.context then throw (.unsupportedOperationalExpr root)
  let stored : IndexedFact Nat := { expression with payload := root }
  match arena.indexedFacts[root]? with
  | some none =>
      let indexedFacts := arena.indexedFacts.set! root (some stored)
      pure { arena with indexedFacts }
  | some (some existing) =>
      if existing == stored then pure arena
      else throw (.unsupportedOperationalExpr root)
  | none => throw (.invalidOperationalExprRef root)

def OperationalExprArena.pushScalar
    (arena : OperationalExprArena)
    (node : OperationalScalarExprNode) : OperationalExprArena × Nat :=
  ({ arena with
      scalarNodes := arena.scalarNodes.push node
      indexedScalars := arena.indexedScalars.push none },
    arena.scalarNodes.size)

def OperationalExprArena.pushScalarConcrete
    (arena : OperationalExprArena)
    (fact : OperationalScalarFact) : OperationalExprArena × Nat :=
  arena.pushScalar (.concrete fact)

def OperationalExprArena.scalarContextFor
    (arena : OperationalExprArena) (root : Nat) : Except OperationalError IndexContext := do
  let rec visit : Nat → Nat → IndexContext → Except OperationalError IndexContext
    | 0, _, _ => throw (.unsupportedOperationalExpr root)
    | fuel + 1, id, context => do
        match arena.scalarNodes[id]? with
        | none => throw (.invalidOperationalExprRef id)
        | some (.concrete _) => pure context
        | some (.primitive _ arguments _) =>
            arguments.foldlM (fun accumulated argument => visit fuel argument accumulated) context
        | some (.selectExact domain branches) => do
            let context ← branches.foldlM
              (fun accumulated branch => visit fuel branch accumulated) context
            domain.identity.expression.freeVariables.foldlM (fun accumulated binder =>
              if accumulated.binders.contains binder then pure accumulated
              else match extendContext accumulated binder with
                | some next => pure next
                | none => throw (.unsupportedOperationalExpr root)) context
        | some (.selectShared domain _ _ representative) => do
            let context ← visit fuel representative context
            domain.identity.expression.freeVariables.foldlM (fun accumulated binder =>
              if accumulated.binders.contains binder then pure accumulated
              else match extendContext accumulated binder with
                | some next => pure next
                | none => throw (.unsupportedOperationalExpr root)) context
  let context ← visit (arena.scalarNodes.size + 1) root emptyContext
  if validateContext context then pure context else throw (.unsupportedOperationalExpr root)

def OperationalExprArena.indexedScalar
    (arena : OperationalExprArena) (root : Nat) : Except OperationalError IndexedOperationalFact := do
  match arena.indexedScalars[root]? with
  | some (some fact) =>
      if fact.payload == root && validateContext fact.context then
        pure { fact with payload := .scalar root }
      else throw (.unsupportedOperationalExpr root)
  | some none =>
      let context ← arena.scalarContextFor root
      let storage ← match arena.scalarNodes[root]? with
        | some (.selectShared ..) => pure .sharedTemplate
        | some (.selectExact ..) => pure .explicitTable
        | some (.concrete _) => pure .explicitTable
        | some (.primitive ..) => pure .explicitTable
        | none => throw (.invalidOperationalExprRef root)
      pure { context, payload := .scalar root, storage }
  | none => throw (.invalidOperationalExprRef root)

def OperationalExprArena.rememberIndexedScalar
    (arena : OperationalExprArena) (expression : IndexedOperationalFact) :
    Except OperationalError OperationalExprArena := do
  let root ← match expression.payload with
    | .scalar root => pure root
    | .matrix root => throw (.unsupportedOperationalExpr root)
    | .directValue root => throw (.unsupportedOperationalExpr root)
  if !validateContext expression.context then throw (.unsupportedOperationalExpr root)
  let derived ← arena.scalarContextFor root
  if derived != expression.context then throw (.unsupportedOperationalExpr root)
  let stored : IndexedFact Nat := { expression with payload := root }
  match arena.indexedScalars[root]? with
  | some none =>
      let indexedScalars := arena.indexedScalars.set! root (some stored)
      pure { arena with indexedScalars }
  | some (some existing) =>
      if existing == stored then pure arena
      else throw (.unsupportedOperationalExpr root)
  | none => throw (.invalidOperationalExprRef root)

/-- Read an indexed scalar only when it is mathematically concrete.  In particular, this never
chooses the representative of a shared family or one branch of an unresolved exact selection. -/
def OperationalExprArena.concreteIndexedScalar
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact) : Except OperationalError OperationalScalarFact := do
  if !expression.context.binders.isEmpty then
    throw (.unsupportedOperationalExpr expression.payload)
  let root ← match expression.payload with
    | .scalar root => pure root
    | .matrix root => throw (.unsupportedOperationalExpr root)
    | .directValue root => throw (.unsupportedOperationalExpr root)
  match arena.scalarNodes[root]? with
  | some (.concrete fact) => pure fact
  | some (.primitive _ _ result) => pure result
  | some _ => throw (.unsupportedOperationalExpr expression.payload)
  | none => throw (.invalidOperationalExprRef expression.payload)

def OperationalExprArena.pushScalarSelection
    (arena : OperationalExprArena)
    (selection : DynamicSelectionIdentity)
    (branches : Array Nat) : Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  if branches.isEmpty then throw (.invalidCount 0 0)
  let (arena, domain) := arena.internSelectionDomain selection branches.size
  let (arena, root) := arena.pushScalar (.selectExact domain branches)
  let expression ← arena.indexedScalar root
  let arena ← arena.rememberIndexedScalar expression
  pure (arena, expression)

def OperationalExprArena.invalidateIndexedExpr
    (arena : OperationalExprArena) (root : OperationalExprId) : OperationalExprArena :=
  if root < arena.indexedFacts.size then
    { arena with indexedFacts := arena.indexedFacts.set! root none }
  else arena

def operationalExprDiagnosticShape
    (arena : OperationalExprArena)
    (id : OperationalExprId) : String :=
  match arena.get? id with
  | none => s!"{id}:missing"
  | some expression => match expression.node with
      | .concrete fact =>
          s!"{id}:concrete(relation={matrixFactHasRelation fact})"
      | .primitive operation arguments =>
          s!"{id}:primitive({primitiveOperationDiagnosticName operation.kind},arity={arguments.size},choice={expression.containsSelection})"
      | .select domain (.exact branches) =>
          s!"{id}:exact(domain={domain.ordinal},count={domain.count},stored={branches.size})"
      | .select domain (.shared _ _) =>
          s!"{id}:shared(domain={domain.ordinal},count={domain.count})"

def operationalExprDiagnosticNeighborhood
    (arena : OperationalExprArena)
    (id : OperationalExprId) : String :=
  match arena.get? id with
  | some { node := .primitive _ arguments, .. } =>
      s!"root={operationalExprDiagnosticShape arena id}, arguments={reprStr (arguments.map
        (operationalExprDiagnosticShape arena))}"
  | _ => s!"root={operationalExprDiagnosticShape arena id}"

def operationalFactDiagnosticShape
    (arena : OperationalExprArena) : OperationalFact → String
  | expression@{ payload := .matrix _, .. } =>
      s!"indexed({operationalExprDiagnosticNeighborhood arena expression.payload},context={
        expression.context.binders.size})"
  | expression@{ payload := .scalar _, .. } =>
      s!"indexed-scalar(root={expression.payload.root},context={
        expression.context.binders.size})"
  | expression@{ payload := .directValue _, .. } =>
      s!"direct-indexed(context={expression.context.binders.size})"

def OperationalExprArena.invalidateEvaluationMemo
    (arena : OperationalExprArena)
    (id : OperationalExprId) : OperationalExprArena := {
  arena with evaluationState := {
    arena.evaluationState with
    totalMemo := arena.evaluationState.totalMemo.set! id none
    noiseMemo := arena.evaluationState.noiseMemo.set! id none
    schemaFactMemo := arena.evaluationState.schemaFactMemo.set! id none
    schemaMemo := arena.evaluationState.schemaMemo.set! id none
  }
}

def OperationalExprArena.push
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
  ({ arena with
      nodes := arena.nodes.push {
        expression with containsSelection, ownerScope := arena.activeScope, ownerNode }
      indexedFacts := arena.indexedFacts.push none
      evaluationState := {
        arena.evaluationState with
        totalMemo := arena.evaluationState.totalMemo.push none
        noiseMemo := arena.evaluationState.noiseMemo.push none
        schemaFactMemo := arena.evaluationState.schemaFactMemo.push none
        schemaMemo := arena.evaluationState.schemaMemo.push none
      }
    },
    arena.nodes.size)

def OperationalExprArena.pushConcrete
    (arena : OperationalExprArena)
    (fact : OperationalMatrixFact) : OperationalExprArena × OperationalExprId :=
  arena.push {
    matrixType := fact.matrixType
    node := .concrete fact
    ownerNode := some fact.subject.node }

/-- Store one internal matrix fact in the arena and expose only its canonical empty-context
indexed representation at a wire boundary. -/
def OperationalExprArena.liftConcreteMatrixFact
    (arena : OperationalExprArena)
    (fact : OperationalMatrixFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, root) := arena.pushConcrete fact
  let expression ← arena.indexedExpr root
  pure (← arena.rememberIndexedExpr expression, expression)

/-- Promote a concrete fixed-assignment matrix result into the request-owned direct carrier.
The exposed wire fact contains only a direct value ID, so later migrated pointwise operations
cannot reconstruct or traverse the legacy selection DAG. -/
def OperationalExprArena.promoteConcreteMatrixFact
    (arena : OperationalExprArena)
    (fact : OperationalMatrixFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (fixed, reference) := arena.direct.fixed.pushMatrix fact
  let direct := { arena.direct with fixed }
  let (direct, value) ← match direct.pushShared emptyContext (.matrix fact.matrixType) reference with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr direct.values.size)
  pure ({ arena with direct }, {
    context := emptyContext
    payload := .directValue value
    storage := .sharedTemplate
  })

/-- Promote a selection-free scalar atom into the direct carrier.  This is intentionally limited
to concrete legacy leaves: a delayed scalar expression cannot be paired with a direct family by
silently choosing a representative. -/
def OperationalExprArena.promoteConcreteScalarFact
    (arena : OperationalExprArena)
    (fact : OperationalScalarFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let schema := operationalScalarSchema fact
  let (fixed, reference) := arena.direct.fixed.pushScalar fact
  let direct := { arena.direct with fixed }
  let (direct, value) ← match direct.pushShared emptyContext (.scalar schema) reference with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr direct.values.size)
  pure ({ arena with direct }, {
    context := emptyContext, payload := .directValue value, storage := .sharedTemplate })

/-- Normalize one relation operand to the direct carrier.  Existing direct values retain their
context; only selection-free concrete legacy leaves are promoted. -/
def OperationalExprArena.promoteDirectRelationOperand
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) :=
  match fact with
  | direct@{ payload := .directValue _, .. } => pure (arena, direct)
  | { context := { binders := #[] }, payload := .matrix root, .. } =>
      match arena.get? root with
      | some { node := .concrete value, .. } => arena.promoteConcreteMatrixFact value
      | _ => throw (.unsupportedOperationalExpr root)
  | { context := { binders := #[] }, payload := .scalar root, .. } =>
      match arena.scalarNodes[root]? with
      | some (.concrete value) => arena.promoteConcreteScalarFact value
      | _ => throw (.unsupportedOperationalExpr root)
  | { payload := .matrix root, .. } | { payload := .scalar root, .. } =>
      throw (.unsupportedOperationalExpr root)

def OperationalExprArena.pushPrimitive
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

def OperationalExprArena.concreteFact
    (arena : OperationalExprArena)
    (id : OperationalExprId) : Except OperationalError OperationalMatrixFact := do
  match arena.get? id with
  | some { node := .concrete fact, .. } => pure fact
  | some _ => throw (.unsupportedOperationalExpr id)
  | none => throw (.invalidOperationalExprRef id)

def OperationalExprArena.checkedType
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

def validateSelectedMatrixSummary
    (representative : OperationalExprId)
    (summary : SelectedMatrixSummary) : Except OperationalError OperationalMatrixFact := do
  let fact ← match summary.conservativeFact with
    | some fact => pure fact
    | none => throw (.unsupportedOperationalExpr representative)
  if summary.uniformSchema != some (operationalUniformSchema fact) ||
      summary.relationFree != !matrixFactHasRelation fact ||
      summary.sharedLastPublicIdentity != boundaryLastPublicIdentity? fact ||
      false then
    throw (.unsupportedOperationalExpr representative)
  pure fact

def concreteRepresentativeFitsEnvelope
    (representative envelope : OperationalMatrixFact) : Bool :=
  if representative.matrixType != envelope.matrixType then false
  else match representative.totalHardBound, envelope.totalHardBound with
    | .closedInt (.constant representativeBound), .closedInt (.constant envelopeBound) =>
        representativeBound <= envelopeBound
    | representativeBound, envelopeBound => representativeBound == envelopeBound

def OperationalExprArena.pushCheckedSchemaEnvelope
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
  let conservativeFact ← validateSelectedMatrixSummary representative summary
  if conservativeFact != fact || expression.matrixType != fact.matrixType then
    throw (.unsupportedOperationalExpr representative)
  let (arena, domain) := arena.internSelectionDomain selection count
  let (arena, schema) := arena.internValidatedSchema summary
  pure (arena.push {
    matrixType := expression.matrixType
    node := .select domain (.shared representative schema)
  })

def OperationalExprArena.pushSelect
    {α : Type} [SelectionIdentityLike α]
    (arena : OperationalExprArena)
    (selection : α)
    (branches : ChoiceStorage) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let domainCount? := SelectionIdentityLike.domainCount? selection
  let selection := SelectionIdentityLike.identity selection
  match branches with
  | .exact values =>
      match domainCount? with
      | some count =>
          if values.size != count then throw (.invalidCount 0 (Int.ofNat values.size))
      | none => pure ()
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
      if count = 0 then
        throw (.unsupportedOperationalExpr representative)
      let conservativeFact ← validateSelectedMatrixSummary representative summary
      if expression.matrixType != conservativeFact.matrixType then
        throw (.operationalExprTypeMismatch representative representative)
      match expression.node with
      | .concrete representativeFact =>
          if !concreteRepresentativeFitsEnvelope representativeFact conservativeFact then
            throw (.unsupportedOperationalExpr representative)
      | _ => pure ()
      let (arena, domain) := arena.internSelectionDomain selection count
      let (arena, schema) := arena.internValidatedSchema summary
      pure (arena.push {
        matrixType := expression.matrixType
        node := .select domain (.shared representative schema)
      })

/-- Preserve a matrix family domain even when every lane reuses the same expression ID.  Ordinary
selection may canonicalize equal branches, but family extraction needs the indexed function domain. -/
def OperationalExprArena.pushFamilyExactSelection
    {α : Type} [SelectionIdentityLike α]
    (arena : OperationalExprArena)
    (selection : α)
    (branches : Array OperationalExprId) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let domainCount? := SelectionIdentityLike.domainCount? selection
  let selection := SelectionIdentityLike.identity selection
  match domainCount? with
  | some count =>
      if branches.size != count then throw (.invalidCount 0 (Int.ofNat branches.size))
  | none => pure ()
  let first ← match branches[0]? with
    | some first => pure first
    | none => throw (.invalidCount 0 0)
  let matrixType ← arena.checkedType first (branches.extract 1 branches.size)
  let (arena, domain) := arena.internSelectionDomain selection branches.size
  pure (arena.push { matrixType, node := .select domain (.exact branches) })

def OperationalExprArena.pushSharedSelection
    (arena : OperationalExprArena)
    (selection : DynamicSelectionIdentity)
    (count representative : Nat)
    (summary : SelectedMatrixSummary) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let (arena, domain) := arena.internSelectionDomain selection count
  let (arena, schema) := arena.internValidatedSchema summary
  arena.pushSelect domain (.shared representative schema)

def OperationalExprArena.pushExactSelection
    (arena : OperationalExprArena)
    (selection : DynamicSelectionIdentity)
    (branches : Array OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let mut arena := arena
  let mut ids : Array OperationalExprId := #[]
  for branch in branches do
    match branch.payload with
    | .matrix id => ids := ids.push id
    | .directValue id => throw (.unsupportedOperationalExpr id)
    | .scalar id => throw (.unsupportedOperationalExpr id)
  arena.pushSelect selection (.exact ids)

/-- An operation over an envelope may recover either the complete exact alternatives or an already
checked post-operation envelope carried by its representative expression.  Either form already
has the required selection identity and logical domain and must not be wrapped in a second,
redundant envelope. -/
def OperationalExprArena.isMatchingSelection
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

def OperationalExprArena.containsSelection
    (arena : OperationalExprArena)
    (root : OperationalExprId) : Except OperationalError Bool :=
  match arena.get? root with
  | some expression => pure expression.containsSelection
  | none => throw (.invalidOperationalExprRef root)

def OperationalExprEvaluationState.empty
    (arena : OperationalExprArena) : OperationalExprEvaluationState := {
  totalMemo := Array.replicate arena.nodes.size none
  noiseMemo := Array.replicate arena.nodes.size none
  schemaFactMemo := Array.replicate arena.nodes.size none
  schemaMemo := Array.replicate arena.nodes.size none
}

def OperationalExprEvaluationState.forEnvironment
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (state : OperationalExprEvaluationState) : OperationalExprEvaluationState :=
  if state.environment == some environment then state else {
    state with
    environment := some environment
    totalMemo := Array.replicate arena.nodes.size none
    noiseMemo := Array.replicate arena.nodes.size none
    schemaFactMemo := Array.replicate arena.nodes.size none
    schemaMemo := Array.replicate arena.nodes.size none
    totalStats := {}
    noiseStats := {}
    schemaStats := {}
  }

/-- Transfer classes, rather than broad operation names, are the closed registry keys.  In
particular relation-consuming multiplication cannot inherit the ordinary multiplication row. -/
inductive PrimitiveTransferClass where
  | addSubtract
  | multiplyOrdinary
  | tensor
  | concat
  | transform
  | scale
  | bggGrouping
  deriving BEq, DecidableEq

inductive CompositionalTransfer where
  | supported (transfer : EnvelopeSummaryTransferOperation)
  | requiresConcreteStructure
  deriving BEq

def primitiveTransferClass (operation : PrimitiveOperation) : PrimitiveTransferClass :=
  match operation.kind with
  | .add _ => .addSubtract
  | .multiply _ _ => .multiplyOrdinary
  | .tensor => .tensor
  | .concat _ => .concat
  | .transform _ | .slice _ _ => .transform
  | .scale .. => .scale
  | .bggGrouping => .bggGrouping

/-- Closed registry used by generic choice lifting.  Every transfer-class constructor has exactly
one equation, so adding a class makes this definition and its inventory fixture non-exhaustive. -/
def compositionalTransferRegistry : PrimitiveTransferClass → CompositionalTransfer
  | .addSubtract => .supported .addSubtract
  | .multiplyOrdinary => .requiresConcreteStructure
  | .tensor => .requiresConcreteStructure
  | .concat => .requiresConcreteStructure
  | .transform => .supported .transform
  | .scale => .supported .scale
  | .bggGrouping => .requiresConcreteStructure

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

def classifyIntBinary : IntBinaryOp → Unit
  | .add | .subtract | .multiply | .divide | .remainder => ()

def classifyIntCompare : IntCompareOp → Unit
  | .equal | .less | .lessEqual => ()

def classifyRealBinary : RealBinaryOp → Unit
  | .add | .subtract | .multiply | .divide => ()

def classifyConcatAxis : ConcatAxis → Unit
  | .rows | .columns | .diagonal => ()

def classifyHashVariant : Mxx.HashVariant → Unit
  | .plain | .decomposed | .smallDecomposed => ()

def classifyLoopInputMode : LoopInputMode → Unit
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
  | .liftIntegerToConstantPolynomial _
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

structure OperationalNumericSlot where
  matrixMaximum : Option Int := none
  integerLower : Option Int := none
  integerUpper : Option Int := none
  deriving Inhabited

abbrev OperationalNumericState := Array OperationalNumericSlot

def scalarFactClosedMaximum : OperationalScalarFact → Option Int
  | .trapdoor fact => match fact.maximum with
      | .closedInt (.constant maximum) => some maximum
      | _ => none
  | _ => none

def scalarFactNumericSlot : OperationalScalarFact → OperationalNumericSlot
  | .trapdoor fact => { matrixMaximum := scalarFactClosedMaximum (.trapdoor fact) }
  | .integer fact => { integerLower := some fact.lower, integerUpper := some fact.upper }
  | _ => {}

def scalarFactNumericExpressions
    (slot : Nat) : OperationalScalarFact → List (OperationalBoundPath × OperationalBoundExpr)
  | .trapdoor fact => [(.matrixMaximum 0 slot, fact.maximum)]
  | .integer fact => [
      (.integerLower 0 slot, fact.lowerExpression),
      (.integerUpper 0 slot, fact.upperExpression)
    ]
  | _ => []

def OperationalExprArena.foldScalarFacts
    {α : Type}
    (arena : OperationalExprArena)
    (root : Nat)
    (initial : α)
    (combine : α → OperationalScalarFact → Except OperationalError α) :
    Except OperationalError α := do
  let rec visit : Nat → Nat → α → Except OperationalError α
    | 0, root, _ => throw (.unsupportedOperationalExpr root)
    | fuel + 1, root, accumulated => match arena.scalarNodes[root]? with
        | none => throw (.invalidOperationalExprRef root)
        | some (.concrete fact) => combine accumulated fact
        | some (.primitive _ _ result) => combine accumulated result
        | some (.selectExact _ branches) =>
            branches.foldlM (fun result branch => visit fuel branch result) accumulated
        | some (.selectShared _ _ _ representative) => visit fuel representative accumulated
  visit (arena.scalarNodes.size + 1) root initial

/-- Fold the concrete matrix leaves of an indexed matrix DAG.  A carried matrix is always kept in
the arena: primitive nodes contribute their input leaves, Exact selections contribute every stored
lane, and Shared selections contribute only their checked representative. -/
def OperationalExprArena.foldMatrixConcreteLeaves
    {α : Type}
    (arena : OperationalExprArena)
    (root : OperationalExprId)
    (initial : α)
    (combine : α → OperationalMatrixFact → Except OperationalError α) :
    Except OperationalError α := do
  let rec visit : Nat → OperationalExprId → α → Except OperationalError α
    | 0, _, _ => throw (.unsupportedOperationalExpr root)
    | fuel + 1, id, accumulated => match arena.get? id with
      | none => throw (.invalidOperationalExprRef id)
      | some { node := .concrete fact, .. } => combine accumulated fact
      | some { node := .primitive _ arguments, .. } =>
          arguments.foldlM (fun result argument => visit fuel argument result) accumulated
      | some { node := .select _ (.exact branches), .. } =>
          branches.foldlM (fun result branch => visit fuel branch result) accumulated
      | some { node := .select _ (.shared representative _), .. } =>
          visit fuel representative accumulated
  visit (arena.nodes.size + 1) root initial

def mapIndexedMatrixLeaves
    (_cacheNamespace : String)
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact)
    (map : OperationalMatrixFact → OperationalMatrixFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  let root ← match expression.payload with
    | .matrix root => pure root
    | .directValue root => throw (.unsupportedOperationalExpr root)
    | .scalar root => throw (.unsupportedOperationalExpr root)
  let rec visit : Nat → OperationalExprArena → Std.HashMap OperationalExprId OperationalExprId →
      OperationalExprId → Except OperationalError
        (OperationalExprArena × Std.HashMap OperationalExprId OperationalExprId × OperationalExprId)
    | 0, _, _, id => throw (.unsupportedOperationalExpr id)
    | fuel + 1, arena, memo, id => match memo[id]? with
      | some mapped => pure (arena, memo, mapped)
      | none => do
          let expression ← match arena.get? id with
            | some expression => pure expression
            | none => throw (.invalidOperationalExprRef id)
          let (arena, memo, mapped) ← match expression.node with
            | .concrete fact =>
                let mapped := map fact
                if mapped == fact then pure (arena, memo, id)
                else
                  let (nextArena, mapped) := arena.pushConcrete mapped
                  pure (nextArena, memo, mapped)
            | .primitive operation arguments => do
                let mut arena := arena
                let mut memo := memo
                let mut mappedArguments := #[]
                for argument in arguments do
                  let (nextArena, nextMemo, mapped) ← visit fuel arena memo argument
                  arena := nextArena
                  memo := nextMemo
                  mappedArguments := mappedArguments.push mapped
                if mappedArguments == arguments then pure (arena, memo, id)
                else
                  let (nextArena, mapped) := arena.push {
                    expression with node := .primitive operation mappedArguments }
                  pure (nextArena, memo, mapped)
            | .select selection (.exact branches) => do
                let mut arena := arena
                let mut memo := memo
                let mut mappedBranches := #[]
                for branch in branches do
                  let (nextArena, nextMemo, mapped) ← visit fuel arena memo branch
                  arena := nextArena
                  memo := nextMemo
                  mappedBranches := mappedBranches.push mapped
                if mappedBranches == branches then pure (arena, memo, id)
                else
                  let (nextArena, mapped) ← arena.pushSelect selection.identity (.exact mappedBranches)
                  pure (nextArena, memo, mapped)
            | .select selection (.shared representative summaryId) => do
                let (arena, memo, mappedRepresentative) ← visit fuel arena memo representative
                let summary ← arena.validatedSchema summaryId
                let conservative ← validateSelectedMatrixSummary id summary
                let mappedConservative := map conservative
                let mappedSummary ← match transferSelectedMatrixSummary .instantiationMap
                    #[summary] mappedConservative with
                  | some mapped => pure mapped
                  | none => throw (.unsupportedOperationalExpr id)
                if mappedRepresentative == representative && mappedSummary == summary then
                  pure (arena, memo, id)
                else
                  let (nextArena, mapped) ← arena.pushSharedSelection selection.identity selection.count
                    mappedRepresentative mappedSummary
                  pure (nextArena, memo, mapped)
          pure (arena, memo.insert id mapped, mapped)
  let (arena, _, mappedRoot) ← visit (arena.nodes.size + 1) arena {} root
  let mapped : IndexedOperationalFact := { expression with payload := .matrix mappedRoot }
  pure (← arena.rememberIndexedExpr mapped, mapped)

def factClosedMaximum
    (arena : OperationalExprArena) : OperationalFact → Except OperationalError (Option Int)
  | { payload := .scalar root, .. } =>
      arena.foldScalarFacts root none fun maximum fact =>
        let next := scalarFactClosedMaximum fact
        pure <| match maximum, next with
          | none, value | value, none => value
          | some left, some right => some (max left right)
  | { payload := .matrix root, .. } =>
      arena.foldMatrixConcreteLeaves root none fun maximum fact =>
        let next := match fact.totalHardBound with
          | .closedInt (.constant value) => some value
          | _ => none
        pure <| match maximum, next with
          | none, value | value, none => value
          | some left, some right => some (max left right)
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)

def factNumericSlot
    (arena : OperationalExprArena) (fact : OperationalFact) :
    Except OperationalError OperationalNumericSlot := do
  match fact with
  | { payload := .matrix _, .. } =>
      pure { matrixMaximum := ← factClosedMaximum arena fact }
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)
  | { payload := .scalar root, .. } =>
      arena.foldScalarFacts root {} fun accumulated scalar =>
        let next := scalarFactNumericSlot scalar
        pure {
          matrixMaximum := match accumulated.matrixMaximum, next.matrixMaximum with
            | none, value | value, none => value
            | some left, some right => some (max left right)
          integerLower := match accumulated.integerLower, next.integerLower with
            | none, value | value, none => value
            | some left, some right => some (min left right)
          integerUpper := match accumulated.integerUpper, next.integerUpper with
            | none, value | value, none => value
            | some left, some right => some (max left right)
        }

def lookupPrevious
    (states : List OperationalNumericState) : OperationalBoundPath → Option Int
  | .matrixMaximum depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.matrixMaximum)
  | .integerLower depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.integerLower)
  | .integerUpper depth slot => states[depth]? >>= fun state => state[slot]? >>= (·.integerUpper)

def operationalBoundPathSlot : OperationalBoundPath → Nat
  | .matrixMaximum _ slot | .integerLower _ slot | .integerUpper _ slot => slot

def operationalBoundPathAtCurrentDepth : OperationalBoundPath → Bool
  | .matrixMaximum depth _ | .integerLower depth _ | .integerUpper depth _ => depth = 0

def numericStateFromComponents
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

def factMaximumExpr
    (arena : OperationalExprArena) : OperationalFact → Except OperationalError (Option OperationalBoundExpr)
  | { payload := .matrix root, .. } =>
      arena.foldMatrixConcreteLeaves root none fun maximum fact =>
        pure <| match maximum with
          | none => some fact.totalHardBound
          | some value => some (.maximum value fact.totalHardBound)
  | _ => pure none

def factNumericExpressions
    (arena : OperationalExprArena)
    (slot : Nat) : OperationalFact →
    Except OperationalError (List (OperationalBoundPath × OperationalBoundExpr))
  | fact@{ payload := .matrix _, .. } => do
      match ← factMaximumExpr arena fact with
      | some maximum => pure [(.matrixMaximum 0 slot, maximum)]
      | none => pure []
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)
  | { payload := .scalar root, .. } =>
      arena.foldScalarFacts root [] fun accumulated fact =>
        pure <| scalarFactNumericExpressions slot fact |>.foldl (fun result component =>
          if result.any (fun existing => existing.1 == component.1) then
            result.map fun existing => if existing.1 != component.1 then existing else
              (existing.1, match existing.1 with
                | .integerLower .. => .minimum existing.2 component.2
                | .matrixMaximum .. | .integerUpper .. => .maximum existing.2 component.2)
          else result ++ [component]) accumulated

def mapIndexedScalarLeaves
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact)
    (map : OperationalScalarFact → OperationalScalarFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  let root ← match expression.payload with
    | .scalar root => pure root
    | .matrix root => throw (.unsupportedOperationalExpr root)
    | .directValue root => throw (.unsupportedOperationalExpr root)
  let rec visit : Nat → OperationalExprArena → Nat →
      Except OperationalError (OperationalExprArena × Nat)
    | 0, _, root => throw (.unsupportedOperationalExpr root)
    | fuel + 1, arena, root => match arena.scalarNodes[root]? with
        | none => throw (.invalidOperationalExprRef root)
        | some (.concrete fact) => pure (arena.pushScalarConcrete (map fact))
        | some (.primitive kind arguments result) => do
            let (arena, arguments) ← arguments.foldlM (fun (arena, mapped) argument => do
              let (arena, argument) ← visit fuel arena argument
              pure (arena, mapped.push argument)) (arena, #[])
            pure (arena.pushScalar (.primitive kind arguments (map result)))
        | some (.selectExact domain branches) => do
            let (arena, branches) ← branches.foldlM (fun (arena, mapped) branch => do
              let (arena, branch) ← visit fuel arena branch
              pure (arena, mapped.push branch)) (arena, #[])
            pure (arena.pushScalar (.selectExact domain branches))
        | some (.selectShared domain binder subject representative) => do
            let (arena, representative) ← visit fuel arena representative
            pure (arena.pushScalar (.selectShared domain binder subject representative))
  let (arena, root) ← visit (arena.scalarNodes.size + 1) arena root
  let mapped : IndexedOperationalFact := { expression with payload := .scalar root }
  let arena ← arena.rememberIndexedScalar mapped
  pure (arena, mapped)

def pushIndexedScalarFact
    (arena : OperationalExprArena)
    (fact : OperationalScalarFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, root) := arena.pushScalarConcrete fact
  let expression ← arena.indexedScalar root
  pure (← arena.rememberIndexedScalar expression, expression)

def replaceOperationalFactorHardBound
    (bound : OperationalBoundExpr)
    (factor : OperationalFactorKey) : OperationalFactorKey :=
  let update (summary : OperationalBoundedFactorSummary) := { summary with hardBound := bound }
  { factor with
    leaf := match factor.leaf with
      | .boundedSummary origin summary => .boundedSummary origin (update summary)
      | leaf => leaf
    boundedSummary := factor.boundedSummary.map update
  }

def abstractCarriedScalarMaximum
    (slot : Nat) : OperationalScalarFact → OperationalScalarFact
  | .trapdoor fact => .trapdoor { fact with maximum := .previous (.matrixMaximum 0 slot) }
  | .integer fact => .integer {
      fact with
      lowerExpression := .previous (.integerLower 0 slot)
      upperExpression := .previous (.integerUpper 0 slot)
    }
  | fact => fact

def abstractCarriedMaximum
    (slot : Nat)
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | expression@{ payload := .matrix _, .. } =>
      do
      let maximum := OperationalBoundExpr.previous (.matrixMaximum 0 slot)
      let (arena, mapped) ← mapIndexedMatrixLeaves s!"carried-abstract:{slot}" arena expression
        fun fact => { fact with
          totalHardBound := maximum
          polynomial := fact.polynomial.map fun term => { term with product := {
            term.product with
            factors := term.product.factors.map (replaceOperationalFactorHardBound maximum) }} }
      pure (arena, mapped)
  | expression@{ payload := .scalar _, .. } => do
      let (arena, mapped) ← mapIndexedScalarLeaves arena expression
        (abstractCarriedScalarMaximum slot)
      pure (arena, mapped)
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)

def setFactMaximum
    (maximum : Int)
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | expression@{ payload := .matrix _, .. } => do
      let (arena, mapped) ← mapIndexedMatrixLeaves s!"carried-maximum:{maximum}" arena expression
        fun fact => { fact with totalHardBound := .closedInt (.constant maximum) }
      pure (arena, mapped)
  | expression@{ payload := .scalar _, .. } => do
      let update : OperationalScalarFact → OperationalScalarFact
        | .trapdoor fact => .trapdoor { fact with maximum := .closedInt (.constant maximum) }
        | fact => fact
      let (arena, mapped) ← mapIndexedScalarLeaves arena expression update
      pure (arena, mapped)
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)

def setFactMaximumExpr
    (maximum : OperationalBoundExpr)
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | expression@{ payload := .matrix _, .. } => do
      let (arena, mapped) ← mapIndexedMatrixLeaves "carried-maximum-expression" arena expression
        fun fact => { fact with totalHardBound := maximum }
      pure (arena, mapped)
  | expression@{ payload := .scalar _, .. } => do
      let update : OperationalScalarFact → OperationalScalarFact
        | .trapdoor fact => .trapdoor { fact with maximum }
        | fact => fact
      let (arena, mapped) ← mapIndexedScalarLeaves arena expression update
      pure (arena, mapped)
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)

structure CarriedFactorSchema where
  transforms : List OperationalFactorTransform
  inputType : MatrixTypeExpr
  outputType : MatrixTypeExpr
  role : OperationalFactorRole
  boundedType : Option MatrixTypeExpr
  boundedMetadata : Option OperationalMatrixMetadata
  protections : List OperationalCompressionProtection
  deriving BEq

structure CarriedSignalTermSchema where
  coefficient : Int
  factors : List CarriedFactorSchema
  modes : List OperationalProductMode
  outputType : MatrixTypeExpr
  deriving BEq

/-- The loop invariant records only normalized signal structure.  Bounded-only terms contribute
numeric recurrence slots, so their canonical encodings `[]` and `[0]` are deliberately identical.
For a signal term, however, the ordered factor roles, shapes, transforms, and matrix metadata stay
visible; changing any of them is a schema change even when the number of Large factors agrees. -/
def carriedSignalSignature
    (polynomial : OperationalPolynomial) : List CarriedSignalTermSchema :=
  polynomial.filterMap fun term =>
    if term.product.factors.any fun factor => factor.role == .large then
      some {
        coefficient := term.coefficient
        factors := term.product.factors.map fun factor => {
          transforms := factor.transforms
          inputType := factor.inputType
          outputType := factor.outputType
          role := factor.role
          boundedType := factor.boundedSummary.map (·.matrixType)
          boundedMetadata := factor.boundedSummary.map (·.metadata)
          protections := factor.protections
        }
        modes := term.product.modes
        outputType := term.product.outputType
      }
    else none

def scalarSchemaTag : OperationalScalarFact → Nat
  | .integer _ => 0
  | .boolean => 1
  | .real => 2
  | .trapdoor _ => 3
  | .bytes _ => 4
  | .typedBlob _ => 5
  | .unknown _ => 6

def sameCarriedMatrixFactSchema
    (left right : OperationalMatrixFact) : Bool :=
  left.matrixType == right.matrixType &&
  left.matrixParams.modulus == right.matrixParams.modulus &&
  left.matrixParams.ringDimension == right.matrixParams.ringDimension &&
  left.matrixParams.rows == right.matrixParams.rows &&
  left.matrixParams.columns == right.matrixParams.columns &&
  left.metadata == right.metadata && left.canonicalRange == right.canonicalRange &&
  left.identity.isNone && right.identity.isNone && left.relations.isEmpty &&
  right.relations.isEmpty &&
  carriedSignalSignature left.polynomial == carriedSignalSignature right.polynomial

def sameCarriedSchema
    (arena : OperationalExprArena) : OperationalFact → OperationalFact → Bool
  | left@{ payload := .matrix leftRoot, .. }, right@{ payload := .matrix rightRoot, .. } =>
      left.context == right.context && left.storage == right.storage &&
      match arena.foldMatrixConcreteLeaves leftRoot ([] : List OperationalMatrixFact)
          (fun facts fact => pure (facts ++ [fact])),
        arena.foldMatrixConcreteLeaves rightRoot ([] : List OperationalMatrixFact)
          (fun facts fact => pure (facts ++ [fact])) with
      | .ok leftFacts, .ok rightFacts => leftFacts.length == rightFacts.length &&
          (leftFacts.zip rightFacts).all fun (left, right) =>
            sameCarriedMatrixFactSchema left right
      | _, _ => false
  | { payload := .scalar left, .. }, { payload := .scalar right, .. } =>
      match arena.scalarNodes[left]?, arena.scalarNodes[right]? with
      | some (.concrete left), some (.concrete right)
      | some (.primitive _ _ left), some (.primitive _ _ right)
      | some (.concrete left), some (.primitive _ _ right)
      | some (.primitive _ _ left), some (.concrete right) =>
          scalarSchemaTag left == scalarSchemaTag right
      | _, _ => left == right
  | _, _ => false

def carriedLargeFactorCounts
    (arena : OperationalExprArena) : OperationalFact → List Nat
  | { payload := .matrix root, .. } =>
      match arena.foldMatrixConcreteLeaves root ([] : List Nat) fun counts fact =>
          pure (counts ++ fact.polynomial.map operationalLargeFactorCount) with
      | .ok counts => counts
      | .error _ => []
  | { payload := .scalar _, .. } => []
  | { payload := .directValue _, .. } => []

def intExprIsClosed : IntExpr → Bool
  | .constant _ => true
  | .parameter _ => true
  | .loopIndex _ => false
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprIsClosed left && intExprIsClosed right
  | .log2Ceil value => intExprIsClosed value

def intExprUsesLoop (slot : Nat) : IntExpr → Bool
  | .constant _ | .parameter _ => false
  | .loopIndex candidate => candidate == slot
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprUsesLoop slot left || intExprUsesLoop slot right
  | .log2Ceil value => intExprUsesLoop slot value

def intExprUsesParameter (name : String) : IntExpr → Bool
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
def evaluateIntOverLoops
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

def evaluateIntMinimum
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail => pure (tail.foldl min first)

def evaluateIntMaximum
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail => pure (tail.foldl max first)

def evaluateIntMaximumAbsolute
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  let values ← evaluateIntOverLoops environment domains expression
  pure (values.foldl (fun maximum value => max maximum (absolute value)) 0)

/-- Check every contextual assignment, rather than only the cutoff maximum. -/
def validateContextualCutoffNonnegative
    (node : Nat)
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (cutoff : IntExpr) : Except OperationalError OperationalBoundExpr := do
  let minimum ← evaluateIntMinimum environment domains cutoff
  if minimum < 0 then throw (.invalidBound node minimum)
  pure (.contextual .maximum environment domains cutoff)

/-- A gadget has no preimage-sampler cutoff contract; indexed and loop identities preserve this
distinction from sampled trapdoors. -/
def publicIdentityIsGadget : PublicMatrixIdentity → Bool
  | .gadget .. => true
  | .sampledTrapdoor .. => false
  | .indexed _ _ source | .loopInstance _ _ source => publicIdentityIsGadget source

def sameContextualDomainKey : OperationalParameterDomain → OperationalParameterDomain → Bool
  | .loopIndex left _, .loopIndex right _ => left == right
  | .parameter left _ _ _, .parameter right _ _ _ => left == right
  | _, _ => false

/-- Merge cutoff assignment domains without allowing an equal key to overwrite a different
definition.  The resulting enumeration visits each logical assignment exactly once. -/
def mergeContextualCutoffDomains
    (node : Nat)
    (left right : List OperationalParameterDomain) : Except OperationalError
    (List OperationalParameterDomain) := do
  let rec insert : List OperationalParameterDomain → OperationalParameterDomain →
      Except OperationalError (List OperationalParameterDomain)
    | [], candidate => pure [candidate]
    | head :: tail, candidate =>
        if sameContextualDomainKey head candidate then
          if head == candidate then pure (head :: tail) else throw (.preimageCutoffMismatch node)
        else return head :: (← insert tail candidate)
  right.foldlM insert left

/-- Preimage and trapdoor cutoffs must agree per merged assignment, not merely at extrema. -/
def validatePreimageCutoffAgreement
    (node : Nat)
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain)
    (preimageCutoff : IntExpr)
    (publicIdentity : PublicMatrixIdentity)
    (trapdoorCutoff : Option OperationalBoundExpr) : Except OperationalError Unit := do
  match trapdoorCutoff with
  | none =>
      if publicIdentityIsGadget publicIdentity then pure () else throw (.missingPreimageCutoff node)
  | some (.contextual _ trapdoorEnvironment trapdoorDomains trapdoorExpression) =>
      let mergedDomains ← mergeContextualCutoffDomains node trapdoorDomains domains
      let trapdoorValues ← evaluateIntOverLoops trapdoorEnvironment mergedDomains trapdoorExpression
      let preimageValues ← evaluateIntOverLoops environment mergedDomains preimageCutoff
      if trapdoorValues != preimageValues then throw (.preimageCutoffMismatch node)
  | some _ => throw (.preimageCutoffMismatch node)

def evaluateIntInvariant
    (environment : ParamEnvironment) (domains : List OperationalParameterDomain)
    (expression : IntExpr) : Except OperationalError Int := do
  match ← evaluateIntOverLoops environment domains expression with
  | [] => throw .nonClosedExpression
  | first :: tail =>
      if tail.all (· == first) then pure first else throw .nonClosedExpression

def extendParameterDomains
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

def instantiateBoundLoopIndex (slot index : Nat) : OperationalBoundExpr → OperationalBoundExpr
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

def shiftPreviousDepthFrom (cutoff : Nat) : OperationalBoundExpr → OperationalBoundExpr
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

def shiftPreviousDepth := shiftPreviousDepthFrom 0

def OperationalBoundExpr.usesPrevious : OperationalBoundExpr → Bool
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

def OperationalBoundExpr.evaluateInArena
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (previousState : OperationalState)
    (expression : OperationalBoundExpr) : Except OperationalError Int := do
  let numericState ← previousState.mapM (factNumericSlot arena)
  expression.evaluateWithStates environment [numericState]

def OperationalBoundExpr.evaluate
    (environment : ParamEnvironment)
    (previousState : OperationalState)
    (expression : OperationalBoundExpr) : Except OperationalError Int :=
  expression.evaluateInArena environment {} previousState

def setFactRecurrenceState
    (count : Nat)
    (paths : List OperationalBoundPath)
    (initial transition : List OperationalBoundExpr)
    (slot : Nat)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | expression@{ payload := .matrix _, .. } =>
      do
      let maximum := OperationalBoundExpr.recurrenceState
        count paths initial transition (.matrixMaximum 0 slot)
      let (arena, mapped) ← mapIndexedMatrixLeaves s!"carried-recurrence:{slot}" arena expression
        fun fact => { fact with totalHardBound := maximum }
      pure (arena, mapped)
  | expression@{ payload := .scalar _, .. } => do
      let update : OperationalScalarFact → OperationalScalarFact
        | .trapdoor fact =>
            let maximum := OperationalBoundExpr.recurrenceState count paths initial transition
              (.matrixMaximum 0 slot)
            .trapdoor { fact with maximum }
        | .integer fact =>
            let lowerExpression := OperationalBoundExpr.recurrenceState count paths
              initial transition (.integerLower 0 slot)
            let upperExpression := OperationalBoundExpr.recurrenceState count paths
              initial transition (.integerUpper 0 slot)
            .integer { fact with lowerExpression, upperExpression }
        | fact => fact
      let (arena, mapped) ← mapIndexedScalarLeaves arena expression update
      let scalar ← arena.foldScalarFacts mapped.payload.root none fun _ fact => pure (some fact)
      match scalar with
      | some (.integer fact) =>
          let lower ← fact.lowerExpression.evaluateWithStates environment []
          let upper ← fact.upperExpression.evaluateWithStates environment []
          if lower > upper then throw (.invalidBound slot lower)
          let (arena, mapped) ← mapIndexedScalarLeaves arena mapped fun
            | .integer integer => .integer { integer with lower, upper }
            | fact => fact
          pure (arena, mapped)
      | _ => pure (arena, mapped)
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)

def evaluateTransition
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (previousState : OperationalState)
    (transition : Array OperationalBoundExpr) :
    Except OperationalError (OperationalExprArena × OperationalState) := do
  if transition.size != previousState.size then
    throw (.unsupportedOutputArity transition.size previousState.size)
  let values ← transition.toList.mapM
    (OperationalBoundExpr.evaluateInArena environment arena previousState)
  let (arena, next) ← values.zip previousState.toList |>.foldlM
    (fun (arena, next) (value, previous) => do
      let (arena, fact) ← setFactMaximum value arena previous
      pure (arena, next ++ [fact])) (arena, [])
  pure (arena, next.toArray)

def repeatTransition
    (count : Nat)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (transition : Array OperationalBoundExpr)
    (state : OperationalState) : Except OperationalError (OperationalExprArena × OperationalState) :=
  match count with
  | 0 => pure (arena, state)
  | count + 1 => do
      let (arena, next) ← evaluateTransition environment arena state transition
      repeatTransition count environment arena transition next

def fallbackMatrixFact
    (node port : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment) : Except OperationalError OperationalMatrixFact := do
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
  pure (fact.initializePrimitivePolynomial .large)

def defaultFact
    (node : Nat)
    (port : Nat)
    (wireType : WireTypeExpr)
    (environment : ParamEnvironment) : Except OperationalError OperationalMatrixFact :=
  match wireType with
  | .matrix matrixType => fallbackMatrixFact node port matrixType environment
  | .preimage matrixType => fallbackMatrixFact node port matrixType environment
  | _ => throw (.outputTypeMismatch node)

def defaultScalarFact
    (node port : Nat)
    (wireType : WireTypeExpr)
    (environment : ParamEnvironment)
    (domains : List OperationalParameterDomain := []) : Except OperationalError OperationalScalarFact :=
  match wireType with
  | .trapdoor matrixType _ _ _ cutoff => do
      let cap ← match matrixCap matrixType environment with
        | some cap => pure cap
        | none => throw (.invalidMatrixParameters node)
      let params ← match matrixType.evaluate environment (.constant cap) with
        | some params => pure params
        | none => throw (.invalidMatrixParameters node)
      pure (.trapdoor {
        subject := { node, port }
        matrixType
        matrixParams := params
        maximum := .closedInt (.constant cap)
        preimageCutoff := some (← validateContextualCutoffNonnegative node environment domains cutoff)
        publicIdentity := .sampledTrapdoor temporaryScope { node, port := 0 }
      })
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
  | _ => throw (.outputTypeMismatch node)

def lookupFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalFact :=
  match facts.values[wire.node]?.bind fun outputs => outputs[wire.port]? with
  | some fact => pure fact
  | none => throw (.missingOperand node wire)

def requireBooleanFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError Unit := do
  match ← lookupFact node facts wire with
  | expression@{ payload := .scalar _, .. } =>
      match ← facts.arena.concreteIndexedScalar expression with
      | .boolean => pure ()
      | _ => throw (.operandNotBoolean node wire)
  | _ => throw (.operandNotBoolean node wire)

def requireRealFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError Unit := do
  match ← lookupFact node facts wire with
  | expression@{ payload := .scalar _, .. } =>
      match ← facts.arena.concreteIndexedScalar expression with
      | .real => pure ()
      | _ => throw (.operandNotReal node wire)
  | _ => throw (.operandNotReal node wire)

def trapdoorFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalTrapdoorFact := do
  match ← lookupFact node facts wire with
  | expression@{ payload := .scalar _, .. } =>
      match ← facts.arena.concreteIndexedScalar expression with
      | .trapdoor fact => pure fact
      | _ => throw (.missingPublicIdentity node wire)
  | _ => throw (.missingPublicIdentity node wire)

def integerFact
    (node port : Nat)
    (lower upper : Int) : Except OperationalError OperationalScalarFact := do
  if lower > upper then throw (.invalidBound node lower)
  pure (.integer {
    subject := { node, port }
    origin := .local temporaryScope { node, port }
    lower
    upper
    lowerExpression := .closedInt (.constant lower)
    upperExpression := .closedInt (.constant upper)
  })

def integerFactWithExpressions
    (node port : Nat)
    (lower upper : Int)
    (lowerExpression upperExpression : OperationalBoundExpr) :
    Except OperationalError OperationalScalarFact := do
  if lower > upper then throw (.invalidBound node lower)
  pure (.integer {
    subject := { node, port }
    origin := .local temporaryScope { node, port }
    lower
    upper
    lowerExpression
    upperExpression
  })

structure OperationalIntegerInterval where
  lower : Int
  upper : Int
  lowerExpression : OperationalBoundExpr
  upperExpression : OperationalBoundExpr

def integerBinaryInterval
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
    (bound : Int) : Except OperationalError OperationalMatrixFact := do
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
  pure (fact.initializePrimitivePolynomial .bounded)

def cappedMatrixFactExpr
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : OperationalBoundExpr) : Except OperationalError OperationalMatrixFact := do
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
  pure (fact.initializePrimitivePolynomial .bounded)

def classifiedMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : Int)
    (large : Bool)
    (canonicalRange : CanonicalRange := .unknown)
    (metadata : OperationalMatrixMetadata := {}) : Except OperationalError OperationalMatrixFact := do
  let fact ← cappedMatrixFact nodeIndex outputPort matrixType environment bound
  let role := if large then OperationalFactorRole.large else .bounded
  pure (({ fact with
    totalHardBound := .closedInt (.constant bound), canonicalRange, metadata
  }).initializePrimitivePolynomial role)

def classifiedMatrixFactExpr
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (bound : OperationalBoundExpr)
    (large : Bool)
    (canonicalRange : CanonicalRange := .unknown)
    (metadata : OperationalMatrixMetadata := {}) : Except OperationalError OperationalMatrixFact := do
  let fact ← cappedMatrixFactExpr nodeIndex outputPort matrixType environment bound
  let cap ← match matrixCap matrixType environment with
    | some value => pure value | none => throw (.invalidMatrixParameters nodeIndex)
  let totalHardBound := .minimum (.closedInt (.constant cap)) bound
  let role := if large then OperationalFactorRole.large else .bounded
  pure (({ fact with totalHardBound, canonicalRange, metadata
    }).initializePrimitivePolynomial role)

def matrixTargetSummary (fact : OperationalMatrixFact) : RelationTargetSummary := {
  origin := fact.origin
  matrixType := fact.matrixType
  matrixParams := fact.matrixParams
  totalHardBound := fact.totalHardBound
  canonicalRange := fact.canonicalRange
  polynomial := relationSnapshotPolynomial fact.polynomial
}

def operationalProductFromFactors
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

def factorPublicIdentity? (factor : OperationalFactorKey) : Option PublicMatrixIdentity :=
  match factor.leaf with
  | .primitive (.publicMatrix identity) => some identity
  | _ => none

def factorPrimitiveOrigin? (factor : OperationalFactorKey) : Option MatrixOriginIdentity :=
  match factor.leaf with
  | .primitive (.matrix origin) => some origin
  | _ => none

def relationTargetOrigin : OperationalMatrixRelation → MatrixOriginIdentity
  | .decomposition relation => relation.inputOrigin
  | .preimage relation => relation.targetOrigin

def relationMatcherTarget : OperationalMatrixRelation → RelationTargetSummary
  | .decomposition relation => relation.inputSummary
  | .preimage relation => relation.targetSummary

def relationMatcherPublicIdentity : OperationalMatrixRelation → PublicMatrixIdentity
  | .decomposition relation => relation.publicIdentity
  | .preimage relation => relation.publicIdentity

/-- The relation boundary is exact: both adjacent primitive identities, the declared target
origin, and every concrete matrix shape reconstructed from the target snapshot must agree. -/
def exactAdjacentRelationMatches
    (environment : ParamEnvironment)
    (left right : OperationalFactorKey)
    (relation : OperationalMatrixRelation) : Bool :=
  let target := relationMatcherTarget relation
  let targetShape := target.matrixType.evaluate environment (.constant 0)
  let snapshotRebuilds := !target.polynomial.isEmpty && target.polynomial.all fun snapshotTerm =>
    match operationalPolynomialFromSnapshot [snapshotTerm] with
    | [{ product, .. }] => match operationalProductFromFactors product.factors with
      | .ok rebuilt =>
          rebuilt.modes == snapshotTerm.product.modes &&
            rebuilt.outputType == snapshotTerm.product.outputType &&
            match rebuilt.outputType.evaluate environment (.constant 0), targetShape with
            | some output, some expected =>
                output.modulus == expected.modulus && output.ringDimension == expected.ringDimension &&
                  output.rows == expected.rows && output.columns == expected.columns
            | _, _ => false
      | .error _ => false
    | _ => false
  let adjacentShapeMatches := match left.outputType.evaluate environment (.constant 0),
      right.inputType.evaluate environment (.constant 0), targetShape with
    | some leftShape, some rightShape, some expected =>
        leftShape.modulus == rightShape.modulus &&
          leftShape.ringDimension == rightShape.ringDimension &&
          leftShape.columns == rightShape.rows &&
          leftShape.rows == expected.rows && rightShape.columns == expected.columns &&
          expected.modulus == target.matrixParams.modulus &&
            expected.ringDimension == target.matrixParams.ringDimension &&
            expected.rows == target.matrixParams.rows && expected.columns == target.matrixParams.columns
    | _, _, _ => false
  factorPublicIdentity? left == some (relationMatcherPublicIdentity relation) &&
    factorPrimitiveOrigin? right == some (match relation with
      | .decomposition value => value.producer
      | .preimage value => value.producer) &&
    relationTargetOrigin relation == target.origin && adjacentShapeMatches && snapshotRebuilds

def matchingFactorRelation?
    (environment : ParamEnvironment)
    (left right : OperationalFactorKey) : Option OperationalMatrixRelation := do
  if !left.transforms.isEmpty || !right.transforms.isEmpty then none else pure ()
  right.relations.find? fun relation =>
    (match relation with
    | .decomposition value => value.status == ReconstructionStatus.available
    | .preimage _ => true) && exactAdjacentRelationMatches environment left right relation

def rewriteOperationalTermRelation?
    (node : Nat) (environment : ParamEnvironment)
    (term : OperationalTerm) : Except OperationalError (Option OperationalPolynomial) := do
  let rec visit
      (accumulated : List OperationalFactorKey) :
      List OperationalFactorKey → Except OperationalError (Option OperationalPolynomial)
    | left :: right :: tail =>
        match matchingFactorRelation? environment left right with
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

def rewriteOperationalRelationsWithCount
    (node : Nat) (environment : ParamEnvironment)
    (polynomial : OperationalPolynomial) : Except OperationalError (OperationalPolynomial × Nat) := do
  let rec finishTerm : Nat → OperationalTerm →
      Except OperationalError (OperationalPolynomial × Nat)
    | 0, term => do
        match ← rewriteOperationalTermRelation? node environment term with
        | none => pure ([term], 0)
        | some _ => throw (.invalidMatrixParameters node)
    | fuel + 1, term => do
        match ← rewriteOperationalTermRelation? node environment term with
        | none => pure ([term], 0)
        | some rewritten => do
          let mut finished : OperationalPolynomial := []
          let mut count := 1
          for generated in rewritten do
              let (next, nextCount) ← finishTerm fuel generated
              finished := finished ++ next
              count := count + nextCount
            pure (finished, count)
  let mut finished : OperationalPolynomial := []
  let mut count := 0
  for term in polynomial do
    let (next, nextCount) ← finishTerm 64 term
    finished := finished ++ next
    count := count + nextCount
  pure (normalizeOperationalTerms finished, count)

def rewriteOperationalRelations
    (node : Nat) (environment : ParamEnvironment)
    (polynomial : OperationalPolynomial) : Except OperationalError OperationalPolynomial :=
  return (← rewriteOperationalRelationsWithCount node environment polynomial).1

def sameConcreteMatrixShape (left right : Mxx.SamplerParams) : Bool :=
  left.modulus == right.modulus &&
    left.ringDimension == right.ringDimension &&
    left.rows == right.rows &&
    left.columns == right.columns

/-- Decide matrix-product compatibility from evaluated dimensions rather than the syntax of the
dimension expressions. This accepts equivalent forms such as `2` and `1 * 2`, while remaining
fail-closed when any type expression is not closed under the current parameter environment. -/
def concreteMatrixProductMatches
    (leftType rightType outputType : MatrixTypeExpr)
    (environment : ParamEnvironment) : Bool :=
  match leftType.evaluate environment (.constant 0),
      rightType.evaluate environment (.constant 0),
      outputType.evaluate environment (.constant 0) with
  | some left, some right, some output =>
      let sameRing :=
        left.modulus == right.modulus && left.modulus == output.modulus &&
          left.ringDimension == right.ringDimension &&
          left.ringDimension == output.ringDimension
      let outputShapeMatches (rows columns : Nat) :=
        output.rows == rows && output.columns == columns
      sameRing &&
        if left.columns == right.rows then
          outputShapeMatches left.rows right.columns
        else if left.rows == 1 && left.columns == 1 then
          outputShapeMatches right.rows right.columns
        else if right.rows == 1 && right.columns == 1 then
          outputShapeMatches left.rows left.columns
        else if left.rows == 1 && right.rows == 1 && left.columns == right.columns then
          outputShapeMatches 1 left.columns
        else false
  | _, _, _ => false

def equivalentRetypeOperationalFactor
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
def equivalentRetypeOperationalPolynomial
    (outputType : MatrixTypeExpr)
    (input : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  input.mapM fun term => do
    match term.product.factors.reverse with
    | [] => throw .malformedProduct
    | last :: reversePrefix =>
        let replacement := equivalentRetypeOperationalFactor outputType reversePrefix.isEmpty last
        let factors := (replacement :: reversePrefix).reverse
        pure { term with product := { term.product with factors, outputType } }

def retypeMatrixFact
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

def valueOriginAt
    (scope : ScopeTemplateKey)
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalValueOrigin := do
  match ← lookupFact node facts wire with
  | expression@{ payload := .scalar _, .. } =>
      match ← facts.arena.concreteIndexedScalar expression with
      | .integer fact => pure fact.origin
      | .bytes fact => pure fact.origin
      | _ => pure (.local scope wire)
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)
  | { payload := .matrix root, .. } =>
      match facts.arena.concreteFact root with
      | .ok { origin := .value originScope originWire, .. } => pure (.local originScope originWire)
      | .ok { origin := .protocolInput input, .. } => pure (.protocolInput input)
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
  let firstTermSummary ← match summaries.head? with
    | some summary => pure summary
    | none => throw .malformedProduct
  let hardBound := (noise.zip summaries).foldl (fun current pair =>
    .add current (.multiply
      (.closedInt (.constant (operationalAbsoluteCoefficient pair.1.coefficient)))
      pair.2.hardBound)) (.closedInt (.constant 0))
  let tokens := [.sumStart] ++ ((noise.zip summaries).flatMap fun (term, summary) =>
    boundedNoiseTermTokens term summary) ++ [.summaryBound hardBound, .sumEnd]
  pure (some {
    matrixType := firstTerm.product.outputType
    rowCount := firstTermSummary.rowCount
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

/-- A decoder residual is eligible for numeric noise evaluation only once normalization and
relation consumption have removed every signal (`Large`) term. -/
def OperationalMatrixFact.rejectResidualLargeTerms
    (fact : OperationalMatrixFact) : Except OperationalError Unit :=
  if fact.polynomial.any operationalTermIsSignal then
    throw (.residualContainsLargeTerm fact.subject.node)
  else
    pure ()

def flatErrorAt (node : Nat) : OperationalFlatError → OperationalError
  | error => .flat node error

def polynomialMatrixFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (polynomial : OperationalPolynomial)
    (canonicalRange : CanonicalRange := .unknown) : Except OperationalError OperationalMatrixFact := do
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
  pure {
    subject := { node := nodeIndex, port := outputPort }
    origin := .value temporaryScope { node := nodeIndex, port := outputPort }
    matrixType
    matrixParams := params
    totalHardBound
    polynomial
    metadata
    canonicalRange
  }

def erasePrimitiveSelectionFactBounds
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

def samePrimitiveSelectionShape
    (left right : OperationalMatrixFact) : Bool :=
  operationalUniformSchema (erasePrimitiveSelectionFactBounds left) ==
    operationalUniformSchema (erasePrimitiveSelectionFactBounds right)

def maximumPrimitiveSelectionBound
    (first : OperationalBoundExpr)
    (remaining : List OperationalBoundExpr) : OperationalBoundExpr :=
  remaining.foldl OperationalBoundExpr.maximum first

/-- Close every complete mutually-exclusive primitive result before taking the branch maximum.
This is the selection-envelope analogue of the endpoint join below, but lives next to primitive
construction so later operations never need to carry an already relation-free family exactly. -/
def summarizePrimitiveSelectionFacts
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
    throw (.unsupportedOperationalExpr 0)
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
        rowCount := first.matrixParams.rows
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
    pure { output with
      subject := first.subject
      origin := first.origin
      identity := if facts.all (·.identity == first.identity) then first.identity else none }

/-- Build the result of pushing a primitive through an exact selection. Graph IR dimension
expressions are compared by evaluated shape.  Complete relation-free branches are joined by their
full branch maximum; relation-bearing branches use an envelope only when every branch proves the
same first relation boundary. Otherwise their exact identities remain available downstream. -/
def OperationalExprArena.pushPrimitiveSelection
    {α : Type} [SelectionIdentityLike α]
    (arena : OperationalExprArena)
    (selection : α)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
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
    let mut state := OperationalExprEvaluationState.forEnvironment
      arena environment arena.evaluationState
    let mut branchFacts : Array OperationalMatrixFact := #[]
    let mut derivationFailed := false
    for branch in branches do
      if !derivationFailed then
        match deriveSchema arena environment branch state with
        | .ok (fact, nextState) =>
            branchFacts := branchFacts.push fact
            state := nextState
        | .error _ => derivationFailed := true
    let arena := { arena with evaluationState := state }
    if derivationFailed then
      arena.pushSelect selection (.exact branches)
    else
      let allBranchSummary := selectedMatrixSummary branchFacts
      if !allBranchSummary.relationFree then
        -- Compact selections never carry relations. Relation producers are represented only in
        -- the direct indexed carrier, which retains their exact producer and selector identity.
        arena.pushSelect selection (.exact branches)
      else match summarizePrimitiveSelectionFacts environment branchFacts with
      | .ok conservativeFact =>
        -- The representative keeps the first branch's structure and identities.  Only the
        -- validated schema owns the conservative all-branch join; replacing the representative
        -- with that join would discard the relation structure later consumers must correlate.
        let summary := {
          selectedMatrixSummary #[conservativeFact] with
          relationFree := allBranchSummary.relationFree
          sharedLastPublicIdentity := allBranchSummary.sharedLastPublicIdentity
        }
        let arena := { arena with choiceJoinCount := arena.choiceJoinCount + 1 }
        arena.pushCheckedSchemaEnvelope selection branches.size first summary conservativeFact
      | .error _ =>
        arena.pushSelect selection (.exact branches)

/-- Preserve a strict canonical coefficient range through ordinary matrix multiplication when
both inputs are known constant-polynomial matrices.  In that case no negacyclic convolution term
can wrap a negative coefficient to a residue near the modulus: every output coefficient is a sum
of `left.columns` nonnegative scalar products.  For general polynomial inputs the quotient-ring
signs make any sub-modulus range unsafe, so the result remains unknown. -/
def constantPolynomialProductCanonicalRange
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

/-- Contract only the matching block boundaries of a column-concatenated left operand and a
row-concatenated right operand.  The ordered concat snapshots, rather than the visible embedded
terms, are authoritative: an all-zero block therefore remains a required physical lane. -/
def contractComplementaryBlocks
    (node : Nat)
    (expectedOutput : MatrixTypeExpr)
    (left right : OperationalMatrixFact)
    (raw : OperationalPolynomial) : Except OperationalError OperationalPolynomial := do
  let layoutMatchesOwner (axis : ConcatAxis) (owner : MatrixTypeExpr)
      (partitions : Array OperationalBlockPartition) : Bool :=
    if partitions.isEmpty then false else
    match partitions[0]? with
    | none => false
    | some first =>
        let sameRing := partitions.all fun partition => operationalSameRing partition.matrixType owner
        match axis with
        | .columns => sameRing &&
            partitions.all (fun partition => operationalDimensionEqual partition.matrixType.rows first.matrixType.rows) &&
            operationalDimensionEqual first.matrixType.rows owner.rows &&
            operationalDimensionEqual
              (partitions.foldl (fun total partition => .add total partition.matrixType.columns)
                (.constant 0)) owner.columns
        | .rows => sameRing &&
            partitions.all (fun partition => operationalDimensionEqual partition.matrixType.columns
              first.matrixType.columns) &&
            operationalDimensionEqual first.matrixType.columns owner.columns &&
            operationalDimensionEqual
              (partitions.foldl (fun total partition => .add total partition.matrixType.rows)
                (.constant 0)) owner.rows
        | .diagonal => false
  match left.blockLayout, right.blockLayout with
  | some leftLayout, some rightLayout =>
      if leftLayout.axis != .columns || rightLayout.axis != .rows then pure raw
      else if !layoutMatchesOwner .columns left.matrixType leftLayout.partitions ||
          !layoutMatchesOwner .rows right.matrixType rightLayout.partitions ||
          !operationalSameRing left.matrixType right.matrixType ||
          !operationalSameRing left.matrixType expectedOutput ||
          !operationalDimensionEqual left.matrixType.rows expectedOutput.rows ||
          !operationalDimensionEqual right.matrixType.columns expectedOutput.columns then
        throw (.malformedRelation node)
      else if leftLayout.partitions.size != rightLayout.partitions.size then
        throw (.malformedRelation node)
      else do
        let mut contracted : OperationalPolynomial := []
        for index in [:leftLayout.partitions.size] do
          let leftPart ← match leftLayout.partitions[index]? with
            | some partition => pure partition
            | none => throw (.malformedRelation node)
          let rightPart ← match rightLayout.partitions[index]? with
            | some partition => pure partition
            | none => throw (.malformedRelation node)
          if !operationalSameRing leftPart.matrixType rightPart.matrixType ||
              !operationalDimensionEqual leftPart.matrixType.columns rightPart.matrixType.rows then
            throw (.malformedRelation node)
          let product ← multiplyOperationalPolynomials leftPart.polynomial rightPart.polynomial
            |>.mapError (flatErrorAt node)
          contracted := contracted ++ product
        normalizeOperationalPolynomial contracted |>.mapError (flatErrorAt node)
  | _, _ => pure raw

def multiplyConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial
    |>.mapError (flatErrorAt nodeIndex)
  let contracted ← contractComplementaryBlocks nodeIndex matrixType left right raw
  let rewritten ← rewriteOperationalRelations nodeIndex environment contracted
  let polynomial ← match rule with
    | .matrixMultiplyRelation declaredRight => do
        if declaredRight != rightWire then throw (.missingRelation nodeIndex declaredRight)
        if rewritten == contracted then throw (.missingRelation nodeIndex rightWire)
        pure rewritten
    | _ => pure rewritten
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
    (constantPolynomialProductCanonicalRange left right)

/-- Concrete direct reduction records exact relation applications without changing the produced
fact.  Ordinary consumers deliberately project the fact alone through `multiplyConcreteMatrixFacts`. -/
def multiplyConcreteMatrixFactsWithRelationRewriteCount
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError (OperationalMatrixFact × Nat) := do
  let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial
    |>.mapError (flatErrorAt nodeIndex)
  let contracted ← contractComplementaryBlocks nodeIndex matrixType left right raw
  let (rewritten, rewriteCount) ← rewriteOperationalRelationsWithCount nodeIndex environment contracted
  let polynomial ← match rule with
    | .matrixMultiplyRelation declaredRight => do
        if declaredRight != rightWire then throw (.missingRelation nodeIndex declaredRight)
        if rewritten == contracted then throw (.missingRelation nodeIndex rightWire)
        pure rewritten
    | _ => pure rewritten
  let fact ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
    (constantPolynomialProductCanonicalRange left right)
  pure (fact, rewriteCount)

def addConcreteMatrixFacts
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
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial

/-- A direct value is evaluable only when every binder in its declared context has an assigned,
in-range lane.  Checking the whole context at each recursive root prevents an unrelated leaf from
silently accepting a partial assignment; mapped roots construct and validate their own source
assignment before their recursive evaluation. -/
def completeDirectIndexAssignment
    (parameters : ParamEnvironment)
    (context : IndexContext)
    (indices : IndexValueEnvironment) : Bool :=
  validateContext context && context.binders.all fun binder =>
    (evaluateIndexExpr parameters context indices (.variable binder)).isSome

/-- Build one delayed direct matrix operation.  Mixing a migrated direct value with a legacy
expression is rejected, rather than silently routing through the former selection engine. -/
def OperationalExprArena.pushDirectMatrixPointwiseN
    (arena : OperationalExprArena)
    (operation : PrimitiveOperation)
    (inputs : Array OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let inputIds ← inputs.mapM fun input => match input.payload with
    | .directValue id => pure id
    | .matrix id | .scalar id => throw (.unsupportedOperationalExpr id)
  let inputSchemas ← inputIds.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef id)
  if !matrixOperationSchemasValid operation inputSchemas operation.outputType then
    throw (.outputTypeMismatch operation.ownerNode)
  let (direct, result) ← match arena.direct.pushPointwise (.matrix operation) inputIds with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })

def OperationalExprArena.pushDirectMatrixPointwise
    (arena : OperationalExprArena)
    (operation : PrimitiveOperation)
    (left right : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) :=
  arena.pushDirectMatrixPointwiseN operation #[left, right]

/-- Store a relation producer over its complete Graph-IR operand list.  Matrix and trapdoor
inputs remain direct values, so reduction can require one shared key/ordinal instead of taking a
legacy representative. -/
def validateDirectRelationDescriptor (operation : DirectRelationOperation) : Except OperationalError Unit := do
  match operation.kind with
  | .preimage maximum loopDomains =>
      let _ ← validateContextualCutoffNonnegative operation.ownerNode
        operation.parameterEnvironment loopDomains maximum
      pure ()
  | .decomposition declaredType base small digitCount loopDomains layouts =>
      let bound ← evaluateIntInvariant operation.parameterEnvironment loopDomains base
      let count ← evaluateIntInvariant operation.parameterEnvironment loopDomains digitCount
      if bound <= 1 || count <= 0 then throw (.gadgetLayoutMismatch operation.ownerNode)
      let params ← match declaredType.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      let descriptor ← resolveGadgetLayout operation.ownerNode layouts params
      let expected := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
      if bound != descriptor.base || count.toNat != expected then
        throw (.gadgetLayoutMismatch operation.ownerNode)

/-- Validate relation types after all descriptor expressions have been closed in the producer's
parameter environment.  This is stricter than the carrier's syntactic registry: equivalent
templates are accepted only when their evaluated product or digit expansion has exactly the
declared target shape. -/
def validateDirectRelationSchemas
    (operation : DirectRelationOperation)
    (inputs : Array OperationalIndexedPayloadSchema)
    (output : MatrixTypeExpr) : Except OperationalError Unit := do
  match operation.kind, inputs with
  | .preimage _ _, #[.matrix publicType, .scalar (.trapdoor trapdoorType), .matrix targetType] =>
      let publicParams ← match publicType.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      let trapdoorParams ← match trapdoorType.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      if !sameConcreteMatrixShape publicParams trapdoorParams ||
          !concreteMatrixProductMatches publicType output targetType operation.parameterEnvironment then
        throw (.outputTypeMismatch operation.ownerNode)
  | .decomposition declaredType base small digitCount loopDomains layouts, #[.matrix inputType] =>
      let declaredParams ← match declaredType.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      let inputParams ← match inputType.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      let outputParams ← match output.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      let bound ← evaluateIntInvariant operation.parameterEnvironment loopDomains base
      let count ← evaluateIntInvariant operation.parameterEnvironment loopDomains digitCount
      let descriptor ← resolveGadgetLayout operation.ownerNode layouts declaredParams
      let expectedCount := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
      if !sameConcreteMatrixShape declaredParams outputParams || bound != descriptor.base ||
          count.toNat != expectedCount || outputParams.modulus != inputParams.modulus ||
          outputParams.ringDimension != inputParams.ringDimension ||
          outputParams.rows != inputParams.rows * count.toNat ||
          outputParams.columns != inputParams.columns then
        throw (.outputTypeMismatch operation.ownerNode)
  | _, _ => throw (.outputTypeMismatch operation.ownerNode)

def OperationalExprArena.pushDirectRelationPointwise
    (arena : OperationalExprArena)
    (operation : DirectRelationOperation)
    (inputs : Array OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  validateDirectRelationDescriptor operation
  let inputIds ← inputs.mapM fun input => match input.payload with
    | .directValue id => pure id
    | .matrix id | .scalar id => throw (.unsupportedOperationalExpr id)
  let inputSchemas ← inputIds.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef id)
  if !relationOperationSchemasValid operation inputSchemas operation.outputType then
    throw (.outputTypeMismatch operation.ownerNode)
  validateDirectRelationSchemas operation inputSchemas operation.outputType
  let (direct, result) ← match arena.direct.pushPointwise (.relation operation) inputIds with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  pure ({ arena with direct }, {
    context := value.context, payload := .directValue result, storage := value.storage })

def OperationalExprArena.pushDirectValueScalarPointwise
    (arena : OperationalExprArena)
    (operation : DirectValueScalarOperation)
    (input : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let inputId ← match input.payload with
    | .directValue id => pure id
    | .matrix id | .scalar id => throw (.unsupportedOperationalExpr id)
  let inputSchema ← match arena.direct.valueAt? inputId with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef inputId)
  let (direct, result) ← match arena.direct.pushPointwise (.matrixToScalar operation) #[inputId] with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr inputId)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  if !pointwiseSchemasValid (.matrixToScalar operation) #[inputSchema] value.payload.schema then
    throw (.outputTypeMismatch operation.ownerNode)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })

def OperationalExprArena.pushDirectIntegerLiftPointwise
    (arena : OperationalExprArena)
    (operation : DirectValueMatrixOperation)
    (input : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let inputId ← match input.payload with
    | .directValue id => pure id
    | .matrix id | .scalar id => throw (.unsupportedOperationalExpr id)
  let inputSchema ← match arena.direct.valueAt? inputId with
    | some value => pure value.payload.schema
    | none => throw (.invalidOperationalExprRef inputId)
  let (direct, result) ← match arena.direct.pushPointwise (.matrixFromScalar operation) #[inputId] with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr inputId)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  if !pointwiseSchemasValid (.matrixFromScalar operation) #[inputSchema] value.payload.schema then
    throw (.outputTypeMismatch operation.ownerNode)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })

def OperationalExprArena.pushDirectBggGrouping
    (arena : OperationalExprArena)
    (vector publicKey plaintext : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let vectorId ← match vector.payload with
    | .directValue id => pure id
    | .matrix id | .scalar id => throw (.unsupportedOperationalExpr id)
  let matrixType ← match arena.direct.valueAt? vectorId with
    | some { payload := payload, .. } => match payload.schema with
      | .matrix matrixType => pure matrixType
      | .scalar _ => throw (.unsupportedOperationalExpr vectorId)
    | none => throw (.invalidOperationalExprRef vectorId)
  let operation : PrimitiveOperation := {
    kind := .bggGrouping, outputType := matrixType, ownerScope := arena.activeScope,
    ownerNode := arena.activeNode.getD 0, outputPort := 0, parameterEnvironment := [] }
  arena.pushDirectMatrixPointwiseN operation #[vector, publicKey, plaintext]

/-- Deterministic one-domain lifting shared by all primitive operations.  It inspects only
immediate Choice arguments, chooses the first domain in argument order, and never constructs or
visits a Cartesian product.  Independent domains remain nested below one delayed Primitive. -/
partial def liftPrimitiveOperation
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
      match operation.kind with
      | .multiply (.matrixMultiplyRelation _) _ =>
          throw (.unsupportedOperationalExpr operation.ownerNode)
      | _ => pure ()
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
      let domain := firstDomain
      let arena := { arena with
        domainComparisonCount := arena.domainComparisonCount + choices.size }
      let matching := choices.filter fun (_, candidate, _) => candidate == domain
      let hasExact := matching.any fun (_, _, storage) => match storage with
        | .exact _ => true
        | .shared .. => false
      let hasIndependentDomain := choices.any fun (_, candidate, _) => candidate != domain
      if hasIndependentDomain then
        match compositionalTransferRegistry (primitiveTransferClass operation) with
        | .supported _ => return ← pushImmediate arena arguments
        | .requiresConcreteStructure =>
            let liftsOneExactDomain := match operation.kind with
              | .concat _ | .bggGrouping => hasExact
              | _ => false
            if !liftsOneExactDomain then
              return ← pushImmediate arena arguments
      if hasExact then
        let mut arena := { arena with
          exactBranchVisitCount := arena.exactBranchVisitCount + domain.count }
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
          let (nextArena, output) ← liftPrimitiveOperation operation summaryOperation
            concreteTransfer evaluateRepresentative arena branchArguments _fuel
          arena := nextArena
          outputs := outputs.push output
        arena.pushPrimitiveSelection domain operation.outputType operation.parameterEnvironment
          evaluateRepresentative outputs
      else
        let representativeArguments := arguments.zip expressions |>.map fun (argument, expression) =>
          match expression.node with
          | .select candidate (.shared representative _) =>
              if candidate == domain then representative else argument
          | _ => argument
        let (arena, output) ← liftPrimitiveOperation operation summaryOperation concreteTransfer
          evaluateRepresentative arena representativeArguments _fuel
        let arena := arena
        let schemaIds := matching.filterMap fun (_, _, storage) => match storage with
          | .shared _ schema => some schema
          | .exact _ => none
        let summaries ← schemaIds.mapM arena.validatedSchema
        let mut state := OperationalExprEvaluationState.forEnvironment
          arena operation.parameterEnvironment arena.evaluationState
        let mut conservativeArguments : Array OperationalMatrixFact := #[]
        for (argument, expression) in arguments.zip expressions do
          let (fact, nextState) ← match expression.node with
            | .select candidate (.shared _ schemaId) =>
                if candidate == domain then do
                  let summary ← arena.validatedSchema schemaId
                  pure (← validateSelectedMatrixSummary argument summary, state)
                else evaluateRepresentative arena operation.parameterEnvironment argument state
            | _ =>
                evaluateRepresentative arena operation.parameterEnvironment argument state
          conservativeArguments := conservativeArguments.push fact
          state := nextState
        let arena := { arena with evaluationState := state }
        let conservativeOutput ← concreteTransfer conservativeArguments
        let outputSummary ← match transferSelectedMatrixSummary
            summaryOperation summaries conservativeOutput with
          | some summary => pure summary
          | none =>
              throw (.unsupportedOperationalExpr output)
        arena.pushCheckedSchemaEnvelope domain domain.count output outputSummary conservativeOutput

def addOperationalExprIds
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

def addIndexedOperationalFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (subtract : Bool)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) :=
  liftIndexedOperationalFact arena left right fun arena leftRoot rightRoot =>
    addOperationalExprIds nodeIndex outputPort matrixType subtract environment evaluateRepresentative
      arena leftRoot rightRoot (arena.nodes.size + 1)

def addOperationalExprFacts
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
  let (arena, result) ← addIndexedOperationalFact nodeIndex outputPort matrixType subtract environment
    evaluateRepresentative arena left right
  let arena ← arena.rememberIndexedExpr result
  pure (arena, result)

def multiplyOperationalExprIds
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
  let effectiveRule ← match rule with
    | .matrixMultiplyRelation _ => throw (.unsupportedOperationalExpr nodeIndex)
    | _ => pure rule
  let operation : PrimitiveOperation := {
    kind := .multiply effectiveRule rightWire
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
    multiplyConcreteMatrixFacts nodeIndex outputPort matrixType effectiveRule rightWire environment
      leftFact rightFact
  liftPrimitiveOperation operation .multiplyOrdinary concreteTransfer evaluateRepresentative arena
    #[left, right] fuel

def multiplyIndexedOperationalFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (rule : DerivationRule)
    (rightWire : WireRef)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) :=
  liftIndexedOperationalFact arena left right fun arena leftRoot rightRoot =>
    multiplyOperationalExprIds nodeIndex outputPort matrixType rule rightWire environment
      evaluateRepresentative arena leftRoot rightRoot (arena.nodes.size + 1)

def multiplyOperationalExprFacts
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
  let (arena, result) ← multiplyIndexedOperationalFact nodeIndex outputPort matrixType rule rightWire
    environment evaluateRepresentative arena left right
  let arena ← arena.rememberIndexedExpr result
  pure (arena, result)

def operationalProductTokens
    (term : OperationalTerm) : List OperationalCompressionToken :=
  [.productStart, .termStart term.coefficient] ++
    term.product.factors.flatMap (fun factor => match factor.leaf with
      | .primitive identity => [.primitive identity] ++
          factor.transforms.map OperationalCompressionToken.transform
      | .boundedSummary origin _ => origin.tokens
      | .exactTransform tokens _ => tokens) ++
    term.product.modes.map OperationalCompressionToken.productMode ++
    [.intermediateType term.product.outputType, .termEnd, .productEnd]

def tensorOperationalPolynomials
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
        rowCount := leftSummary.rowCount * rightSummary.rowCount
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

def tensorConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (left right : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← tensorOperationalPolynomials matrixType left.polynomial right.polynomial
    |>.mapError (flatErrorAt nodeIndex)
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial

def tensorOperationalExprIds
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

def tensorIndexedOperationalFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) :=
  liftIndexedOperationalFact arena left right fun arena leftRoot rightRoot =>
    tensorOperationalExprIds nodeIndex outputPort matrixType environment evaluateRepresentative arena
      leftRoot rightRoot (arena.nodes.size + 1)

def tensorOperationalExprFacts
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (left right : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (arena, result) ← tensorIndexedOperationalFact nodeIndex outputPort matrixType environment
    evaluateRepresentative arena left right
  let arena ← arena.rememberIndexedExpr result
  pure (arena, result)

structure OperationalExprTransformMemo where
  outputs : Std.HashMap OperationalExprId OperationalExprId := {}
  hits : Nat := 0
  misses : Nat := 0

def mapOperationalExprWithFuelCached
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
          let conservativeFact ← validateSelectedMatrixSummary representative summary
          let mappedConservativeFact ← mapFact conservativeFact
          let mappedSummary ← match transferSelectedMatrixSummary
              summaryOperation #[summary] mappedConservativeFact with
            | some value =>
                pure (structuralSummaryMap.map (fun mapSummary => mapSummary value) |>.getD value)
            | none => throw (.unsupportedOperationalExpr representative)
          let mappedSelection := mapSelection selection.identity
          if mapped == representative && mappedSelection == selection.identity &&
              mappedSummary == summary then
            pure (arena, memo, root)
          else
            let (arena, output) ← arena.pushSharedSelection mappedSelection selection.count mapped
              mappedSummary
            pure (arena, memo, output)
      pure (arena, { memo with outputs := memo.outputs.insert root output }, output)

def mapOperationalExprWithFuel
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

def mapOperationalExprM
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

def mapOperationalExpr
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

def isLoopTemplateSelection
    (binder : FamilyTemplateBinder)
    (origin : OperationalValueOrigin) : Bool :=
  let base : OperationalValueOrigin :=
    .local temporaryScope { node := binder.producerNode, port := 0 }
  let namespacedBase : OperationalValueOrigin :=
    .local binder.owner { node := binder.producerNode, port := 0 }
  let rec containsBase : OperationalValueOrigin → Bool
    | candidate@(.local ..) => candidate == base || candidate == namespacedBase
    | .loopInstance _ _ source => containsBase source
    | .indexed _ _ source => containsBase source
    | _ => false
  containsBase origin

partial def loopTemplateStaticRoot
    (binder : FamilyTemplateBinder)
    (lane : Nat) : OperationalExprArena → OperationalExprId → Nat →
    Except OperationalError (OperationalExprArena × OperationalExprId)
  | _, root, 0 => throw (.unsupportedOperationalExpr root)
  | arena, root, fuel + 1 => do
      let expression ← match arena.get? root with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef root)
      match expression.node with
      | .select selection (.exact branches) =>
          if isLoopTemplateSelection binder selection.index then
            match branches[lane]? with
            | some branch => pure (arena, branch)
            | none => throw (.invalidCount binder.producerNode lane)
          else pure (arena, root)
      | .select selection (.shared _ _) =>
          if isLoopTemplateSelection binder selection.index && lane >= selection.count then
            throw (.invalidCount binder.producerNode lane)
          else
            -- Shared deliberately has no recoverable lane array. Keep the validated pair intact;
            -- the following instantiation map specializes its representative and schema together.
            pure (arena, root)
      | .primitive operation arguments =>
          let mut arena := arena
          let mut mapped : Array OperationalExprId := #[]
          for argument in arguments do
            let (nextArena, nextArgument) ←
              loopTemplateStaticRoot binder lane arena argument fuel
            arena := nextArena
            mapped := mapped.push nextArgument
          if mapped == arguments then pure (arena, root)
          else pure (arena.pushPrimitive operation.ownerNode operation.outputPort
            operation.outputType operation.parameterEnvironment operation.kind mapped)
      | .concrete _ => pure (arena, root)

def parameterScalarPolynomial
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
    rowCount := 1
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

def transposeOperationalMatrixType (type : MatrixTypeExpr) : MatrixTypeExpr := {
  type with rows := type.columns, columns := type.rows
}

/-- A factor-level transform has only its declared output type, not the enclosing evaluated matrix
parameters.  Its sparse-row certificate is always cleared; when the output row expression is not
locally concrete, retain no usable row witness rather than carrying the old dimension across a
shape-changing transform. -/
def transformedOperationalRowCount (outputType : MatrixTypeExpr) : Int :=
  match normalizeOperationalDimension outputType.rows with
  | .constant rows => rows
  | _ => 0

def transformOperationalFactor
    (transform : OperationalFactorTransform)
    (outputType : MatrixTypeExpr)
    (factor : OperationalFactorKey) : OperationalFactorKey :=
  let transformSummary (summary : OperationalBoundedFactorSummary) := {
    summary with
    matrixType := outputType
    rowCount := transformedOperationalRowCount outputType
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

def replaceOperationalFactorAt
    (index : Nat)
    (replacement : OperationalFactorKey)
    (factors : List OperationalFactorKey) : List OperationalFactorKey :=
  let rec visit : Nat → List OperationalFactorKey → List OperationalFactorKey
    | _, [] => []
    | 0, _ :: tail => replacement :: tail
    | remaining + 1, head :: tail => head :: visit remaining tail
  visit index factors

def rowBoundaryIndex (product : OperationalProductKey) : Nat :=
  let rec visit : Nat → List OperationalProductMode → Nat
    | index, .leftPolynomialScalarBroadcast :: tail => visit (index + 1) tail
    | index, _ => index
  visit 0 product.modes

def columnBoundaryIndex (product : OperationalProductKey) : Nat :=
  let rec skipRightScalars : Nat → List OperationalProductMode → Nat
    | index, [] => index
    | index, .rightPolynomialScalarBroadcast :: tail => skipRightScalars (index - 1) tail
    | index, _ => index
  skipRightScalars (product.factors.length - 1) product.modes.reverse

def transformOperationalBoundary
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
        let rowProduct ← applyAt (rowBoundaryIndex term.product)
          (.rowEmbed .diagonal part)
          (fun matrixType => { matrixType with rows := outputType.rows })
        let index := columnBoundaryIndex rowProduct
        let factor ← match rowProduct.factors[index]? with
          | some factor => pure factor
          | none => throw .malformedProduct
        let replacement := transformOperationalFactor (.columnEmbed .diagonal part)
          { factor.outputType with columns := outputType.columns } factor
        operationalProductFromFactors (replaceOperationalFactorAt index replacement rowProduct.factors)
  pure { term with product }

def concatOperationalPolynomials
    (axis : ConcatAxis)
    (outputType : MatrixTypeExpr)
    (inputs : List OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let rows ← inputs.zipIdx.mapM fun (terms, part) =>
    terms.mapM (transformOperationalBoundary axis part outputType)
  pure (normalizeOperationalTerms rows.flatten)

def concatCanonicalRange (inputs : Array OperationalMatrixFact) : CanonicalRange :=
  if inputs.all (fun input => match input.canonicalRange with
      | .below _ => true
      | .unknown => false) then
    .below (inputs.foldl (fun result input => match input.canonicalRange with
      | .below value => max result value
      | .unknown => result) 0)
  else .unknown

/-- Negation preserves a canonical coefficient interval only for the exact-zero interval.
Ordinary canonical representatives are not closed under additive inversion: for example, the
negation of a coefficient in `[0, 2)` modulo `17` can be `16`. -/
def negateCanonicalRange : CanonicalRange → CanonicalRange
  | .below upper => if upper <= 1 then .below upper else .unknown
  | .unknown => .unknown

/-- Scaling preserves a canonical coefficient interval only for a provably identity scalar, or
for the exact-zero interval.  Other scalars require a separately proved modular range transfer. -/
def scaleCanonicalRange (scalarValues : List Int) : CanonicalRange → CanonicalRange
  | .below upper =>
      if upper <= 1 || (!scalarValues.isEmpty && scalarValues.all (· == 1)) then .below upper
      else .unknown
  | .unknown => .unknown

def concatConcreteMatrixFacts
    (nodeIndex outputPort : Nat)
    (axis : ConcatAxis)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (inputs : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  let polynomial ← concatOperationalPolynomials axis matrixType
    (inputs.toList.map (·.polynomial)) |>.mapError (flatErrorAt nodeIndex)
  let result ← polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
    (concatCanonicalRange inputs)
  pure { result with blockLayout := some { axis, partitions := inputs.map fun input => {
    matrixType := input.matrixType, polynomial := input.polynomial } } }

def transposeOperationalPolynomial
    (terms : OperationalPolynomial) : Except OperationalFlatError OperationalPolynomial := do
  let terms ← terms.mapM fun term => do
    let factors := term.product.factors.reverse.map fun factor =>
      transformOperationalFactor .transpose (transposeOperationalMatrixType factor.outputType) factor
    let product ← operationalProductFromFactors factors
    pure { term with product }
  pure (normalizeOperationalTerms terms)

def sliceOperationalPolynomial
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

def boundedStructuralTransformPolynomial
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

def transformConcreteMatrixFact
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
    | .rowEmbed axis part | .columnEmbed axis part =>
        input.polynomial.mapM (transformOperationalBoundary axis part matrixType)
          |>.mapError (flatErrorAt nodeIndex)
  let canonicalRange := match operation with
    | .negate => negateCanonicalRange input.canonicalRange
    | .transpose | .rowSlice _ _ | .columnSlice _ _ | .rowEmbed _ _ | .columnEmbed _ _ =>
        input.canonicalRange
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial canonicalRange

def transformOperationalExprId
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

def transformIndexedOperationalFact
    (nodeIndex outputPort : Nat)
    (matrixType : MatrixTypeExpr)
    (transform : OperationalFactorTransform)
    (environment : ParamEnvironment)
    (evaluateRepresentative : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (input : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) :=
  mapIndexedOperationalFact arena input fun arena root =>
    transformOperationalExprId nodeIndex outputPort matrixType transform environment
      evaluateRepresentative arena root (arena.nodes.size + 1)

def transformOperationalExprFact
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
  let (arena, result) ← transformIndexedOperationalFact nodeIndex outputPort matrixType operation
    environment evaluateRepresentative arena input
  let arena ← arena.rememberIndexedExpr result
  pure (arena, result)

def scaleConcreteMatrixFact
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
  polynomialMatrixFact nodeIndex outputPort matrixType environment polynomial
    (scaleCanonicalRange scalarValues input.canonicalRange)

def scaleOperationalExprId
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
    (root : OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  if !scalarValues.isEmpty && scalarValues.all (· == 1) then return (arena, root)
  let operation : PrimitiveOperation := {
    kind := .scale scalar scalarValues loopDomains
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
    scaleConcreteMatrixFact nodeIndex outputPort matrixType scalar scalarValues environment
      loopDomains input
  liftPrimitiveOperation operation .scale concreteTransfer evaluateRepresentative arena #[root] fuel

def scaleIndexedOperationalFact
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
    (input : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) :=
  mapIndexedOperationalFact arena input fun arena root =>
    scaleOperationalExprId nodeIndex outputPort matrixType scalar scalarValues environment loopDomains
      evaluateRepresentative arena root (arena.nodes.size + 1)

def scaleOperationalExprFact
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
  let (arena, result) ← scaleIndexedOperationalFact nodeIndex outputPort matrixType scalar scalarValues
    environment loopDomains evaluateRepresentative arena input
  let arena ← arena.rememberIndexedExpr result
  pure (arena, result)

/-- A fixed-assignment direct kernel constructs a fresh executable output.  Its owner descriptor,
not the identities carried by its operands, determines the output wire and scope.  In particular
this changes neither polynomial factors nor relation snapshots transported from the inputs. -/
def directPointwiseMatrixOutput
    (ownerScope : Option ScopeTemplateKey)
    (ownerNode outputPort : Nat)
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let scope := ownerScope.getD temporaryScope
  { fact with
    subject := { node := ownerNode, port := outputPort }
    origin := .value scope { node := ownerNode, port := outputPort }
  }

/-- Matrix-to-scalar kernels likewise construct a fresh scalar output without rebinding any
matrix operand identity. -/
def directPointwiseScalarOutput
    (ownerScope : Option ScopeTemplateKey)
    (ownerNode outputPort : Nat)
    (fact : OperationalScalarFact) : OperationalScalarFact :=
  let scope := ownerScope.getD temporaryScope
  match fact with
  | .integer value => .integer {
      value with
      subject := { node := ownerNode, port := outputPort }
      origin := .local scope { node := ownerNode, port := outputPort }
    }
  | value => value

/-- Concrete operands accepted by the relation-producing direct kernel.  This deliberately has
no arena reference: all identities, target snapshots, and trapdoor ownership come from the
already-correlated physical lane facts. -/
inductive DirectRelationArgument where
  | matrix (fact : OperationalMatrixFact)
  | trapdoor (fact : OperationalTrapdoorFact)

def rebindDirectRelationProducer
    (output : OperationalMatrixFact) : OperationalMatrixFact :=
  ({ output with relations := output.relations.map fun relation => match relation with
    | OperationalMatrixRelation.preimage value =>
        OperationalMatrixRelation.preimage { value with producer := output.origin }
    | OperationalMatrixRelation.decomposition value =>
        OperationalMatrixRelation.decomposition { value with producer := output.origin }
  }).refreshPrimitivePolynomial

/-- Apply one fully-aligned preimage/decomposition lane.  All graph operands are explicit in
`arguments`; this never selects a representative from another family or consults an arena. -/
def applyDirectRelationProducer
    (operation : DirectRelationOperation)
    (matrixType : MatrixTypeExpr)
    (arguments : Array DirectRelationArgument) : Except OperationalError OperationalMatrixFact := do
  if operation.outputType != matrixType then throw (.unsupportedOperationalExpr operation.ownerNode)
  let output ← match operation.kind with
  | .preimage maximum loopDomains => do
      let publicFact ← match arguments[0]? with
        | some (DirectRelationArgument.matrix fact) => pure fact
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let trapdoor ← match arguments[1]? with
        | some (DirectRelationArgument.trapdoor fact) => pure fact
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let target ← match arguments[2]? with
        | some (DirectRelationArgument.matrix fact) => pure fact
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 3 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let publicIdentity ← match publicFact.identity with
        | some identity => pure identity
        | none => throw (.missingPublicIdentity operation.ownerNode { node := 0, port := 0 })
      if publicIdentity != trapdoor.publicIdentity then throw (.publicIdentityMismatch operation.ownerNode)
      let _ ← validatePreimageCutoffAgreement operation.ownerNode operation.parameterEnvironment loopDomains
        maximum trapdoor.publicIdentity trapdoor.preimageCutoff
      let bound := OperationalBoundExpr.contextual .maximum operation.parameterEnvironment loopDomains maximum
      let result ← cappedMatrixFactExpr operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment bound
      let relation : PreimageRelation := {
        producer := result.origin, publicIdentity, targetOrigin := target.origin,
        targetSummary := matrixTargetSummary target }
      pure ({ result with relations := [.preimage relation] }).refreshPrimitivePolynomial
  | .decomposition declaredType base small digitCount loopDomains layouts => do
      let input ← match arguments with
        | #[DirectRelationArgument.matrix fact] => pure fact
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let bound ← evaluateIntInvariant operation.parameterEnvironment loopDomains base
      let count ← evaluateIntInvariant operation.parameterEnvironment loopDomains digitCount
      if bound <= 1 || count <= 0 then throw (.gadgetLayoutMismatch operation.ownerNode)
      let params ← match declaredType.evaluate operation.parameterEnvironment (.constant 0) with
        | some value => pure value | none => throw (.invalidMatrixParameters operation.ownerNode)
      let descriptor ← resolveGadgetLayout operation.ownerNode layouts params
      let expectedCount := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
      if count.toNat != expectedCount || bound != descriptor.base then
        throw (.gadgetLayoutMismatch operation.ownerNode)
      let publicIdentity := PublicMatrixIdentity.gadget descriptor.paramsId params
        input.matrixParams.rows bound small count.toNat
      let result ← cappedMatrixFact operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment (Int.ofNat (Mxx.gadgetDecompositionBound bound small))
      let status := if !small then ReconstructionStatus.available else
        match input.canonicalRange with
        | .below upper => if upper <= descriptor.smallestCrtModulus then .available
            else .smallRangeMissing descriptor.smallestCrtModulus
        | .unknown => .smallRangeMissing descriptor.smallestCrtModulus
      let relation : DecompositionRelation := {
        producer := result.origin, publicIdentity, inputOrigin := input.origin,
        inputSummary := matrixTargetSummary input, base := bound, small := small,
        digitCount := count.toNat, status }
      let canonicalRange : CanonicalRange :=
        if small then .below bound.natAbs else .unknown
      let output : OperationalMatrixFact := { result with
        canonicalRange := canonicalRange
        relations := [OperationalMatrixRelation.decomposition relation]
      }
      pure output.refreshPrimitivePolynomial
  pure (rebindDirectRelationProducer
    (directPointwiseMatrixOutput operation.ownerScope operation.ownerNode operation.outputPort output))

/-- Apply a matrix-only delayed pointwise descriptor to already aligned concrete inputs.  Both
fixed-assignment evaluation and structural direct-family reduction use this one dispatcher, so a
new operation cannot accidentally acquire different transfer semantics in the two paths. -/
def applyDirectMatrixPointwiseOperation
    (operation : PrimitiveOperation)
    (matrixType : MatrixTypeExpr)
    (arguments : Array OperationalMatrixFact) : Except OperationalError OperationalMatrixFact := do
  if operation.outputType != matrixType then throw (.unsupportedOperationalExpr operation.ownerNode)
  let output ← match operation.kind with
  | .add subtract => do
      let left ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let right ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 2 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      addConcreteMatrixFacts operation.ownerNode operation.outputPort matrixType subtract
        operation.parameterEnvironment left right
  | .multiply rule rightWire => do
      let left ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let right ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 2 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      multiplyConcreteMatrixFacts operation.ownerNode operation.outputPort matrixType rule rightWire
        operation.parameterEnvironment left right
  | .tensor => do
      let left ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let right ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 2 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      tensorConcreteMatrixFacts operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment left right
  | .concat axis =>
      concatConcreteMatrixFacts operation.ownerNode operation.outputPort axis matrixType
        operation.parameterEnvironment arguments
  | .transform transform => do
      let input ← match arguments with
        | #[input] => pure input
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      transformConcreteMatrixFact operation.ownerNode operation.outputPort matrixType transform
        operation.parameterEnvironment input
  | .slice rows columns => do
      let input ← match arguments with
        | #[input] => pure input
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let polynomial ← sliceOperationalPolynomial rows columns matrixType input.polynomial
        |>.mapError (flatErrorAt operation.ownerNode)
      polynomialMatrixFact operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment polynomial input.canonicalRange
  | .scale scalar values loopDomains => do
      let input ← match arguments with
        | #[input] => pure input
        | _ => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      scaleConcreteMatrixFact operation.ownerNode operation.outputPort matrixType scalar values
        operation.parameterEnvironment loopDomains input
  | _ => throw (.unsupportedOperationalExpr operation.ownerNode)
  pure (directPointwiseMatrixOutput operation.ownerScope operation.ownerNode operation.outputPort output)

/-- The structural direct reducer uses this instrumented twin.  It is intentionally private to
reduction telemetry: fixed-assignment queries and acceptance retain the fact-only dispatcher. -/
private def applyDirectMatrixPointwiseOperationWithRelationRewriteCount
    (operation : PrimitiveOperation)
    (matrixType : MatrixTypeExpr)
    (arguments : Array OperationalMatrixFact) : Except OperationalError (OperationalMatrixFact × Nat) := do
  if operation.outputType != matrixType then throw (.unsupportedOperationalExpr operation.ownerNode)
  match operation.kind with
  | .multiply rule rightWire => do
      let left ← match arguments[0]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let right ← match arguments[1]? with
        | some fact => pure fact | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      if arguments.size != 2 then throw (.unsupportedOutputArity operation.ownerNode arguments.size)
      let (output, rewriteCount) ← multiplyConcreteMatrixFactsWithRelationRewriteCount
        operation.ownerNode operation.outputPort matrixType rule rightWire operation.parameterEnvironment left right
      pure (directPointwiseMatrixOutput operation.ownerScope operation.ownerNode operation.outputPort output,
        rewriteCount)
  | _ => pure (← applyDirectMatrixPointwiseOperation operation matrixType arguments, 0)

def applyDirectMatrixFromScalarOperation
    (operation : DirectValueMatrixOperation)
    (matrixType : MatrixTypeExpr)
    (input : OperationalScalarFact) : Except OperationalError OperationalMatrixFact := do
  let integer ← match input with
    | .integer value => pure value
    | _ => throw (.operandNotInteger operation.ownerNode { node := 0, port := 0 })
  let output ← match operation.kind with
  | .liftIntegerToConstantPolynomial declaredType => do
      if !operationalMatrixTypeEqual declaredType matrixType then
        throw (.outputTypeMismatch operation.ownerNode)
      let params ← match matrixType.evaluate operation.parameterEnvironment (.constant 0) with
        | some params => pure params
        | none => throw (.invalidMatrixParameters operation.ownerNode)
      if params.rows != 1 || params.columns != 1 || params.modulus <= 0 ||
          params.ringDimension == 0 then
        throw (.invalidMatrixParameters operation.ownerNode)
      let bound := OperationalBoundExpr.maximum
        (.negate integer.lowerExpression) integer.upperExpression
      classifiedMatrixFactExpr operation.ownerNode operation.outputPort matrixType
        operation.parameterEnvironment bound false (.below params.modulus.toNat)
        { isConstantPolynomial := true }
  pure (directPointwiseMatrixOutput operation.ownerScope operation.ownerNode operation.outputPort output)

def applyDirectMatrixToScalarOperation
    (operation : DirectValueScalarOperation)
    (matrix : OperationalMatrixFact) : Except OperationalError OperationalScalarFact := do
  let output ← match operation.kind with
  | .extractCoefficient position => do
      let position ← evaluateIntInvariant operation.parameterEnvironment [] position
      if position < 0 || position >= Int.ofNat matrix.matrixParams.ringDimension then
        throw (.invalidCount operation.ownerNode position)
      let upper := match matrix.canonicalRange with
        | .below upper => Int.ofNat upper
        | .unknown => matrix.matrixParams.modulus
      if upper <= 0 then throw (.invalidMatrixParameters operation.ownerNode)
      integerFact operation.ownerNode operation.outputPort 0 (upper - 1)
  | .thresholdDecodeBool ciphertextModulus plaintextModulus length => do
      let ciphertext ← evaluateIntInvariant operation.parameterEnvironment [] ciphertextModulus
      let plaintext ← evaluateIntInvariant operation.parameterEnvironment [] plaintextModulus
      let count ← evaluateIntInvariant operation.parameterEnvironment [] length
      if matrix.matrixParams.rows != 1 || matrix.matrixParams.columns != 1 ||
          ciphertext != matrix.matrixParams.modulus || plaintext <= 1 || count <= 0 ||
          count > Int.ofNat matrix.matrixParams.ringDimension then
        throw (.invalidMatrixParameters operation.ownerNode)
      pure .boolean
  | .thresholdDecodeInt ciphertextModulus plaintextModulus length => do
      let ciphertext ← evaluateIntInvariant operation.parameterEnvironment [] ciphertextModulus
      let plaintext ← evaluateIntInvariant operation.parameterEnvironment [] plaintextModulus
      let count ← evaluateIntInvariant operation.parameterEnvironment [] length
      if matrix.matrixParams.rows != 1 || matrix.matrixParams.columns != 1 ||
          ciphertext != matrix.matrixParams.modulus || plaintext <= 1 || count <= 0 ||
          count > Int.ofNat matrix.matrixParams.ringDimension then
        throw (.invalidMatrixParameters operation.ownerNode)
      integerFact operation.ownerNode operation.outputPort 0 (plaintext - 1)
  pure (directPointwiseScalarOutput operation.ownerScope operation.ownerNode operation.outputPort output)

def applyDirectScalarPointwiseOperation
    (kind : OperationalScalarPrimitiveKind)
    (arguments : Array OperationalScalarFact) : Except OperationalError OperationalScalarFact := do
  match kind, arguments with
  | .boolToInt, #[.boolean] => integerFact 0 0 0 1
  | .intBinary operation, #[.integer left, .integer right] => do
      let interval ← integerBinaryInterval 0 operation left right
      integerFactWithExpressions 0 0 interval.lower interval.upper
        interval.lowerExpression interval.upperExpression
  | .intCompare _, #[.integer _, .integer _] => pure .boolean
  | .intToReal, #[.integer _] => pure .real
  | .realBinary _, #[.real, .real] => pure .real
  | .realSqrt, #[.real] => pure .real
  | _, _ => throw (.unsupportedOperationalExpr 0)



mutual

/-- Evaluate one direct indexed matrix value at a complete index assignment.  This is the only
place direct delayed nodes invoke the fixed-assignment matrix kernels; it does not inspect the
legacy selection DAG. -/
def DirectOperationalIndexedArena.matrixFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (indices : IndexValueEnvironment)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError OperationalMatrixFact
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      if !completeDirectIndexAssignment parameters value.context indices then
        throw (.unsupportedOperationalExpr id)
      match value.payload with
      | .shared (.matrix _) (.matrix reference) =>
          match arena.fixed.matrices[reference]? with
          | some fact => pure fact
          | none => throw (.invalidOperationalExprRef reference)
      | .explicit (.matrix _) binder references => do
          let lane ← match evaluateIndexExpr parameters value.context indices (.variable binder) with
            | some lane => pure lane
            | none => throw (.unsupportedOperationalExpr id)
          let reference ← match references[lane.toNat]? with
            | some (.matrix reference) => pure reference
            | some _ => throw (.unsupportedOperationalExpr id)
            | none => throw (.invalidOperationalExprRef lane.toNat)
          match arena.fixed.matrices[reference]? with
          | some fact => pure fact
          | none => throw (.invalidOperationalExprRef reference)
      | .explicitValues (.matrix _) binder values => do
          let lane ← match evaluateIndexExpr parameters value.context indices (.variable binder) with
            | some lane => pure lane
            | none => throw (.unsupportedOperationalExpr id)
          let branch ← match values[lane.toNat]? with
            | some branch => pure branch
            | none => throw (.invalidOperationalExprRef lane.toNat)
          arena.matrixFactAt parameters indices branch fuel
      | .mapped (.matrix _) source map => do
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          let sourceIndices ← map.source.binders.toList.mapM fun binder => do
            let expression ← match map.assignmentFor binder with
              | some expression => pure expression
              | none => throw (.unsupportedOperationalExpr id)
            let lane ← match evaluateIndexExpr parameters value.context indices expression with
              | some lane => pure lane
              | none => throw (.unsupportedOperationalExpr id)
            pure (.variable binder, lane)
          arena.matrixFactAt parameters sourceIndices source fuel
      | .matrixResultBound (.matrix _) source totalHardBound => do
          let source ← arena.matrixFactAt parameters indices source fuel
          pure { source with totalHardBound }
      | .pointwise (.matrix matrixType) (.matrix operation) inputs => do
          let arguments ← inputs.mapM fun input => arena.matrixFactAt parameters indices input fuel
          applyDirectMatrixPointwiseOperation operation matrixType arguments
      | .pointwise (.matrix matrixType) (.relation operation) inputs => do
          let arguments ← inputs.mapM fun input => do
            let value ← match arena.valueAt? input with
              | some value => pure value
              | none => throw (.invalidOperationalExprRef input)
            match value.payload.schema with
            | .matrix _ => return .matrix (← arena.matrixFactAt parameters indices input fuel)
            | .scalar (.trapdoor _) =>
                match ← arena.scalarFactAt parameters indices input fuel with
                | .trapdoor fact => return .trapdoor fact
                | _ => throw (.unsupportedOperationalExpr input)
            | .scalar _ => throw (.unsupportedOperationalExpr input)
          applyDirectRelationProducer operation matrixType arguments
      | .pointwise (.matrix matrixType) (.matrixFromScalar operation) inputs => do
          let input ← match inputs with
            | #[input] => pure input
            | _ => throw (.unsupportedOutputArity operation.ownerNode inputs.size)
          applyDirectMatrixFromScalarOperation operation matrixType
            (← arena.scalarFactAt parameters indices input fuel)
      | _ => throw (.unsupportedOperationalExpr id)

/-- Evaluate a direct indexed scalar at a complete assignment. Matrix-to-scalar kernels evaluate
their matrix input at that identical assignment, preserving shared-selector correlation. -/
def DirectOperationalIndexedArena.scalarFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (indices : IndexValueEnvironment)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError OperationalScalarFact
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      if !completeDirectIndexAssignment parameters value.context indices then
        throw (.unsupportedOperationalExpr id)
      match value.payload with
      | .shared (.scalar _) (.scalar reference) =>
          match arena.fixed.scalars[reference]? with
          | some fact => pure fact
          | none => throw (.invalidOperationalExprRef reference)
      | .explicit (.scalar _) binder references => do
          let lane ← match evaluateIndexExpr parameters value.context indices (.variable binder) with
            | some lane => pure lane
            | none => throw (.unsupportedOperationalExpr id)
          let reference ← match references[lane.toNat]? with
            | some (.scalar reference) => pure reference
            | some _ => throw (.unsupportedOperationalExpr id)
            | none => throw (.invalidOperationalExprRef lane.toNat)
          match arena.fixed.scalars[reference]? with
          | some fact => pure fact
          | none => throw (.invalidOperationalExprRef reference)
      | .explicitValues (.scalar _) binder values => do
          let lane ← match evaluateIndexExpr parameters value.context indices (.variable binder) with
            | some lane => pure lane
            | none => throw (.unsupportedOperationalExpr id)
          let branch ← match values[lane.toNat]? with
            | some branch => pure branch
            | none => throw (.invalidOperationalExprRef lane.toNat)
          arena.scalarFactAt parameters indices branch fuel
      | .mapped (.scalar _) source map => do
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          let sourceIndices ← map.source.binders.toList.mapM fun binder => do
            let expression ← match map.assignmentFor binder with
              | some expression => pure expression
              | none => throw (.unsupportedOperationalExpr id)
            let lane ← match evaluateIndexExpr parameters value.context indices expression with
              | some lane => pure lane
              | none => throw (.unsupportedOperationalExpr id)
            pure (.variable binder, lane)
          arena.scalarFactAt parameters sourceIndices source fuel
      | .pointwise (.scalar _) (.matrixToScalar operation) inputs => do
          let input ← match inputs with
            | #[input] => pure input
            | _ => throw (.unsupportedOutputArity operation.ownerNode inputs.size)
          applyDirectMatrixToScalarOperation operation
            (← arena.matrixFactAt parameters indices input fuel)
      | .pointwise (.scalar _) (.scalar kind) inputs => do
          let arguments ← inputs.mapM fun input => arena.scalarFactAt parameters indices input fuel
          applyDirectScalarPointwiseOperation kind arguments
      | _ => throw (.unsupportedOperationalExpr id)

end

/-- Read a direct matrix value only after its indexed context has been fully assigned.  Empty
contexts are complete assignments, so ordinary direct wires never fall back to the removed
selection evaluator. -/
def OperationalExprArena.directValueFactAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError OperationalMatrixFact := do
  let root ← match expression.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  if !expression.context.binders.isEmpty then throw (.unsupportedOperationalExpr root)
  arena.direct.matrixFactAt environment [] root (arena.direct.values.size + 1)

def OperationalExprArena.directValueScalarFactAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError OperationalScalarFact := do
  let root ← match expression.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  if !expression.context.binders.isEmpty then throw (.unsupportedOperationalExpr root)
  arena.direct.scalarFactAt environment [] root (arena.direct.values.size + 1)

def integerFactAt
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef) : Except OperationalError OperationalIntegerFact := do
  match ← lookupFact node facts wire with
  | expression@{ payload := .scalar _, .. } =>
      match ← facts.arena.concreteIndexedScalar expression with
      | OperationalScalarFact.integer fact => pure fact
      | _ => throw (.operandNotInteger node wire)
  | expression@{ payload := .directValue _, .. } =>
      match ← facts.arena.directValueScalarFactAt [] expression with
      | OperationalScalarFact.integer fact => pure fact
      | _ => throw (.operandNotInteger node wire)
  | _ => throw (.operandNotInteger node wire)

/-- Read the complete interval of an ordered direct integer family without choosing one lane.
Only fixed scalar tables and mapped views thereof are accepted here; arithmetic/opaque scalar
nodes remain fail-closed until their indexed interval transfer is implemented. -/
partial def DirectOperationalIndexedArena.integerInterval
    (arena : DirectOperationalIndexedArena)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError (Int × Int)
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      let intervals ← match value.payload with
        | .shared (.scalar .integer) (.scalar reference) =>
            match arena.fixed.scalars[reference]? with
            | some (.integer fact) => pure [(fact.lower, fact.upper)]
            | _ => throw (.operandNotInteger 0 { node := 0, port := 0 })
        | .explicit (.scalar .integer) _ references => references.toList.mapM fun reference =>
            match reference with
            | .scalar scalar => match arena.fixed.scalars[scalar]? with
              | some (.integer fact) => pure (fact.lower, fact.upper)
              | _ => throw (.operandNotInteger 0 { node := 0, port := 0 })
            | .matrix _ => throw (.operandNotInteger 0 { node := 0, port := 0 })
        | .explicitValues (.scalar .integer) _ values =>
            values.toList.mapM fun child => arena.integerInterval child fuel
        | .mapped (.scalar .integer) source _ => return ← arena.integerInterval source fuel
        | _ => throw (.operandNotInteger 0 { node := 0, port := 0 })
      match intervals with
      | [] => throw (.invalidCount 0 0)
      | first :: rest => pure <| rest.foldl (fun (lower, upper) (nextLower, nextUpper) =>
          (min lower nextLower, max upper nextUpper)) first

def directSingleIndexBinder
    (node : Nat)
    (expression : IndexedOperationalFact) : Except OperationalError IndexVariable :=
  match expression.context.binders.toList with
  | [binder] => pure binder
  | _ => throw (.loopInputModeMismatch node 0)

/-- Materialize the exact prepared-scope path and integer-family producer wire for a gather
owner.  Source-family identity is retained by the gathered matrix provenance instead. -/
def operationalGatherIndicesWire (scope : ScopeTemplateKey) (wire : WireRef) : GatherLookupWire := {
  scope := scope.toGatherScopeTemplateKey
  node := wire.node
  port := wire.port
}

/-- Read a deterministic representative of a direct indexed matrix value for structural bound
queries.  Each free binder is assigned its first valid lane; the direct carrier retains the
complete context for identity-sensitive rewrites, while the fixed transfer supplies its
construction-uniform hard bound. -/
def OperationalExprArena.directValueRepresentativeFactAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError OperationalMatrixFact := do
  let root ← match expression.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  if !validateContext expression.context then throw (.unsupportedOperationalExpr root)
  let indices := expression.context.binders.toList.map fun binder => (.variable binder, 0)
  arena.direct.matrixFactAt environment indices root (arena.direct.values.size + 1)

/-- Enumerate every declared direct-index assignment before reducing a family to one numeric
bound.  This is deliberately structural: an unresolved cardinality or an unevaluable mapped
selector remains an error instead of silently choosing lane zero. -/
def directIndexAssignments
    (environment : ParamEnvironment)
    (context : IndexContext) : Except OperationalError (List IndexValueEnvironment) := do
  if !validateContext context then throw (.unsupportedOperationalExpr context.binders.size)
  context.binders.foldrM (fun binder tails => do
    let count ← match binder.count.evaluate environment with
      | some count => pure count
      | none => throw .nonClosedExpression
    if count <= 0 then throw (.invalidCount binder.slot count)
    pure <| (List.range count.toNat).flatMap fun lane =>
      tails.map fun tail => (.variable binder, Int.ofNat lane) :: tail) [[]]

/-- Materialize every fixed assignment of one direct indexed matrix value.  The carrier itself
continues to store a compact table or mapped template; only bound consumers enumerate the finite
context, exactly as they do for legacy exact selections. -/
def OperationalExprArena.directValueFactsAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError (List OperationalMatrixFact) := do
  let root ← match expression.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  let assignments ← directIndexAssignments environment expression.context
  assignments.mapM fun indices =>
    arena.direct.matrixFactAt environment indices root (arena.direct.values.size + 1)

/-- The canonical structural driver of one correlated direct-family lane.  This deliberately
retains the complete owner-bearing index expression; its ordinal records the selected physical
lane without reconstructing a root assignment environment. -/
abbrev DirectCorrelationKey := IndexExpr

structure ReducedDirectMatrixFact where
  key : Option DirectCorrelationKey
  ordinal : Nat
  fact : OperationalMatrixFact

structure ReducedDirectScalarFact where
  key : Option DirectCorrelationKey
  ordinal : Nat
  fact : OperationalScalarFact

structure ReducedDirectRelationArgument where
  key : Option DirectCorrelationKey
  ordinal : Nat
  payload : DirectRelationArgument

/- The reporting identity of a map deliberately omits `closedIndex`: that field records how a
transport was admitted, whereas a rewrite event is identified by its fully transported source
assignment.  Thus `A → 0` and `A → B → 0` normalize to the same event path. -/
private structure DirectRelationRewriteMapKey where
  source : IndexContext
  destination : IndexContext
  assignments : Array IndexExpr
  deriving BEq, DecidableEq, Repr

private def directRelationRewriteMapKey (map : IndexMap) : DirectRelationRewriteMapKey := {
  source := map.source, destination := map.destination, assignments := map.assignments }

private def normalizeDirectRelationRewriteMaps (maps : List IndexMap) : List DirectRelationRewriteMapKey :=
  let rec normalize (current : IndexMap) : List IndexMap → Option DirectRelationRewriteMapKey
    | [] => some (directRelationRewriteMapKey current)
    | next :: remaining => do
        if current.destination != next.source then none else pure ()
        let assignments ← current.assignments.toList.mapM (reindex next)
        let composed : IndexMap := {
          source := current.source, destination := next.destination, assignments := assignments.toArray }
        normalize composed remaining
  /- Reducer descent prepends each outer map, so restore source-to-destination order before
  composing. -/
  match maps.reverse with
  | [] => []
  | first :: remaining => match normalize first remaining with
    | some key => [key]
    | none => maps.map directRelationRewriteMapKey

/-- Reindexing clones a delayed carrier node, so its allocation ID is not an event identity.
The executable owner identifies the one pointwise graph operation across those clones. -/
private structure DirectRelationRewritePointwiseKey where
  ownerScope : Option ScopeTemplateKey
  ownerNode : Nat
  outputPort : Nat
  deriving BEq, DecidableEq, Repr

/-- One successful exact relation application during structural direct reduction.  The pointwise
node, normalized fully transported map path, correlated physical lane, and local rewrite ordinal
make events stable across shared DAG traversal and repeated bound queries. -/
structure DirectRelationRewriteEventKey where
  pointwise : DirectRelationRewritePointwiseKey
  maps : List DirectRelationRewriteMapKey
  key : Option DirectCorrelationKey
  ordinal : Nat
  localOrdinal : Nat
  deriving BEq, DecidableEq, Repr

private def deduplicateDirectRelationRewriteEvents
    (events : List DirectRelationRewriteEventKey) : List DirectRelationRewriteEventKey :=
  events.foldl (fun retained event => if retained.contains event then retained else retained ++ [event]) []

private structure ReducedDirectMatrixEvaluation where
  entries : List ReducedDirectMatrixFact
  rewriteEvents : List DirectRelationRewriteEventKey := []

private structure ReducedDirectScalarEvaluation where
  entries : List ReducedDirectScalarFact
  rewriteEvents : List DirectRelationRewriteEventKey := []

/-- The sole correlation zipper for mixed direct relation operands.  A singleton shared input is
allowed with every lane; otherwise every physical operand must have the driver's exact key and
ordinal.  This is intentionally the same fail-closed rule as the existing matrix/scalar zippers. -/
def alignDirectRelationArguments
    (owner id : Nat)
    (inputs : List (List ReducedDirectRelationArgument)) :
    Except OperationalError (List (Array DirectRelationArgument × Option DirectCorrelationKey × Nat)) := do
  if inputs.isEmpty || inputs.any List.isEmpty then throw (.invalidCount owner 0)
  let driver? := inputs.find? fun entries => entries.length > 1
  match driver? with
  | some driver => driver.mapM fun driverEntry => do
      let arguments ← inputs.mapM fun entries => match entries with
        | [entry] =>
            if entry.key.isNone || (entry.key == driverEntry.key && entry.ordinal == driverEntry.ordinal)
            then pure entry.payload else throw (.unsupportedOperationalExpr id)
        | _ => match entries.find? fun entry =>
          entry.key == driverEntry.key && entry.ordinal == driverEntry.ordinal with
          | some entry => pure entry.payload
          | none => throw (.unsupportedOperationalExpr id)
      pure (arguments.toArray, driverEntry.key, driverEntry.ordinal)
  | none => do
      let entries ← inputs.mapM fun entries => match entries with
        | [entry] => pure entry
        | _ => throw (.unsupportedOperationalExpr id)
      let driver? := entries.find? fun entry => entry.key.isSome
      match driver? with
      | some driver =>
          if entries.all fun entry => entry.key.isNone ||
              (entry.key == driver.key && entry.ordinal == driver.ordinal) then
            pure [(entries.map (·.payload) |>.toArray, driver.key, driver.ordinal)]
          else throw (.unsupportedOperationalExpr id)
      | none => pure [(entries.map (·.payload) |>.toArray, none, 0)]

private def transportDirectCorrelation
    (parameters : ParamEnvironment)
    (id : OperationalIndexedValueId)
    (maps : List IndexMap)
    (key : Option DirectCorrelationKey)
    (ordinal : Nat) : Except OperationalError (Option (Option DirectCorrelationKey × Nat)) := do
  let mut key := key
  let mut ordinal := ordinal
  for map in maps do
    match key with
    | none => pure ()
    | some source => do
        /- A gather's codomain is an `IntExpr` domain witness, so its free index atoms contain
        only the runtime position.  It nevertheless substitutes the mapped source family lane;
        always transport the complete gather key through that map. -/
        let transports := match source with
          | .gather _ _ _ => true
          | _ => source.freeVariables.any map.source.binders.contains
        if transports then
          let translated ← match reindex map source with
            | some translated => pure translated
            | none => throw (.unsupportedOperationalExpr id)
          match translated with
          | .constant lane =>
              if ordinal != lane then return none
              key := none
              ordinal := 0
          | .offset (.constant lane) amount =>
              let lane := Int.ofNat lane + amount
              if lane < 0 || ordinal != lane.toNat then return none
              key := none
              ordinal := 0
          | .variable destination => key := some (.variable destination)
          | .offset (.variable destination) amount =>
              let destinationOrdinal := Int.ofNat ordinal - amount
              let count ← match destination.count.evaluate parameters with
                | some count => pure count
                | none => throw .nonClosedExpression
              if destinationOrdinal < 0 || destinationOrdinal >= count then return none
              key := some (.variable destination)
              ordinal := destinationOrdinal.toNat
          /- A gather is a dependent function application.  `ordinal` remains the physical
          source-table lane, while the complete owner-bearing gather expression names the
          (possibly repeated) runtime lookup.  This represents every source lane once and never
          expands it for each output position.  Two gathered operands can therefore zip only
          when both the source-index value and lookup-position identities are structurally exact.
          The fixed leaves have already been reindexed through this same map by
          `reindexDirectMatrixFact`, so relation owners, targets, and provenance carry this
          exact gather identity as well. -/
          | gathered@(.gather _ _ _) => key := some gathered
          | .offset _ _ => throw (.unsupportedOperationalExpr id)
  pure (some (key, ordinal))

mutual

private def reducedDirectMatrixFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (maps : List IndexMap)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError ReducedDirectMatrixEvaluation
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      match value.payload with
      | .shared (.matrix _) (.matrix reference) => do
          let fact ← match arena.fixed.matrices[reference]? with
            | some fact => pure fact
            | none => throw (.invalidOperationalExprRef reference)
          pure { entries := [{ key := none, ordinal := 0, fact }] }
      | .explicit (.matrix _) binder references => do
          let mapped ← references.toList.mapIdxM fun ordinal reference => do
            let fact ← match reference with
              | .matrix reference => match arena.fixed.matrices[reference]? with
                | some fact => pure fact
                | none => throw (.invalidOperationalExprRef reference)
              | .scalar _ => throw (.unsupportedOperationalExpr id)
            let entry := if references.size == 1 then (none, 0) else (some (.variable binder), ordinal)
            match ← transportDirectCorrelation parameters id maps entry.1 entry.2 with
            | some (key, ordinal) => pure (some { key, ordinal, fact })
            | none => pure none
          let entries := mapped.filterMap fun entry => entry
          pure { entries := entries }
      | .explicitValues (.matrix _) binder values => do
          let lanes ← values.toList.mapIdxM fun ordinal child => do
            let outer := if values.size == 1 then (none, 0) else (some (.variable binder), ordinal)
            let outer ← transportDirectCorrelation parameters id maps outer.1 outer.2
            match outer with
            | none => pure ({ entries := [] } : ReducedDirectMatrixEvaluation)
            | some (none, _) =>
                let evaluation ← reducedDirectMatrixFactAt arena parameters maps child fuel
                return evaluation
            | some (some outerKey, outerOrdinal) => do
                let evaluation ← reducedDirectMatrixFactAt arena parameters maps child fuel
                let entries := evaluation.entries
                match entries with
                | [entry] => match entry.key with
                    | none =>
                        let entry := { entry with key := some outerKey, ordinal := outerOrdinal }
                        pure { entries := [entry], rewriteEvents := evaluation.rewriteEvents }
                    | some _ =>
                        if entry.key == some outerKey && entry.ordinal == outerOrdinal then
                          pure evaluation
                        else throw (.unsupportedOperationalExpr id)
                | _ =>
                    match entries.filter fun entry =>
                      entry.key == some outerKey && entry.ordinal == outerOrdinal with
                    | [entry] => pure { entries := [entry], rewriteEvents := evaluation.rewriteEvents }
                    | _ => throw (.unsupportedOperationalExpr id)
          let entries := lanes.flatMap (fun lane => lane.entries)
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (lanes.flatMap (fun lane => lane.rewriteEvents))
          pure { entries := entries, rewriteEvents := rewriteEvents }
      | .mapped (.matrix _) source map => do
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          reducedDirectMatrixFactAt arena parameters (map :: maps) source fuel
      | .matrixResultBound (.matrix _) source totalHardBound => do
          let evaluation ← reducedDirectMatrixFactAt arena parameters maps source fuel
          pure { entries := evaluation.entries.map fun entry =>
            { entry with fact := { entry.fact with totalHardBound } }, rewriteEvents := evaluation.rewriteEvents }
      | .pointwise (.matrix matrixType) (.matrix operation) inputs => do
          let inputEvaluations ← inputs.toList.mapM fun input =>
            reducedDirectMatrixFactAt arena parameters maps input fuel
          let inputEntries := inputEvaluations.map (·.entries)
          let rec zipEntries : List (List ReducedDirectMatrixFact) →
              Except OperationalError (List (Array OperationalMatrixFact × Option DirectCorrelationKey × Nat))
            | [] => throw (.invalidCount operation.ownerNode 0)
            | inputs => do
                if inputs.any List.isEmpty then throw (.invalidCount operation.ownerNode 0)
                let driver? := inputs.find? fun entries => entries.length > 1
                match driver? with
                | some driver => driver.mapM fun driverEntry => do
                    let arguments ← inputs.mapM fun entries => match entries with
                      | [entry] =>
                          if entry.key.isNone ||
                              (entry.key == driverEntry.key && entry.ordinal == driverEntry.ordinal) then
                            pure entry.fact
                          else throw (.unsupportedOperationalExpr id)
                      | _ => match entries.find? fun entry =>
                          entry.key == driverEntry.key && entry.ordinal == driverEntry.ordinal with
                        | some entry => pure entry.fact
                        | none => throw (.unsupportedOperationalExpr id)
                    pure (arguments.toArray, driverEntry.key, driverEntry.ordinal)
                | none => do
                    let entries ← inputs.mapM fun entries => match entries with
                      | [entry] => pure entry
                      | _ => throw (.unsupportedOperationalExpr id)
                    let driver? := entries.find? fun entry => entry.key.isSome
                    match driver? with
                    | some driver =>
                        if entries.all fun entry => entry.key.isNone ||
                            (entry.key == driver.key && entry.ordinal == driver.ordinal) then
                          pure [(entries.map (·.fact) |>.toArray, driver.key, driver.ordinal)]
                        else throw (.unsupportedOperationalExpr id)
                    | none => pure [(entries.map (·.fact) |>.toArray, none, 0)]
          let aligned ← zipEntries inputEntries
          let entriesAndEvents ← aligned.mapM fun (arguments, key, ordinal) => do
            let (fact, rewriteCount) ←
              applyDirectMatrixPointwiseOperationWithRelationRewriteCount operation matrixType arguments
            let pointwise : DirectRelationRewritePointwiseKey := {
              ownerScope := operation.ownerScope, ownerNode := operation.ownerNode,
              outputPort := operation.outputPort }
            let events := (List.range rewriteCount).map fun localOrdinal => {
              pointwise := pointwise,
              maps := normalizeDirectRelationRewriteMaps maps, key := key,
              ordinal := ordinal,
              localOrdinal := localOrdinal }
            pure ({ key, ordinal, fact }, events)
          let entries := entriesAndEvents.map (fun value => value.1)
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (inputEvaluations.flatMap (fun value => value.rewriteEvents) ++
              entriesAndEvents.flatMap (fun value => value.2))
          pure { entries := entries, rewriteEvents := rewriteEvents }
      | .pointwise (.matrix matrixType) (.relation operation) inputs => do
          let inputEvaluations ← inputs.toList.mapM fun inputId => do
            let input ← match arena.valueAt? inputId with
              | some value => pure value
              | none => throw (.invalidOperationalExprRef inputId)
            match input.payload.schema with
            | .matrix _ =>
                let evaluation ← reducedDirectMatrixFactAt arena parameters maps inputId fuel
                pure (evaluation.entries.map fun entry =>
                  ReducedDirectRelationArgument.mk entry.key entry.ordinal
                    (DirectRelationArgument.matrix entry.fact), evaluation.rewriteEvents)
            | .scalar (.trapdoor _) =>
                let evaluation ← reducedDirectScalarFactAt arena parameters maps inputId fuel
                let entries ← evaluation.entries.mapM fun entry => do
                  match entry.fact with
                  | OperationalScalarFact.trapdoor fact =>
                      pure (ReducedDirectRelationArgument.mk entry.key entry.ordinal
                        (DirectRelationArgument.trapdoor fact))
                  | _ => throw (.unsupportedOperationalExpr id)
                pure (entries, evaluation.rewriteEvents)
            | .scalar _ => throw (.unsupportedOperationalExpr id)
          let inputEntries := inputEvaluations.map (·.1)
          let aligned ← alignDirectRelationArguments operation.ownerNode id inputEntries
          let entries ← aligned.mapM fun (arguments, key, ordinal) => do
            let fact ← applyDirectRelationProducer operation matrixType arguments
            pure { key, ordinal, fact }
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (inputEvaluations.flatMap (fun value => value.2))
          pure { entries := entries, rewriteEvents := rewriteEvents }
      | .pointwise (.matrix matrixType) (.matrixFromScalar operation) inputs => do
          let input ← match inputs with
            | #[input] => pure input
            | _ => throw (.unsupportedOutputArity operation.ownerNode inputs.size)
          let evaluation ← reducedDirectScalarFactAt arena parameters maps input fuel
          let entries ← evaluation.entries.mapM fun entry => do
            let fact ← applyDirectMatrixFromScalarOperation operation matrixType entry.fact
            pure { key := entry.key, ordinal := entry.ordinal, fact }
          pure { entries, rewriteEvents := evaluation.rewriteEvents }
      | _ => throw (.unsupportedOperationalExpr id)

private def reducedDirectScalarFactAt
    (arena : DirectOperationalIndexedArena)
    (parameters : ParamEnvironment)
    (maps : List IndexMap)
    (id : OperationalIndexedValueId) : Nat → Except OperationalError ReducedDirectScalarEvaluation
  | 0 => throw (.unsupportedOperationalExpr id)
  | fuel + 1 => do
      let value ← match arena.valueAt? id with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef id)
      match value.payload with
      | .shared (.scalar _) (.scalar reference) => do
          let fact ← match arena.fixed.scalars[reference]? with
            | some fact => pure fact
            | none => throw (.invalidOperationalExprRef reference)
          pure { entries := [{ key := none, ordinal := 0, fact }] }
      | .explicit (.scalar _) binder references => do
          let mapped ← references.toList.mapIdxM fun ordinal reference => do
            let fact ← match reference with
              | .scalar reference => match arena.fixed.scalars[reference]? with
                | some fact => pure fact
                | none => throw (.invalidOperationalExprRef reference)
              | .matrix _ => throw (.unsupportedOperationalExpr id)
            let entry := if references.size == 1 then (none, 0) else (some (.variable binder), ordinal)
            match ← transportDirectCorrelation parameters id maps entry.1 entry.2 with
            | some (key, ordinal) => pure (some { key, ordinal, fact })
            | none => pure none
          let entries := mapped.filterMap fun entry => entry
          pure { entries := entries }
      | .explicitValues (.scalar _) binder values => do
          let lanes ← values.toList.mapIdxM fun ordinal child => do
            let outer := if values.size == 1 then (none, 0) else (some (.variable binder), ordinal)
            let outer ← transportDirectCorrelation parameters id maps outer.1 outer.2
            match outer with
            | none => pure ({ entries := [] } : ReducedDirectScalarEvaluation)
            | some (none, _) =>
                let evaluation ← reducedDirectScalarFactAt arena parameters maps child fuel
                return evaluation
            | some (some outerKey, outerOrdinal) => do
                let evaluation ← reducedDirectScalarFactAt arena parameters maps child fuel
                let entries := evaluation.entries
                match entries with
                | [entry] => match entry.key with
                    | none =>
                        let entry := { entry with key := some outerKey, ordinal := outerOrdinal }
                        pure { entries := [entry], rewriteEvents := evaluation.rewriteEvents }
                    | some _ =>
                        if entry.key == some outerKey && entry.ordinal == outerOrdinal then
                          pure evaluation
                        else throw (.unsupportedOperationalExpr id)
                | _ =>
                    match entries.filter fun entry =>
                      entry.key == some outerKey && entry.ordinal == outerOrdinal with
                    | [entry] => pure { entries := [entry], rewriteEvents := evaluation.rewriteEvents }
                    | _ => throw (.unsupportedOperationalExpr id)
          let entries := lanes.flatMap (fun lane => lane.entries)
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (lanes.flatMap (fun lane => lane.rewriteEvents))
          pure { entries := entries, rewriteEvents := rewriteEvents }
      | .mapped (.scalar _) source map => do
          if !map.transportValid || map.destination != value.context then
            throw (.unsupportedOperationalExpr id)
          reducedDirectScalarFactAt arena parameters (map :: maps) source fuel
      | .pointwise (.scalar _) (.matrixToScalar operation) inputs => do
          let input ← match inputs with
            | #[input] => pure input
            | _ => throw (.unsupportedOutputArity operation.ownerNode inputs.size)
          let evaluation ← reducedDirectMatrixFactAt arena parameters maps input fuel
          let entries ← evaluation.entries.mapM fun entry => do
            let fact ← applyDirectMatrixToScalarOperation operation entry.fact
            pure { key := entry.key, ordinal := entry.ordinal, fact }
          pure { entries, rewriteEvents := evaluation.rewriteEvents }
      | .pointwise (.scalar _) (.scalar kind) inputs => do
          let inputEvaluations ← inputs.toList.mapM fun input =>
            reducedDirectScalarFactAt arena parameters maps input fuel
          let inputEntries := inputEvaluations.map (·.entries)
          let rec zipEntries : List (List ReducedDirectScalarFact) →
              Except OperationalError (List (Array OperationalScalarFact × Option DirectCorrelationKey × Nat))
            | [] => throw (.unsupportedOperationalExpr id)
            | inputs => do
                if inputs.any List.isEmpty then throw (.unsupportedOperationalExpr id)
                let driver? := inputs.find? fun entries => entries.length > 1
                match driver? with
                | some driver => driver.mapM fun driverEntry => do
                    let arguments ← inputs.mapM fun entries => match entries with
                      | [entry] =>
                          if entry.key.isNone ||
                              (entry.key == driverEntry.key && entry.ordinal == driverEntry.ordinal) then
                            pure entry.fact
                          else throw (.unsupportedOperationalExpr id)
                      | _ => match entries.find? fun entry =>
                          entry.key == driverEntry.key && entry.ordinal == driverEntry.ordinal with
                        | some entry => pure entry.fact
                        | none => throw (.unsupportedOperationalExpr id)
                    pure (arguments.toArray, driverEntry.key, driverEntry.ordinal)
                | none => do
                    let entries ← inputs.mapM fun entries => match entries with
                      | [entry] => pure entry
                      | _ => throw (.unsupportedOperationalExpr id)
                    let driver? := entries.find? fun entry => entry.key.isSome
                    match driver? with
                    | some driver =>
                        if entries.all fun entry => entry.key.isNone ||
                            (entry.key == driver.key && entry.ordinal == driver.ordinal) then
                          pure [(entries.map (·.fact) |>.toArray, driver.key, driver.ordinal)]
                        else throw (.unsupportedOperationalExpr id)
                    | none => pure [(entries.map (·.fact) |>.toArray, none, 0)]
          let aligned ← zipEntries inputEntries
          let entries ← aligned.mapM fun (arguments, key, ordinal) => do
            let fact ← applyDirectScalarPointwiseOperation kind arguments
            pure { key, ordinal, fact }
          let rewriteEvents := deduplicateDirectRelationRewriteEvents
            (inputEvaluations.flatMap (fun value => value.rewriteEvents))
          pure { entries := entries, rewriteEvents := rewriteEvents }
      | _ => throw (.unsupportedOperationalExpr id)

end

/-- Reduce storage directly to concrete physical lanes while preserving only proven shared
correlation.  Unlike `directValueFactsAt`, this never enumerates a root `IndexValueEnvironment`.
Independent driver keys are rejected before any matrix operation can form a Cartesian product. -/
def OperationalExprArena.reducedDirectValueFactsAt
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) : Except OperationalError (List ReducedDirectMatrixFact) := do
  let root ← match expression.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  return (← reducedDirectMatrixFactAt arena.direct environment [] root
    (arena.direct.values.size + 1)).entries

/-- Structural direct reduction plus the deduplicated exact relation applications it performed.
This is reporting-only: acceptance consumes the public fact projection above and fixed-assignment
queries never invoke it. -/
def OperationalExprArena.reducedDirectValueFactsAtWithRelationRewriteEvents
    (arena : OperationalExprArena)
    (environment : ParamEnvironment)
    (expression : IndexedOperationalFact) :
    Except OperationalError (List ReducedDirectMatrixFact × List DirectRelationRewriteEventKey) := do
  let root ← match expression.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  let evaluation ← reducedDirectMatrixFactAt arena.direct environment [] root
    (arena.direct.values.size + 1)
  pure (evaluation.entries, deduplicateDirectRelationRewriteEvents evaluation.rewriteEvents)

/-- Sequential recurrences consume a direct carrier through its fixed assignments.  This keeps
the recurrence's numeric state independent of storage while rejecting a non-uniform carried
schema instead of summarizing an arbitrary representative. -/
def sequentialFactNumericExpressions
    (arena : OperationalExprArena)
    (slot : Nat)
    (fact : OperationalFact) : Except OperationalError
      (List (OperationalBoundPath × OperationalBoundExpr)) :=
  match fact with
  | expression@{ payload := .directValue _, .. } => do
      let entries ← arena.reducedDirectValueFactsAt [] expression
      let maximum ← match entries with
        | [] => throw (.invalidCount slot 0)
        | first :: remaining => pure <| remaining.foldl (fun bound entry =>
            .maximum bound entry.fact.totalHardBound) first.fact.totalHardBound
      pure [(.matrixMaximum 0 slot, maximum)]
  | _ => factNumericExpressions arena slot fact

def sameSequentialCarriedSchema
    (arena : OperationalExprArena)
    (left right : OperationalFact) : Bool :=
  match left, right with
  | left@{ payload := .directValue _, .. }, right@{ payload := .directValue _, .. } =>
      left.context == right.context &&
      match arena.reducedDirectValueFactsAt [] left, arena.reducedDirectValueFactsAt [] right with
      | .ok leftFacts, .ok rightFacts => leftFacts.length == rightFacts.length &&
          (leftFacts.zip rightFacts).all fun (left, right) =>
            left.key == right.key && left.ordinal == right.ordinal &&
              sameCarriedMatrixFactSchema left.fact right.fact
      | _, _ => false
  | _, _ => sameCarriedSchema arena left right

def sequentialCarriedLargeFactorCounts
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (List Nat) :=
  match fact with
  | expression@{ payload := .directValue _, .. } => do
      let entries ← arena.reducedDirectValueFactsAt [] expression
      pure <| entries.flatMap fun entry => entry.fact.polynomial.map operationalLargeFactorCount
  | _ => pure (carriedLargeFactorCounts arena fact)


/-- Group the already-derived exact signal part of a BGG encoding while retaining its bounded
noise as a separate top-level term.  The complete pre-grouping signal polynomial is embedded in a
flat token sequence, so this cannot create a false cancellation or hide bounded noise.  The paired
public-key/plaintext origins identify the one executable BGG value selected at runtime. -/
def groupExactSignal
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

def groupBggEncodingSignal
    (vector publicKey plaintext : OperationalMatrixFact) :
    Except OperationalFlatError OperationalMatrixFact :=
  groupExactSignal
    [.primitive (.matrix publicKey.origin), .primitive (.matrix plaintext.origin)] vector

def groupPublicKeySignal
    (fact : OperationalMatrixFact) : Except OperationalError OperationalMatrixFact :=
  groupExactSignal [] fact |>.mapError (.flat 0)

/-- Promote the output of a separately validated exact Boolean carrier selection to one Large
signal factor. The validator below proves that this value is exactly `select(bit, zero, carrier)`
with a deterministic constant carrier, so this grouping cannot hide sampler noise. -/
def groupProtocolBooleanSignal
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
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
  { fact with polynomial := [term], metadata := {} }

def groupBggEncodingExprIds
    (environment : ParamEnvironment)
    (deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (vector publicKey plaintext : OperationalExprId)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × OperationalExprId) := do
  let vectorType ← arena.checkedType vector #[]
  let operation : PrimitiveOperation := {
    kind := .bggGrouping
    outputType := vectorType
    ownerScope := arena.activeScope
    ownerNode := (arena.get? vector).bind (·.ownerNode) |>.getD 0
    outputPort := 0
    parameterEnvironment := environment
  }
  let concreteTransfer (arguments : Array OperationalMatrixFact) := do
    if arguments.size != 3 then
      throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    let vector ← match arguments[0]? with
      | some value => pure value
      | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    let publicKey ← match arguments[1]? with
      | some value => pure value
      | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    let plaintext ← match arguments[2]? with
      | some value => pure value
      | none => throw (.unsupportedOutputArity operation.ownerNode arguments.size)
    groupBggEncodingSignal vector publicKey plaintext |>.mapError (.flat operation.ownerNode)
  liftPrimitiveOperation operation .bggGrouping concreteTransfer deriveSchema arena
    #[vector, publicKey, plaintext] fuel

def groupBggEncodingOperationalFacts
    (environment : ParamEnvironment)
    (deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena) :
    OperationalFact → OperationalFact → OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | vector, publicKey, plaintext => do
      let (arena, grouped) ← liftIndexedOperationalFacts arena #[vector, publicKey, plaintext]
        fun arena roots => do
          let vector ← match roots[0]? with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr arena.nodes.size)
          let publicKey ← match roots[1]? with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr arena.nodes.size)
          let plaintext ← match roots[2]? with
            | some value => pure value
            | none => throw (.unsupportedOperationalExpr arena.nodes.size)
          groupBggEncodingExprIds environment deriveSchema arena vector publicKey plaintext
            (arena.nodes.size + 1)
      let arena ← arena.rememberIndexedExpr grouped
      pure (arena, grouped)

def derivationAttachmentRole
    (attachment : DerivationAttachment)
    (role : String) : Except OperationalError WireRef :=
  match attachment.roles.filter (fun candidate => candidate.1 == role) with
  | [(_, wire)] => pure wire
  | _ => throw (.missingDerivationAttachmentRole attachment.ownerNamespace attachment.ruleName role)

def validateDerivationAttachment
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

def replaceOperationalFact
    (node : Nat)
    (facts : OperationalScopeFacts)
    (wire : WireRef)
    (fact : OperationalFact) : Except OperationalError OperationalScopeFacts := do
  let outputs ← match facts.values[wire.node]? with
    | some outputs => pure outputs
    | none => throw (.missingOperand node wire)
  if wire.port >= outputs.size then throw (.missingOperand node wire)
  pure { facts with values := facts.values.set! wire.node (outputs.set! wire.port fact) }

def applyDerivationAttachment
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
    let (arena, grouped) ← facts.arena.pushDirectBggGrouping vector publicKey plaintext
    replaceOperationalFact node { facts with arena } vectorWire grouped
  else if attachment.ownerNamespace == "mxx-correctness" &&
      attachment.ruleName == "protocol-boolean-signal-grouping" then
    let valueWire ← derivationAttachmentRole attachment "value"
    let value ← lookupFact node facts valueWire
    match value with
    | expression@{ payload := .matrix _, .. } =>
        let mapFact (fact : OperationalMatrixFact) := pure (groupProtocolBooleanSignal fact)
        let cacheNamespace := s!"protocol-boolean-group:{node}:{valueWire.node}:{valueWire.port}"
        let (arena, root) ← mapOperationalExprM cacheNamespace .bggGrouping facts.arena
          expression.payload mapFact
        let grouped : IndexedOperationalFact := { expression with payload := .matrix root }
        let arena ← arena.rememberIndexedExpr grouped
        replaceOperationalFact node { facts with arena } valueWire grouped
    | { payload := .scalar _, .. } =>
        throw (.operandNotMatrix node valueWire)
    | expression@{ payload := .directValue root, .. } => do
        let value ← match facts.arena.direct.valueAt? root with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef root)
        if value.context != expression.context then throw (.unsupportedOperationalExpr root)
        let (direct, grouped) ← facts.arena.direct.mapMatrixValue root
          (fun fact => pure (groupProtocolBooleanSignal fact))
        let value ← match direct.valueAt? grouped with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef grouped)
        replaceOperationalFact node { facts with arena := { facts.arena with direct } } valueWire {
          context := value.context, payload := .directValue grouped, storage := value.storage }
  else
    let valueWire ← derivationAttachmentRole attachment "value"
    let value ← lookupFact node facts valueWire
    match value with
    | expression@{ payload := .matrix _, .. } =>
        let mapFact (fact : OperationalMatrixFact) := groupPublicKeySignal fact
        let cacheNamespace := s!"public-key-group:{node}:{valueWire.node}:{valueWire.port}"
        let (arena, root) ← mapOperationalExprM cacheNamespace .bggGrouping facts.arena
          expression.payload mapFact
        let grouped : IndexedOperationalFact := { expression with payload := .matrix root }
        let arena ← arena.rememberIndexedExpr grouped
        replaceOperationalFact node { facts with arena } valueWire grouped
    | { payload := .scalar _, .. } =>
        throw (.operandNotMatrix node valueWire)
    | expression@{ payload := .directValue root, .. } => do
        let value ← match facts.arena.direct.valueAt? root with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef root)
        if value.context != expression.context then throw (.unsupportedOperationalExpr root)
        let (direct, grouped) ← facts.arena.direct.mapMatrixValue root groupPublicKeySignal
        let value ← match direct.valueAt? grouped with
          | some value => pure value
          | none => throw (.invalidOperationalExprRef grouped)
        replaceOperationalFact node { facts with arena := { facts.arena with direct } } valueWire {
          context := value.context, payload := .directValue grouped, storage := value.storage }

def applyPreparedDerivationAttachments
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

def publicIdentityIsLarge : PublicMatrixIdentity → Bool
  | .sampledTrapdoor .. | .gadget .. => true
  | .indexed _ _ source => publicIdentityIsLarge source
  | .loopInstance _ _ source => publicIdentityIsLarge source

def publicIdentityMaximum
    (residueCap : Int) : PublicMatrixIdentity → Int
  | .sampledTrapdoor .. => residueCap
  | .gadget .. => residueCap
  | .indexed _ _ source => publicIdentityMaximum residueCap source
  | .loopInstance _ _ source => publicIdentityMaximum residueCap source

def rebindOperationalScalarFact
    (subject : WireRef) : OperationalScalarFact → OperationalScalarFact
  | .integer fact => .integer { fact with subject }
  | .trapdoor fact => .trapdoor { fact with subject }
  | .bytes fact => .bytes { fact with subject }
  | .boolean => .boolean
  | .real => .real
  | .typedBlob typeName => .typedBlob typeName
  | .unknown wireType => .unknown wireType

def rebindMatrixSubject
    (subject : WireRef) (fact : OperationalMatrixFact) :
    Except OperationalError OperationalMatrixFact :=
  if fact.relations.all fun relation => match relation with
      | .decomposition relation => relation.producer == fact.origin
      | .preimage relation => relation.producer == fact.origin then
    pure { fact with subject }
  else throw (.malformedRelation subject.node)

/-- Arena-aware rebinding for the indexed matrix representation.  It rewrites every concrete
leaf into a fresh expression DAG and retains the existing indexed context on the new root; no
legacy fact-level fallback is involved. -/
def rebindIndexedOperationalFact
    (subject : WireRef)
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  let mapFact := rebindMatrixSubject subject
  let cacheNamespace := s!"indexed-rebind:{subject.node}:{subject.port}:{expression.payload.root}"
  let (arena, root) ← mapOperationalExprM cacheNamespace .instantiationMap arena
    expression.payload mapFact id
  let rebound : IndexedOperationalFact := { expression with payload := .matrix root }
  let arena ← arena.rememberIndexedExpr rebound
  pure (arena, rebound)

/-- Rebind every atom in an indexed scalar DAG.  Selection nodes and their index context are
preserved exactly; a dynamic family is never collapsed to its shared representative. -/
def rebindIndexedScalarFact
    (subject : WireRef)
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  let rec visit : Nat → OperationalExprArena → Nat →
      Except OperationalError (OperationalExprArena × Nat)
    | 0, _, root => throw (.unsupportedOperationalExpr root)
    | fuel + 1, arena, root => do
        match arena.scalarNodes[root]? with
        | none => throw (.invalidOperationalExprRef root)
        | some (.concrete fact) =>
            pure (arena.pushScalarConcrete (rebindOperationalScalarFact subject fact))
        | some (.primitive kind arguments result) => do
            let (arena, arguments) ← arguments.foldlM (fun (arena, rebound) argument => do
              let (arena, argument) ← visit fuel arena argument
              pure (arena, rebound.push argument)) (arena, #[])
            pure (arena.pushScalar
              (.primitive kind arguments (rebindOperationalScalarFact subject result)))
        | some (.selectExact domain branches) => do
            let (arena, branches) ← branches.foldlM (fun (arena, rebound) branch => do
              let (arena, branch) ← visit fuel arena branch
              pure (arena, rebound.push branch)) (arena, #[])
            pure (arena.pushScalar (.selectExact domain branches))
        | some (.selectShared domain binder _ representative) => do
            let (arena, representative) ← visit fuel arena representative
            pure (arena.pushScalar (.selectShared domain binder subject representative))
  let (arena, root) ← visit (arena.scalarNodes.size + 1) arena expression.payload
  let rebound : IndexedOperationalFact := { expression with payload := .scalar root }
  let arena ← arena.rememberIndexedScalar rebound
  pure (arena, rebound)

/-- Rebind a wire-level fact without dropping indexed matrix metadata.  This is the replacement
entry point for the legacy fact-only rebinding helper; family containers recurse structurally while
indexed matrices are transported through the arena. -/
partial def rebindOperationalFact
    (subject : WireRef) : OperationalExprArena → OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | arena, expression@{ payload := .matrix _, .. } => do
      let (arena, rebound) ← rebindIndexedOperationalFact subject arena expression
      pure (arena, rebound)
  | arena, expression@{ payload := .scalar _, .. } => do
      let (arena, rebound) ← rebindIndexedScalarFact subject arena expression
      pure (arena, rebound)
  | arena, expression@{ payload := .directValue root, .. } => do
      let value ← match arena.direct.valueAt? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      if value.context != expression.context then throw (.unsupportedOperationalExpr root)
      let (direct, rebound) ← match value.payload.schema with
        | .matrix _ => arena.direct.mapMatrixValue root (rebindMatrixSubject subject)
        | .scalar _ => arena.direct.mapScalarValue root (pure ∘ rebindOperationalScalarFact subject)
      let value ← match direct.valueAt? rebound with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef rebound)
      pure ({ arena with direct }, {
        context := value.context
        payload := .directValue rebound
        storage := value.storage
      })

def canonicalSelectionExpression
    (origin : OperationalValueOrigin) (count : IntExpr) : IndexExpr :=
  .variable {
    owner := {
      stage := ⟨s!"operational-selection:{reprStr origin}"⟩
      scope := ⟨[]⟩
      node := ⟨0⟩
    }
    slot := 0
    count
  }

def selectionExpressionForOrigin
    (origin : OperationalValueOrigin) : IndexExpr → IndexExpr
  | .constant value => .constant value
  | .variable binder => canonicalSelectionExpression origin binder.count
  | .offset base amount => .offset (selectionExpressionForOrigin origin base) amount
  | .gather owner sourceCount position =>
      .gather owner sourceCount (selectionExpressionForOrigin origin position)

def DynamicSelectionIdentity.withOrigin
    (selection : DynamicSelectionIdentity)
    (origin : OperationalValueOrigin) : DynamicSelectionIdentity := {
  index := origin
  expression := selectionExpressionForOrigin origin selection.expression
}

partial def namespaceFreshValueOrigin
    (scope : ScopeTemplateKey)
    (wire : WireRef) : OperationalValueOrigin → OperationalValueOrigin
  | .local originScope originWire =>
      if originScope == temporaryScope && originWire == wire then .local scope originWire
      else .local originScope originWire
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | .loopInstance slot index source =>
      .loopInstance slot index (namespaceFreshValueOrigin scope wire source)
  | .indexed binder expression source =>
      .indexed binder expression (namespaceFreshValueOrigin scope wire source)

def namespaceFreshOrigin
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
  | .indexed binder expression source =>
      .indexed binder expression (namespaceFreshOrigin scope wire source)

def namespaceFreshPublicIdentity
    (scope : ScopeTemplateKey)
    (wire : WireRef) : PublicMatrixIdentity → PublicMatrixIdentity
  | .sampledTrapdoor originScope originWire =>
      if originScope == temporaryScope && originWire.node == wire.node then
        .sampledTrapdoor scope originWire
      else .sampledTrapdoor originScope originWire
  | identity@(.gadget ..) => identity
  | .indexed binder expression source =>
      .indexed binder expression (namespaceFreshPublicIdentity scope wire source)
  | .loopInstance slot index source =>
      .loopInstance slot index (namespaceFreshPublicIdentity scope wire source)

def mapOperationalPrimitiveIdentity
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

def reindexOperationalValueOrigin
    (map : IndexMap) : OperationalValueOrigin → Option OperationalValueOrigin
  | .local scope wire => some (.local scope wire)
  | .protocolInput input => some (.protocolInput input)
  | .protocolFamilyElement input index =>
      return .protocolFamilyElement input (← reindex map index)
  | .loopInstance slot index source =>
      return .loopInstance slot (← reindex map index) (← reindexOperationalValueOrigin map source)
  | .indexed binder expression source =>
      return .indexed binder (← reindex map expression)
        (← reindexOperationalValueOrigin map source)

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
not representable by legacy `IntExpr` and therefore remains fail-closed until bounds use
`IndexExpr` directly. -/
def reindexIntExpr (map : IndexMap) : IntExpr → Option IntExpr
  | .constant value => some (.constant value)
  | .parameter name => some (.parameter name)
  | .loopIndex slot => do
      let binder ← indexMapSourceBinderForSlot map slot
      indexExprAsIntExpr (← reindex map (.variable binder))
  | .add left right => return .add (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .subtract left right => return .subtract (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .multiply left right => return .multiply (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .divide left right => return .divide (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .roundDivide left right =>
      return .roundDivide (← reindexIntExpr map left) (← reindexIntExpr map right)
  | .log2Ceil value => return .log2Ceil (← reindexIntExpr map value)

def reindexMatrixTypeExpr (map : IndexMap)
    (matrixType : MatrixTypeExpr) : Option MatrixTypeExpr := do
  pure {
    modulus := ← reindexIntExpr map matrixType.modulus
    ringDimension := ← reindexIntExpr map matrixType.ringDimension
    rows := ← reindexIntExpr map matrixType.rows
    columns := ← reindexIntExpr map matrixType.columns
  }

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
      | [destination] => pure ((.loopIndex destination.slot, value) :: tail)
      | _ => none

def reindexParameterDomains
    (map : IndexMap) : List OperationalParameterDomain → Option (List OperationalParameterDomain)
  | [] => some []
  | domain :: remaining => do
      let tail ← reindexParameterDomains map remaining
      match domain with
      | .parameter name environment domains expression =>
          pure (.parameter name (← reindexParamEnvironment map environment)
            (← reindexParameterDomains map domains) (← reindexIntExpr map expression) :: tail)
      | .loopIndex slot _ =>
          let binder ← indexMapSourceBinderForSlot map slot
          let mapped ← reindex map (.variable binder)
          match mapped.freeVariables with
          | [] => pure tail
          | [destination] => match destination.count with
              | .constant count => pure (.loopIndex destination.slot count.toNat :: tail)
              | _ => none
          | _ => none

def reindexOperationalBoundExpr (map : IndexMap) : OperationalBoundExpr → Option OperationalBoundExpr
  | .closedInt value => return .closedInt (← reindexIntExpr map value)
  | .contextual kind environment domains value =>
      return .contextual kind (← reindexParamEnvironment map environment)
        (← reindexParameterDomains map domains) (← reindexIntExpr map value)
  | .previous path => some (.previous path)
  | .negate value => return .negate (← reindexOperationalBoundExpr map value)
  | .add left right =>
      return .add (← reindexOperationalBoundExpr map left) (← reindexOperationalBoundExpr map right)
  | .subtract left right =>
      return .subtract (← reindexOperationalBoundExpr map left)
        (← reindexOperationalBoundExpr map right)
  | .multiply left right =>
      return .multiply (← reindexOperationalBoundExpr map left)
        (← reindexOperationalBoundExpr map right)
  | .divide left right =>
      return .divide (← reindexOperationalBoundExpr map left)
        (← reindexOperationalBoundExpr map right)
  | .minimum left right =>
      return .minimum (← reindexOperationalBoundExpr map left)
        (← reindexOperationalBoundExpr map right)
  | .maximum left right =>
      return .maximum (← reindexOperationalBoundExpr map left)
        (← reindexOperationalBoundExpr map right)
  | .centeredCap modulus value =>
      return .centeredCap (← reindexOperationalBoundExpr map modulus)
        (← reindexOperationalBoundExpr map value)
  | .matrixProduct ringDimension innerDimension left right =>
      return .matrixProduct (← reindexOperationalBoundExpr map ringDimension)
        (← reindexOperationalBoundExpr map innerDimension) (← reindexOperationalBoundExpr map left)
        (← reindexOperationalBoundExpr map right)
  | .recurrence count initial transition slot =>
      return .recurrence count (← initial.mapM (reindexOperationalBoundExpr map))
        (← transition.mapM (reindexOperationalBoundExpr map)) slot
  | .recurrenceState count paths initial transition output =>
      return .recurrenceState count paths (← initial.mapM (reindexOperationalBoundExpr map))
        (← transition.mapM (reindexOperationalBoundExpr map)) output

def reindexMatrixOriginIdentity
    (map : IndexMap) : MatrixOriginIdentity → Option MatrixOriginIdentity
  | .value scope wire => some (.value scope wire)
  | .protocolInput input => some (.protocolInput input)
  | .protocolFamilyElement input index =>
      return .protocolFamilyElement input (← reindex map index)
  | .deterministicHash query =>
      return .deterministicHash {
        query with
        keyOrigin := ← reindexOperationalValueOrigin map query.keyOrigin
        matrixType := ← reindexMatrixTypeExpr map query.matrixType
        parameterEnvironment := ← reindexParamEnvironment map query.parameterEnvironment
        parameterDomains := ← reindexParameterDomains map query.parameterDomains
        tagExpressions := ← query.tagExpressions.mapM (reindexIntExpr map)
        tagDecimalExpressions := ← query.tagDecimalExpressions.mapM (reindexIntExpr map)
        tagU64LeExpressions := ← query.tagU64LeExpressions.mapM (reindexIntExpr map)
        trailingIntegerOrigins := ← query.trailingIntegerOrigins.mapM
          (reindexOperationalValueOrigin map)
      }
  | .loopInstance slot index source =>
      return .loopInstance slot (← reindex map index) (← reindexMatrixOriginIdentity map source)
  | .indexed binder expression source =>
      return .indexed binder (← reindex map expression)
        (← reindexMatrixOriginIdentity map source)

def reindexPublicMatrixIdentity
    (map : IndexMap) : PublicMatrixIdentity → Option PublicMatrixIdentity
  | .sampledTrapdoor scope wire => some (.sampledTrapdoor scope wire)
  | .gadget paramsId params inputRows base small digitCount =>
      some (.gadget paramsId params inputRows base small digitCount)
  | .indexed binder expression source =>
      return .indexed binder (← reindex map expression)
        (← reindexPublicMatrixIdentity map source)
  | .loopInstance slot index source =>
      return .loopInstance slot (← reindex map index) (← reindexPublicMatrixIdentity map source)

def reindexOperationalPrimitiveIdentityFully
    (map : IndexMap) : OperationalPrimitiveIdentity → Option OperationalPrimitiveIdentity
  | .matrix identity => return .matrix (← reindexMatrixOriginIdentity map identity)
  | .publicMatrix identity => return .publicMatrix (← reindexPublicMatrixIdentity map identity)
  | .value identity => return .value (← reindexOperationalValueOrigin map identity)
  | .parameterScalar environment domains value =>
      return .parameterScalar (← reindexParamEnvironment map environment)
        (← reindexParameterDomains map domains) (← reindexIntExpr map value)
  | .identityMatrix type => return .identityMatrix (← reindexMatrixTypeExpr map type)
  | .indexedArtifact input index => return .indexedArtifact input (← reindex map index)
  | .recurrenceResult scope node path => some (.recurrenceResult scope node path)
  | .carriedInput path => some (.carriedInput path)

def reindexOperationalCompressionToken
    (map : IndexMap) : OperationalCompressionToken → Option OperationalCompressionToken
  | .primitive identity => return .primitive (← reindexOperationalPrimitiveIdentityFully map identity)
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
  | .summaryBound bound => return .summaryBound (← reindexOperationalBoundExpr map bound)
  | .summaryMetadata metadata => some (.summaryMetadata metadata)
  | .segmentStart kind length => some (.segmentStart kind length)
  | .segmentEnd => some .segmentEnd

def reindexOperationalBoundedSummary
    (map : IndexMap)
    (summary : OperationalBoundedFactorSummary) : Option OperationalBoundedFactorSummary := do
  pure { summary with
    matrixType := ← reindexMatrixTypeExpr map summary.matrixType
    hardBound := ← reindexOperationalBoundExpr map summary.hardBound
    provenance := ← summary.provenance.mapM (reindexOperationalCompressionToken map)
  }

def reindexRelationSnapshotPolynomial
    (map : IndexMap)
    (polynomial : RelationSnapshotPolynomial) : Option RelationSnapshotPolynomial :=
  polynomial.mapM fun term => do
    let factors ← term.product.factors.mapM fun factor => do
      let leaf : RelationSnapshotFactorLeaf ← match factor.leaf with
        | .primitive identity => pure (.primitive (← reindexOperationalPrimitiveIdentityFully map identity))
        | .boundedSummary origin summary =>
            pure (.boundedSummary
              { origin with tokens := ← origin.tokens.mapM (reindexOperationalCompressionToken map) }
              (← reindexOperationalBoundedSummary map summary))
        | .exactTransform tokens type =>
            pure (.exactTransform (← tokens.mapM (reindexOperationalCompressionToken map))
              (← reindexMatrixTypeExpr map type))
      pure { factor with
        leaf
        inputType := ← reindexMatrixTypeExpr map factor.inputType
        outputType := ← reindexMatrixTypeExpr map factor.outputType
        boundedSummary := ← factor.boundedSummary.mapM (reindexOperationalBoundedSummary map)
      }
    pure { term with product := { term.product with
      factors
      outputType := ← reindexMatrixTypeExpr map term.product.outputType
    } }

def reindexRelationTargetSummary
    (map : IndexMap)
    (summary : RelationTargetSummary) : Option RelationTargetSummary := do
  pure { summary with
    origin := ← reindexMatrixOriginIdentity map summary.origin
    matrixType := ← reindexMatrixTypeExpr map summary.matrixType
    totalHardBound := ← reindexOperationalBoundExpr map summary.totalHardBound
    polynomial := ← reindexRelationSnapshotPolynomial map summary.polynomial
  }

def reindexOperationalMatrixRelation
    (map : IndexMap) : OperationalMatrixRelation → Option OperationalMatrixRelation
  | .decomposition relation =>
      return .decomposition {
        relation with
        producer := ← reindexMatrixOriginIdentity map relation.producer
        publicIdentity := ← reindexPublicMatrixIdentity map relation.publicIdentity
        inputOrigin := ← reindexMatrixOriginIdentity map relation.inputOrigin
        inputSummary := ← reindexRelationTargetSummary map relation.inputSummary
      }
  | .preimage relation =>
      return .preimage {
        relation with
        producer := ← reindexMatrixOriginIdentity map relation.producer
        publicIdentity := ← reindexPublicMatrixIdentity map relation.publicIdentity
        targetOrigin := ← reindexMatrixOriginIdentity map relation.targetOrigin
        targetSummary := ← reindexRelationTargetSummary map relation.targetSummary
      }

def reindexOperationalPolynomial
    (map : IndexMap)
    (polynomial : OperationalPolynomial) : Option OperationalPolynomial :=
  polynomial.mapM fun term => do
    let factors ← term.product.factors.mapM fun factor => do
      let leaf : OperationalFactorLeaf ← match factor.leaf with
        | .primitive identity => pure (.primitive (← reindexOperationalPrimitiveIdentityFully map identity))
        | .boundedSummary origin summary =>
            pure (.boundedSummary
              { origin with tokens := ← origin.tokens.mapM (reindexOperationalCompressionToken map) }
              (← reindexOperationalBoundedSummary map summary))
        | .exactTransform tokens type =>
            pure (.exactTransform (← tokens.mapM (reindexOperationalCompressionToken map))
              (← reindexMatrixTypeExpr map type))
      pure { factor with
        leaf
        inputType := ← reindexMatrixTypeExpr map factor.inputType
        outputType := ← reindexMatrixTypeExpr map factor.outputType
        boundedSummary := ← factor.boundedSummary.mapM (reindexOperationalBoundedSummary map)
        relations := ← factor.relations.mapM (reindexOperationalMatrixRelation map)
      }
    pure { term with product := { term.product with
      factors
      outputType := ← reindexMatrixTypeExpr map term.product.outputType
    } }

def reindexOperationalBlockPartition
    (map : IndexMap)
    (partition : OperationalBlockPartition) : Option OperationalBlockPartition := do
  pure {
    matrixType := ← reindexMatrixTypeExpr map partition.matrixType
    polynomial := ← reindexOperationalPolynomial map partition.polynomial
  }

def reindexOperationalBlockLayout
    (map : IndexMap)
    (layout : OperationalBlockLayout) : Option OperationalBlockLayout := do
  pure { layout with partitions := ← layout.partitions.mapM (reindexOperationalBlockPartition map) }

/-- Exhaustive transport for a matrix payload.  Old selected identities return `none` above, so
callers cannot accidentally retain a pre-indexed selector in an otherwise reindexed fact. -/
def reindexOperationalMatrixFact
    (map : IndexMap)
    (fact : OperationalMatrixFact) : Option OperationalMatrixFact := do
  pure { fact with
    origin := ← reindexMatrixOriginIdentity map fact.origin
    matrixType := ← reindexMatrixTypeExpr map fact.matrixType
    totalHardBound := ← reindexOperationalBoundExpr map fact.totalHardBound
    identity := ← fact.identity.mapM (reindexPublicMatrixIdentity map)
    relations := ← fact.relations.mapM (reindexOperationalMatrixRelation map)
    polynomial := ← reindexOperationalPolynomial map fact.polynomial
    blockLayout := ← fact.blockLayout.mapM (reindexOperationalBlockLayout map)
  }

/-- Reindex one direct matrix carrier value.  The storage map retains the indexed shape and
composes without materializing a family, while every reachable fixed matrix leaf receives the
same capture-free substitution.  Thus carrier metadata and the identities evaluated from those
leaves cannot disagree after static, dynamic, offset, or gather reindexing. -/
def OperationalExprArena.reindexDirectMatrixFact
    (arena : OperationalExprArena)
    (map : IndexMap)
    (expression : IndexedOperationalFact) : Except OperationalError
      (OperationalExprArena × IndexedOperationalFact) := do
  let root ← match expression.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  if !map.transportValid || map.source != expression.context then throw (.unsupportedOperationalExpr root)
  let rootValue ← match arena.direct.valueAt? root with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef root)
  /- A map from the empty context substitutes no source binder.  Parallel-family closing has
  already introduced its destination selector in fixed metadata before it attaches this carrier
  map, so traversing that metadata with an empty-domain substitution would incorrectly reject the
  selector as foreign. -/
  let (direct, reindexed) ← if map.source.binders.isEmpty then pure (arena.direct, root) else
    match rootValue.payload.schema with
    | .matrix _ => arena.direct.mapMatrixValue root fun fact =>
        match reindexOperationalMatrixFact map fact with
        | some fact => pure fact
        | none => throw (.unsupportedOperationalExpr root)
    /- Direct scalar values carry no matrix/provenance relation fields.  Their indexed context is
    still transported by the enclosing mapped carrier; scalar semantic kernels consume that same
    context and never select a representative. -/
    | .scalar _ => pure (arena.direct, root)
  let (direct, mapped) ← match direct.pushMapped reindexed map with
    | some result => pure result
    | none => throw (.unsupportedOperationalExpr root)
  let value ← match direct.valueAt? mapped with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef mapped)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue mapped
    storage := value.storage
  })

abbrev OperationalIndexedMatrixFact := IndexedFact OperationalMatrixFact

def reindexOperationalIndexedMatrixFact
    (map : IndexMap)
    (fact : OperationalIndexedMatrixFact) : Option OperationalIndexedMatrixFact :=
  fact.reindex map reindexOperationalMatrixFact

def reindexOperationalScalarFact
    (map : IndexMap) : OperationalScalarFact → Option OperationalScalarFact
  | .integer fact => do
      pure (.integer {
        fact with
        origin := ← reindexOperationalValueOrigin map fact.origin
        lowerExpression := ← reindexOperationalBoundExpr map fact.lowerExpression
        upperExpression := ← reindexOperationalBoundExpr map fact.upperExpression })
  | .trapdoor fact => do
      let preimageCutoff ← match fact.preimageCutoff with
        | none => pure none
        | some cutoff => reindexOperationalBoundExpr map cutoff
      pure (.trapdoor {
        fact with
        matrixType := ← reindexMatrixTypeExpr map fact.matrixType
        maximum := ← reindexOperationalBoundExpr map fact.maximum
        preimageCutoff
        publicIdentity := ← reindexPublicMatrixIdentity map fact.publicIdentity })
  | .bytes fact => do
      pure (.bytes { fact with origin := ← reindexOperationalValueOrigin map fact.origin })
  | .boolean => some .boolean
  | .real => some .real
  | .typedBlob typeName => some (.typedBlob typeName)
  | .unknown wireType => some (.unknown wireType)

def reindexIndexedScalarFact
    (map : IndexMap)
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact)
    (selectionOverride : DynamicSelectionIdentity → DynamicSelectionIdentity := id) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  if expression.context != map.source || !map.transportValid then
    throw (.unsupportedOperationalExpr expression.payload)
  let rec visit : Nat → OperationalExprArena → Nat →
      Except OperationalError (OperationalExprArena × Nat)
    | 0, _, root => throw (.unsupportedOperationalExpr root)
    | fuel + 1, arena, root => do
        match arena.scalarNodes[root]? with
        | none => throw (.invalidOperationalExprRef root)
        | some (.concrete fact) =>
            let fact ← match reindexOperationalScalarFact map fact with
              | some fact => pure fact
              | none => throw (.unsupportedOperationalExpr root)
            pure (arena.pushScalarConcrete fact)
        | some (.primitive kind arguments result) => do
            let result ← match reindexOperationalScalarFact map result with
              | some result => pure result
              | none => throw (.unsupportedOperationalExpr root)
            let (arena, arguments) ← arguments.foldlM (fun (arena, mapped) argument => do
              let (arena, argument) ← visit fuel arena argument
              pure (arena, mapped.push argument)) (arena, #[])
            pure (arena.pushScalar (.primitive kind arguments result))
        | some (.selectExact domain branches) => do
            let mappedIdentity ← match reindexDynamicSelectionIdentity map domain.identity with
              | some identity => pure (selectionOverride identity)
              | none => throw (.unsupportedOperationalExpr root)
            let (arena, branches) ← branches.foldlM (fun (arena, accumulated) branch => do
              let (arena, branch) ← visit fuel arena branch
              pure (arena, accumulated.push branch)) (arena, #[])
            let (arena, mappedDomain) := arena.internSelectionDomain mappedIdentity domain.count
            pure (arena.pushScalar (.selectExact mappedDomain branches))
        | some (.selectShared domain binder subject representative) => do
            let mappedIdentity ← match reindexDynamicSelectionIdentity map domain.identity with
              | some identity => pure (selectionOverride identity)
              | none => throw (.unsupportedOperationalExpr root)
            let (arena, representative) ← visit fuel arena representative
            let (arena, mappedDomain) := arena.internSelectionDomain mappedIdentity domain.count
            pure (arena.pushScalar (.selectShared mappedDomain binder subject representative))
  let (arena, root) ← visit (arena.scalarNodes.size + 1) arena expression.payload
  let scalarExpression : IndexedFact Nat := { expression with payload := expression.payload.root }
  let reindexed ← match scalarExpression.reindex map (fun _ _ => some root) with
    | some fact => pure fact
    | none => throw (.unsupportedOperationalExpr expression.payload)
  let derived ← arena.scalarContextFor root
  if derived != reindexed.context then throw (.unsupportedOperationalExpr root)
  let result : IndexedOperationalFact := { reindexed with payload := .scalar root }
  let arena ← arena.rememberIndexedScalar result
  pure (arena, result)

def OperationalExprArena.scalarSelectionDomain
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact) :
    Except OperationalError (SelectionDomainId × (Array Nat ⊕ Nat)) := do
  match arena.scalarNodes[expression.payload.root]? with
  | some (.selectExact domain branches) =>
      if branches.size == domain.count then pure (domain, .inl branches)
      else throw (.unsupportedOperationalExpr expression.payload)
  | some (.selectShared domain _ _ representative) => pure (domain, .inr representative)
  | _ => throw (.unsupportedOperationalExpr expression.payload)

def OperationalExprArena.scalarConcrete
    (arena : OperationalExprArena) (root : Nat) : Except OperationalError OperationalScalarFact := do
  match arena.scalarNodes[root]? with
  | some (.concrete fact) => pure fact
  | some (.primitive _ _ result) => pure result
  | some _ => throw (.unsupportedOperationalExpr root)
  | none => throw (.invalidOperationalExprRef root)

def selectIndexedScalarStatic
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact)
    (requested : Nat)
    (subject : WireRef) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (domain, _) ← arena.scalarSelectionDomain expression
  if requested >= domain.count then throw (.invalidCount subject.node requested)
  let sourceBinder ← match domain.identity.expression with
    | .variable binder => pure binder
    | _ => throw (.unsupportedOperationalExpr expression.payload)
  let map ← match closedStaticIndexMap [] expression.context sourceBinder requested with
    | some map => pure map
    | none => throw (.unsupportedOperationalExpr expression.payload)
  let (arena, mapped) ← reindexIndexedScalarFact map arena expression
  let (_, mappedStorage) ← arena.scalarSelectionDomain mapped
  let root ← match mappedStorage with
    | .inl branches => match branches[requested]? with
        | some root => pure root
        | none => throw (.invalidCount subject.node requested)
    | .inr representative => pure representative
  let selected ← arena.indexedScalar root
  let (arena, rebound) ← rebindIndexedScalarFact subject arena selected
  pure (arena, rebound)

def selectIndexedScalarDynamic
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact)
    (selection : DynamicSelectionIdentity) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let (domain, _) ← arena.scalarSelectionDomain expression
  let sourceBinder ← match domain.identity.expression with
    | .variable binder => pure binder
    | _ => throw (.unsupportedOperationalExpr expression.payload)
  let map ← match dynamicIndexMap expression.context sourceBinder selection.expression with
    | some map => pure map
    | none => throw (.unsupportedOperationalExpr expression.payload)
  let (arena, mapped) ← reindexIndexedScalarFact map arena expression fun candidate =>
    if candidate.expression == selection.expression then selection else candidate
  -- Scalar atoms carry their own subjects; selection preserves alternatives in the arena, while
  -- a later static get performs the final subject rebinding.
  pure (arena, mapped)

/-- Check selection identities before applying an `IndexMap` to an arena DAG.  The existing arena
mapper accepts a total selection callback, so this preflight prevents a failed callback from
leaving one nested selector at the predecessor context. -/
partial def validateOperationalExprReindexSelections
    (map : IndexMap)
    (arena : OperationalExprArena) : OperationalExprId → Nat → Except OperationalError Unit
  | root, 0 => throw (.unsupportedOperationalExpr root)
  | root, fuel + 1 => do
      let expression ← match arena.get? root with
        | some expression => pure expression
        | none => throw (.invalidOperationalExprRef root)
      match expression.node with
      | .concrete _ => pure ()
      | .primitive _ arguments =>
          arguments.forM fun argument =>
            validateOperationalExprReindexSelections map arena argument fuel
      | .select domain branches => do
          if (reindexDynamicSelectionIdentity map domain.identity).isNone then
            throw (.unsupportedOperationalExpr root)
          match branches with
          | .exact roots => roots.forM fun branch =>
              validateOperationalExprReindexSelections map arena branch fuel
          | .shared representative _ =>
              validateOperationalExprReindexSelections map arena representative fuel

/-- Apply one index substitution to every identity-bearing leaf and every selection domain of an
arena-backed indexed expression.  The returned context is independently reconstructed from the
mapped DAG before it is persisted. -/
def reindexIndexedOperationalFact
    (map : IndexMap)
    (arena : OperationalExprArena)
    (expression : IndexedOperationalFact)
    (selectionOverride : DynamicSelectionIdentity → DynamicSelectionIdentity := id) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  if expression.context != map.source || !map.transportValid then
    throw (.unsupportedOperationalExpr expression.payload)
  validateOperationalExprReindexSelections map arena expression.payload (arena.nodes.size + 1)
  let mapFact (fact : OperationalMatrixFact) := match reindexOperationalMatrixFact map fact with
    | some mapped => pure mapped
    | none => throw (.unsupportedOperationalExpr expression.payload)
  let mapSelection (selection : DynamicSelectionIdentity) :=
    selectionOverride ((reindexDynamicSelectionIdentity map selection).getD selection)
  let (arena, root) ← mapOperationalExprM "indexed-fact-reindex" .instantiationMap arena
    expression.payload mapFact mapSelection
  let matrixExpression : IndexedFact Nat := { expression with payload := expression.payload.root }
  let reindexed ← match matrixExpression.reindex map (fun _ _ => some root) with
    | some fact => pure fact
    | none => throw (.unsupportedOperationalExpr expression.payload)
  let derived ← arena.indexContextFor root
  if derived != reindexed.context then throw (.unsupportedOperationalExpr root)
  let result : IndexedOperationalFact := { reindexed with payload := .matrix root }
  let arena ← arena.rememberIndexedExpr result
  pure (arena, result)

def indexedArtifactReindexFixture : Bool :=
  let binder : IndexVariable := {
    owner := { stage := ⟨"indexed-artifact-fixture"⟩, scope := ⟨[]⟩, node := ⟨0⟩ }
    slot := 0
    count := .constant 4
  }
  let source : IndexContext := { binders := #[binder] }
  let map : IndexMap := {
    source
    destination := emptyContext
    assignments := #[.constant 3]
  }
  match reindexOperationalPrimitiveIdentityFully map
      (.indexedArtifact ⟨"artifact-producer"⟩ (.variable binder)) with
  | some (.indexedArtifact producer (.constant 3)) => producer.name == "artifact-producer"
  | _ => false

def indexedSelectionReindexFixture : Bool :=
  let selectorVariable : IndexVariable := {
    owner := { stage := ⟨"indexed-selection-fixture"⟩, scope := ⟨[]⟩, node := ⟨0⟩ }
    slot := 0
    count := .constant 4
  }
  let source : IndexContext := { binders := #[selectorVariable] }
  let map : IndexMap := {
    source
    destination := emptyContext
    assignments := #[.constant 2]
  }
  let binder : FamilyTemplateBinder := {
    owner := temporaryScope
    producerNode := 0
    binderSlot := 0
  }
  let selection : DynamicSelectionIdentity := {
    index := .protocolInput ⟨"selector-wire"⟩
    expression := .variable selectorVariable
  }
  let origin : MatrixOriginIdentity :=
    .indexed binder selection.expression (.protocolInput ⟨"selected-family"⟩)
  match reindexMatrixOriginIdentity map origin with
  | some (.indexed _ mapped _) => mapped == .constant 2
  | _ => false

/-- Scalar/value provenance uses the same `IndexMap` as matrix and public identities.  This
prevents a reindexed relation snapshot from retaining a predecessor selector in an embedded
value factor. -/
def indexedValueReindexFixture : Bool :=
  let selectorVariable : IndexVariable := {
    owner := { stage := ⟨"indexed-value-fixture"⟩, scope := ⟨[]⟩, node := ⟨0⟩ }
    slot := 0
    count := .constant 4
  }
  let source : IndexContext := { binders := #[selectorVariable] }
  let map : IndexMap := {
    source
    destination := emptyContext
    assignments := #[.constant 2]
  }
  let binder : FamilyTemplateBinder := {
    owner := temporaryScope
    producerNode := 0
    binderSlot := 0
  }
  let origin : OperationalValueOrigin :=
    .indexed binder (.variable selectorVariable) (.protocolInput ⟨"value-input"⟩)
  match reindexOperationalValueOrigin map origin with
  | some (.indexed _ (.constant 2) (.protocolInput input)) => input.name == "value-input"
  | _ => false

def indexedLoopIdentityReindexFixture : Bool :=
  let lane : IndexVariable := {
    owner := { stage := ⟨"indexed-loop-identity-fixture"⟩, scope := ⟨[]⟩, node := ⟨0⟩ }
    slot := 0
    count := .constant 4
  }
  let source : IndexContext := { binders := #[lane] }
  let map : IndexMap := {
    source
    destination := emptyContext
    assignments := #[.constant 3]
  }
  let value := OperationalValueOrigin.loopInstance 0 (.variable lane)
    (.protocolInput ⟨"loop-value"⟩)
  let matrix := MatrixOriginIdentity.loopInstance 0 (.variable lane)
    (.protocolInput ⟨"loop-matrix"⟩)
  let publicIdentity := PublicMatrixIdentity.loopInstance 0 (.variable lane)
    (.sampledTrapdoor temporaryScope { node := 0, port := 0 })
  let familyValue := OperationalValueOrigin.protocolFamilyElement ⟨"family-value"⟩
    (.variable lane)
  let familyMatrix := MatrixOriginIdentity.protocolFamilyElement ⟨"family-matrix"⟩
    (.variable lane)
  match reindexOperationalValueOrigin map value, reindexMatrixOriginIdentity map matrix,
      reindexPublicMatrixIdentity map publicIdentity,
      reindexOperationalValueOrigin map familyValue,
      reindexMatrixOriginIdentity map familyMatrix with
  | some (.loopInstance _ (.constant 3) _), some (.loopInstance _ (.constant 3) _),
      some (.loopInstance _ (.constant 3) _),
      some (.protocolFamilyElement _ (.constant 3)),
      some (.protocolFamilyElement _ (.constant 3)) => true
  | _, _, _, _, _ => false

def materializedSelectionIdentityFixture : Bool :=
  let first := DynamicSelectionIdentity.fromOrigin (.protocolInput ⟨"shared-selector"⟩) 4
  let same := DynamicSelectionIdentity.fromOrigin (.protocolInput ⟨"shared-selector"⟩) 4
  let distinct := DynamicSelectionIdentity.fromOrigin (.protocolInput ⟨"independent-selector"⟩) 4
  first.expression == same.expression && first.expression != distinct.expression

example : indexedArtifactReindexFixture = true := by native_decide
example : indexedSelectionReindexFixture = true := by native_decide
example : indexedValueReindexFixture = true := by native_decide
example : indexedLoopIdentityReindexFixture = true := by native_decide
example : materializedSelectionIdentityFixture = true := by native_decide

def mapOperationalCompressionToken
    (mapOrigin : MatrixOriginIdentity → MatrixOriginIdentity)
    (mapPublic : PublicMatrixIdentity → PublicMatrixIdentity)
    (mapValue : OperationalValueOrigin → OperationalValueOrigin)
    (mapBound : OperationalBoundExpr → OperationalBoundExpr) :
    OperationalCompressionToken → OperationalCompressionToken
  | .primitive identity =>
      .primitive (mapOperationalPrimitiveIdentity mapOrigin mapPublic mapValue identity)
  | .summaryBound bound => .summaryBound (mapBound bound)
  | token => token

def mapOperationalBoundedSummary
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

def mapOperationalPolynomial
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

def mapRelationSnapshotPolynomial
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

def namespaceFreshSummary
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

def namespaceFreshRelation
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

def shiftTargetPreviousDepth
    (target : RelationTargetSummary) : RelationTargetSummary := {
  target with
  totalHardBound := shiftPreviousDepth target.totalHardBound
  polynomial := mapRelationSnapshotPolynomial id id id shiftPreviousDepth target.polynomial
}

def shiftRelationPreviousDepth :
    OperationalMatrixRelation → OperationalMatrixRelation
  | .decomposition relation => .decomposition {
      relation with inputSummary := shiftTargetPreviousDepth relation.inputSummary }
  | .preimage relation => .preimage {
      relation with targetSummary := shiftTargetPreviousDepth relation.targetSummary }

def shiftMatrixFactPreviousDepth
    (fact : OperationalMatrixFact) : OperationalMatrixFact := {
  fact with
  totalHardBound := shiftPreviousDepth fact.totalHardBound
  relations := fact.relations.map shiftRelationPreviousDepth
  polynomial := mapOperationalPolynomial id id id shiftPreviousDepth
    shiftRelationPreviousDepth fact.polynomial
}

/-- Insert a new innermost recurrence state. References already present in invariant facts then
refer to the enclosing state, while the new carried placeholders continue to use depth zero. -/
partial def shiftFactPreviousDepth
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | expression@{ payload := .scalar _, .. } => do
      let update : OperationalScalarFact → OperationalScalarFact
        | .trapdoor fact => .trapdoor { fact with maximum := shiftPreviousDepth fact.maximum }
        | .integer fact => .integer {
            fact with
            lowerExpression := shiftPreviousDepth fact.lowerExpression
            upperExpression := shiftPreviousDepth fact.upperExpression
          }
        | fact => fact
      let (arena, mapped) ← mapIndexedScalarLeaves arena expression update
      pure (arena, mapped)
  | expression@{ payload := .directValue root, .. } => do
      let (direct, mapped) ← arena.direct.mapMatrixValue root
        (fun fact => pure (shiftMatrixFactPreviousDepth fact))
      let value ← match direct.valueAt? mapped with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef mapped)
      if value.context != expression.context then throw (.unsupportedOperationalExpr mapped)
      pure ({ arena with direct }, {
        context := value.context
        payload := .directValue mapped
        storage := value.storage
      })
  | expression@{ payload := .matrix _, .. } => do
      let (arena, root) ← mapOperationalExpr "shift-previous-depth" .instantiationMap arena
        expression.payload shiftMatrixFactPreviousDepth
      let mapped : IndexedOperationalFact := { expression with payload := .matrix root }
      pure (← arena.rememberIndexedExpr mapped, mapped)

/-- Namespace only identities created by this exact output.  Caller origins transported through
an input are deliberately left unchanged. -/
def namespaceFreshMatrixFact
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (fact : OperationalMatrixFact) : OperationalMatrixFact := {
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

/-- Namespace a newly materialized direct matrix leaf without converting it into a legacy DAG.
Delayed direct nodes retain the already-namespaced identities of their inputs. -/
def namespaceFreshDirectMatrixOutput
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let id ← match fact.payload with
    | .directValue id => pure id
    | .matrix id | .scalar id => throw (.unsupportedOperationalExpr id)
  match arena.direct.valueAt? id with
  | some { context, payload := .shared (.matrix matrixType) (.matrix reference), .. } =>
      let leaf ← match arena.direct.fixed.matrices[reference]? with
        | some leaf => pure leaf
        | none => throw (.invalidOperationalExprRef reference)
      let (fixed, replacement) := arena.direct.fixed.pushMatrix (namespaceFreshMatrixFact scope wire leaf)
      let direct := { arena.direct with fixed }
      let (direct, replacement) ← match direct.pushShared context (.matrix matrixType) replacement with
        | some replacement => pure replacement
        | none => throw (.unsupportedOperationalExpr id)
      pure ({ arena with direct }, { fact with payload := .directValue replacement })
  | some _ => pure (arena, fact)
  | none => throw (.invalidOperationalExprRef id)

def namespaceFreshScalarFact
    (scope : ScopeTemplateKey)
    (wire : WireRef) : OperationalScalarFact → OperationalScalarFact
  | .trapdoor fact => .trapdoor {
      fact with publicIdentity := namespaceFreshPublicIdentity scope wire fact.publicIdentity }
  | .integer fact => .integer {
      fact with origin := namespaceFreshValueOrigin scope wire fact.origin }
  | .bytes fact => .bytes {
      fact with origin := namespaceFreshValueOrigin scope wire fact.origin }
  | fact => fact

def namespaceFreshOutput
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  match fact with
  | expression@{ payload := .scalar _, .. } => do
      let (arena, mapped) ← mapIndexedScalarLeaves arena expression (namespaceFreshScalarFact scope wire)
      pure (arena, mapped)
  | expression@{ payload := .matrix _, .. } =>
      let (arena, root) ← mapOperationalExpr "namespace-fresh-output" .instantiationMap arena
        expression.payload (namespaceFreshMatrixFact scope wire)
      let mapped : IndexedOperationalFact := { expression with payload := .matrix root }
      pure (← arena.rememberIndexedExpr mapped, mapped)
  | expression@{ payload := .directValue _, .. } =>
      namespaceFreshDirectMatrixOutput scope wire arena expression

def namespaceFreshSelectedMatrixSummary
    (scope : ScopeTemplateKey)
    (wire : WireRef)
    (summary : SelectedMatrixSummary) : Option SelectedMatrixSummary := do
  let conservative ← summary.conservativeFact
  let mapped := namespaceFreshMatrixFact scope wire conservative
  transferSelectedMatrixSummary .instantiationMap #[summary] mapped

/-- Namespace the concrete leaves created by one output without rebuilding its expression DAG.
The expression arena is analysis-local and the selected owner has not yet been published in
`OperationalScopeFacts`, so updating those exact owner nodes in place preserves sharing while
leaving every imported subtree untouched. -/
def namespaceOperationalExprInPlace
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
          let mapped := namespaceFreshMatrixFact scope wire fact
          pure (({ arena with nodes := arena.nodes.set! root {
            expression with node := .concrete mapped
          } }).invalidateIndexedExpr root |>.invalidateEvaluationMemo root, visited)
      | .primitive operation arguments =>
          let mut arena := arena
          let mut visited := visited
          for argument in arguments do
            let (nextArena, nextVisited) ←
              namespaceOperationalExprInPlace scope wire arena visited argument fuel
            arena := nextArena
            visited := nextVisited
          pure (({ arena with nodes := arena.nodes.set! root {
            expression with node := .primitive { operation with ownerScope := some scope } arguments
          } }).invalidateIndexedExpr root |>.invalidateEvaluationMemo root, visited)
      | .select selection (.exact branches) =>
          let mut arena := arena
          let mut visited := visited
          for branch in branches do
            let (nextArena, nextVisited) ←
              namespaceOperationalExprInPlace scope wire arena visited branch fuel
            arena := nextArena
            visited := nextVisited
          let mappedSelection := selection.identity.withOrigin
            (namespaceFreshValueOrigin scope wire selection.index)
          let (domainArena, mappedDomain) :=
            arena.internSelectionDomain mappedSelection branches.size
          pure (({ domainArena with nodes := domainArena.nodes.set! root {
            expression with node := .select mappedDomain (.exact branches)
          } }).invalidateIndexedExpr root |>.invalidateEvaluationMemo root, visited)
      | .select selection (.shared representative summary) =>
          let summary ← arena.validatedSchema summary
          let (arena, visited) ← namespaceOperationalExprInPlace scope wire arena visited
            representative fuel
          let mappedSelection := selection.identity.withOrigin
            (namespaceFreshValueOrigin scope wire selection.index)
          let mappedSummary ← match namespaceFreshSelectedMatrixSummary scope wire summary with
            | some mapped => pure {
                mapped with selectionOrigin := some (selectionDomainKind mappedSelection.index) }
            | none => throw (.unsupportedOperationalExpr root)
          let (arena, mappedDomain) := arena.internSelectionDomain mappedSelection selection.count
          let (arena, mappedSchema) := arena.internValidatedSchema mappedSummary
          pure (({ arena with nodes := arena.nodes.set! root {
            expression with node := (.select mappedDomain
              (.shared representative mappedSchema))
          } }).invalidateIndexedExpr root |>.invalidateEvaluationMemo root, visited)

partial def booleanFamilyCount
    (arena : OperationalExprArena) (fact : OperationalFact) : Option Int :=
  match fact with
  | expression@{ payload := .scalar _, .. } => do
      let (domain, storage) ← (arena.scalarSelectionDomain expression).toOption
      let rec allBoolean : Nat → Nat → Bool
        | 0, _ => false
        | fuel + 1, root => match arena.scalarNodes[root]? with
            | some (.concrete .boolean) => true
            | some (.primitive _ _ .boolean) => true
            | some (.primitive ..) => false
            | some (.selectExact _ branches) => branches.all (allBoolean fuel)
            | some (.selectShared _ _ _ representative) => allBoolean fuel representative
            | _ => false
      let roots := match storage with
        | .inl branches => branches
        | .inr representative => #[representative]
      if roots.all (allBoolean (arena.scalarNodes.size + 1)) then
        some (Int.ofNat domain.count)
      else none
  | _ => none

def instantiateHashIdentityLoopIndex
    (slot index : Nat) (identity : DeterministicHashIdentity) : DeterministicHashIdentity :=
  { identity with
    parameterEnvironment := replaceLoopIndex identity.parameterEnvironment slot index
    parameterDomains := instantiateParameterDomains slot index identity.parameterDomains
  }

def instantiateOriginLoopIndex
    (slot index : Nat) : MatrixOriginIdentity → MatrixOriginIdentity
  | .value scope wire => .loopInstance slot (.constant index) (.value scope wire)
  | .protocolInput input => .protocolInput input
  | .protocolFamilyElement input familyIndex => .protocolFamilyElement input familyIndex
  | .deterministicHash identity =>
      .deterministicHash (instantiateHashIdentityLoopIndex slot index identity)
  | .loopInstance existingSlot existingIndex source =>
      .loopInstance existingSlot existingIndex (instantiateOriginLoopIndex slot index source)
  | .indexed binder expression source =>
      .indexed binder expression (instantiateOriginLoopIndex slot index source)

def instantiateValueOriginLoopIndex
    (slot index : Nat) : OperationalValueOrigin → OperationalValueOrigin
  | .local scope wire => .loopInstance slot (.constant index) (.local scope wire)
  | .protocolInput input => .protocolInput input
  | .protocolFamilyElement input familyIndex => .protocolFamilyElement input familyIndex
  | .loopInstance existingSlot existingIndex source =>
      .loopInstance existingSlot existingIndex
        (instantiateValueOriginLoopIndex slot index source)
  | .indexed binder expression source =>
      .indexed binder expression (instantiateValueOriginLoopIndex slot index source)

def instantiatePublicIdentityLoopIndex
    (slot index : Nat) : PublicMatrixIdentity → PublicMatrixIdentity
  | identity@(.gadget ..) => identity
  | .sampledTrapdoor scope wire =>
      .loopInstance slot (.constant index) (.sampledTrapdoor scope wire)
  | .indexed binder expression source =>
      .indexed binder expression (instantiatePublicIdentityLoopIndex slot index source)
  | .loopInstance existingSlot existingIndex source =>
      .loopInstance existingSlot existingIndex
        (instantiatePublicIdentityLoopIndex slot index source)

def instantiateTargetLoopIndex
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

def instantiateRelationLoopIndex
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

def instantiateMatrixFactLoopIndex
    (slot index : Nat) (fact : OperationalMatrixFact) : OperationalMatrixFact := {
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

partial def instantiateFactLoopIndex
    (slot index : Nat) (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | expression@{ payload := .scalar _, .. } => do
      let update : OperationalScalarFact → OperationalScalarFact
        | .trapdoor fact => .trapdoor {
            fact with
            maximum := instantiateBoundLoopIndex slot index fact.maximum
            preimageCutoff := fact.preimageCutoff.map (instantiateBoundLoopIndex slot index)
            publicIdentity := instantiatePublicIdentityLoopIndex slot index fact.publicIdentity }
        | .integer fact => .integer {
            fact with
            origin := instantiateValueOriginLoopIndex slot index fact.origin
            lowerExpression := instantiateBoundLoopIndex slot index fact.lowerExpression
            upperExpression := instantiateBoundLoopIndex slot index fact.upperExpression }
        | .bytes fact => .bytes {
            fact with origin := instantiateValueOriginLoopIndex slot index fact.origin }
        | fact => fact
      let (arena, mapped) ← mapIndexedScalarLeaves arena expression update
      pure (arena, mapped)
  | expression@{ payload := .matrix _, .. } => do
      let (arena, root) ← mapOperationalExpr "instantiate-loop-index" .instantiationMap arena
        expression.payload (instantiateMatrixFactLoopIndex slot index)
      let mapped : IndexedOperationalFact := { expression with payload := .matrix root }
      pure (← arena.rememberIndexedExpr mapped, mapped)
  | expression@{ payload := .directValue root, .. } => do
      let value ← match arena.direct.valueAt? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      if value.context != expression.context then throw (.unsupportedOperationalExpr root)
      let (direct, mapped) ← arena.direct.mapMatrixValue root
        (fun fact => pure (instantiateMatrixFactLoopIndex slot index fact))
      let value ← match direct.valueAt? mapped with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef mapped)
      pure ({ arena with direct }, {
        context := value.context
        payload := .directValue mapped
        storage := value.storage
      })

def joinCanonicalRanges : List CanonicalRange → CanonicalRange
  | [] => .unknown
  | ranges =>
      if ranges.all (fun range => match range with | .below _ => true | .unknown => false) then
        .below (ranges.foldl (fun result range => match range with
          | .below value => max result value
          | .unknown => result) 0)
      else .unknown

/-- Substitute the symbolic index of a previously constructed uniform loop family with the
current consumer-loop index. This preserves correlation when a uniform family is zipped into a
later loop instead of treating the producer's template index as an independent selection. -/
partial def substituteLoopTemplateValueOrigin
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : OperationalValueOrigin → OperationalValueOrigin
  | origin@(.local ..) => if isLoopTemplateSelection binder origin then replacement else origin
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | origin@(.loopInstance slot index source) =>
      if isLoopTemplateSelection binder origin then replacement
      else .loopInstance slot index (substituteLoopTemplateValueOrigin binder replacement source)
  | .indexed indexedBinder expression source =>
      .indexed indexedBinder expression
        (substituteLoopTemplateValueOrigin binder replacement source)

def substituteLoopTemplateHashIdentity
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (identity : DeterministicHashIdentity) : DeterministicHashIdentity := {
  identity with
  keyOrigin := substituteLoopTemplateValueOrigin binder replacement identity.keyOrigin
  trailingIntegerOrigins := identity.trailingIntegerOrigins.map
    (substituteLoopTemplateValueOrigin binder replacement)
}

partial def substituteLoopTemplateMatrixOrigin
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : MatrixOriginIdentity → MatrixOriginIdentity
  | origin@(.value ..) => origin
  | origin@(.protocolInput _) => origin
  | origin@(.protocolFamilyElement _ _) => origin
  | .deterministicHash identity =>
      .deterministicHash (substituteLoopTemplateHashIdentity binder replacement identity)
  | .loopInstance slot index source =>
      .loopInstance slot index (substituteLoopTemplateMatrixOrigin binder replacement source)
  | .indexed selectedBinder expression source =>
      .indexed selectedBinder expression
        (substituteLoopTemplateMatrixOrigin binder replacement source)

partial def substituteLoopTemplatePublicIdentity
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin) : PublicMatrixIdentity → PublicMatrixIdentity
  | identity@(.sampledTrapdoor ..) => identity
  | identity@(.gadget ..) => identity
  | .loopInstance slot index source =>
      .loopInstance slot index (substituteLoopTemplatePublicIdentity binder replacement source)
  | .indexed selectedBinder expression source =>
      .indexed selectedBinder expression
        (substituteLoopTemplatePublicIdentity binder replacement source)

def substituteLoopTemplateTarget
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

def substituteLoopTemplateRelation
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

def substituteLoopTemplateMatrixFact
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

def substituteLoopTemplateSummary
    (binder : FamilyTemplateBinder)
    (replacement : OperationalValueOrigin)
    (summary : SelectedMatrixSummary) : SelectedMatrixSummary := {
  summary with
  sharedLastPublicIdentity := summary.sharedLastPublicIdentity.map
    (substituteLoopTemplatePublicIdentity binder replacement)
  selectionOrigin := some (selectionDomainKind replacement)
}

/-- Select one element of a uniform family by the exact executable index wire.  The wrapper is
structural: two selections compare equal only when the family binder and index-wire instance are
identical.  Arithmetic equivalence of two index computations is never inferred. -/
def dynamicSelectionScope : OperationalValueOrigin → ScopeTemplateKey
  | .local scope _ => scope
  | .protocolInput _ | .protocolFamilyElement _ _ => temporaryScope
  | .loopInstance _ _ source => dynamicSelectionScope source
  | .indexed _ _ source => dynamicSelectionScope source

def indexValueOrigin
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (source : OperationalValueOrigin) : OperationalValueOrigin :=
  .indexed binder selection.expression source

def indexMatrixFact
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (subject : WireRef)
    (fact : OperationalMatrixFact) : OperationalMatrixFact :=
  let mapOrigin (origin : MatrixOriginIdentity) := .indexed binder selection.expression origin
  let mapPublic (identity : PublicMatrixIdentity) := .indexed binder selection.expression identity
  let mapValue := indexValueOrigin binder selection
  let mapTarget (target : RelationTargetSummary) : RelationTargetSummary := {
    target with
    origin := mapOrigin target.origin
    polynomial := mapRelationSnapshotPolynomial mapOrigin mapPublic mapValue id target.polynomial
  }
  let mapRelation : OperationalMatrixRelation → OperationalMatrixRelation
    | .decomposition relation => .decomposition {
        relation with
        producer := mapOrigin relation.producer
        publicIdentity := mapPublic relation.publicIdentity
        inputOrigin := mapOrigin relation.inputOrigin
        inputSummary := mapTarget relation.inputSummary
      }
    | .preimage relation => .preimage {
        relation with
        producer := mapOrigin relation.producer
        publicIdentity := mapPublic relation.publicIdentity
        targetOrigin := mapOrigin relation.targetOrigin
        targetSummary := mapTarget relation.targetSummary
      }
  { fact with
    subject
    origin := mapOrigin fact.origin
    identity := fact.identity.map mapPublic
    relations := fact.relations.map mapRelation
    polynomial := mapOperationalPolynomial mapOrigin mapPublic mapValue id mapRelation fact.polynomial
  }

def indexScalarFact
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (subject : WireRef) : OperationalScalarFact → OperationalScalarFact
  | .integer fact => .integer {
      fact with subject, origin := indexValueOrigin binder selection fact.origin }
  | .trapdoor fact => .trapdoor {
      fact with subject, publicIdentity := .indexed binder selection.expression fact.publicIdentity }
  | .bytes fact => .bytes {
      fact with subject, origin := indexValueOrigin binder selection fact.origin }
  | fact => fact

partial def selectDynamicUniformFact
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (subject : WireRef)
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × OperationalFact)
  | expression@{ payload := .scalar _, .. } => do
      let (arena, mapped) ← mapIndexedScalarLeaves arena expression
        (indexScalarFact binder selection subject)
      pure (arena, mapped)
  | expression@{ payload := .matrix _, .. } => do
      let (arena, root) ← mapOperationalExpr "select-dynamic-uniform" .instantiationMap arena
        expression.payload (indexMatrixFact binder selection subject)
      let mapped : IndexedOperationalFact := { expression with payload := .matrix root }
      pure (← arena.rememberIndexedExpr mapped, mapped)
  | { payload := .directValue root, .. } =>
      throw (.unsupportedOperationalExpr root)

/-- A scalar Shared node is legal only when every stored result is the symbolic template for the
same outer family selector.  Boolean/real/blob atoms have no provenance fields; indexed integer,
trapdoor, and byte atoms must carry the exact binder and selector installed by the family
construction.  Nested selectors are checked recursively without enumerating their logical lanes. -/
partial def scalarHasCheckedSharedTemplate
    (arena : OperationalExprArena)
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (subject : WireRef)
    (root : Nat)
    (fuel : Nat) : Bool :=
  if fuel = 0 then false else
  let validFact : OperationalScalarFact → Bool := fun fact => match fact with
    | .integer fact => fact.subject == subject && match fact.origin with
        | .indexed actualBinder expression _ =>
            actualBinder == binder && expression == selection.expression
        | _ => false
    | .trapdoor fact => fact.subject == subject && match fact.publicIdentity with
        | .indexed actualBinder expression _ =>
            actualBinder == binder && expression == selection.expression
        | _ => false
    | .bytes fact => fact.subject == subject && match fact.origin with
        | .indexed actualBinder expression _ =>
            actualBinder == binder && expression == selection.expression
        | _ => false
    | .boolean | .real | .typedBlob _ | .unknown _ => true
  match arena.scalarNodes[root]? with
  | some (.concrete fact) => validFact fact
  | some (.primitive _ arguments result) =>
      validFact result && arguments.all fun argument =>
        scalarHasCheckedSharedTemplate arena binder selection subject argument (fuel - 1)
  | some (.selectExact _ branches) => !branches.isEmpty && branches.all fun branch =>
      scalarHasCheckedSharedTemplate arena binder selection subject branch (fuel - 1)
  | some (.selectShared _ _ _ representative) =>
      scalarHasCheckedSharedTemplate arena binder selection subject representative (fuel - 1)
  | none => false

def OperationalExprArena.pushOperationalScalarFact
    (arena : OperationalExprArena) : OperationalFact →
    Except OperationalError (OperationalExprArena × Nat)
  | expression@{ payload := .scalar _, .. } => do
      let arena ← arena.rememberIndexedScalar expression
      pure (arena, expression.payload)
  | _ => throw (.unsupportedOperationalExpr arena.scalarNodes.size)

def packIndexedScalarFacts
    (arena : OperationalExprArena)
    (selection : DynamicSelectionIdentity)
    (elements : List OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if elements.isEmpty then throw (.invalidCount 0 0)
  let (arena, roots) ← elements.foldlM (fun (arena, roots) element => do
    let (arena, root) ← arena.pushOperationalScalarFact element
    pure (arena, roots.push root)) (arena, #[])
  let (arena, expression) ← arena.pushScalarSelection selection roots
  pure (arena, expression)

def sharedIndexedScalarFact
    (arena : OperationalExprArena)
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (subject : WireRef)
    (count : Nat)
    (element : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if count = 0 then throw (.invalidCount subject.node 0)
  let (arena, representative) ← match element with
    | expression@{ payload := .scalar _, .. } => do
        let rec visit : Nat → OperationalExprArena → Nat →
            Except OperationalError (OperationalExprArena × Nat)
          | 0, _, root => throw (.unsupportedOperationalExpr root)
          | fuel + 1, arena, root => do
              match arena.scalarNodes[root]? with
              | none => throw (.invalidOperationalExprRef root)
              | some (.concrete scalar) =>
                  pure (arena.pushScalarConcrete
                    (indexScalarFact binder selection subject scalar))
              | some (.primitive kind arguments result) => do
                  let (arena, arguments) ← arguments.foldlM
                    (fun (arena, mapped) argument => do
                      let (arena, argument) ← visit fuel arena argument
                      pure (arena, mapped.push argument)) (arena, #[])
                  pure (arena.pushScalar (.primitive kind arguments
                    (indexScalarFact binder selection subject result)))
              | some (.selectExact domain branches) => do
                  let (arena, branches) ← branches.foldlM (fun (arena, mapped) branch => do
                    let (arena, branch) ← visit fuel arena branch
                    pure (arena, mapped.push branch)) (arena, #[])
                  pure (arena.pushScalar (.selectExact domain branches))
              | some (.selectShared domain innerBinder innerSubject representative) => do
                  let (arena, representative) ← visit fuel arena representative
                  pure (arena.pushScalar
                    (.selectShared domain innerBinder innerSubject representative))
        visit (arena.scalarNodes.size + 1) arena expression.payload
    | _ => throw (.unsupportedOperationalExpr arena.scalarNodes.size)
  if !scalarHasCheckedSharedTemplate arena binder selection subject representative
      (arena.scalarNodes.size + 1) then
    throw (.unsupportedOperationalExpr representative)
  let (arena, domain) := arena.internSelectionDomain selection count
  let (arena, root) := arena.pushScalar (.selectShared domain binder subject representative)
  let expression ← arena.indexedScalar root
  let arena ← arena.rememberIndexedScalar expression
  pure (arena, expression)

/-- Represent a dynamic choice from a construction-uniform matrix family by one checked schema
envelope.  The selected representative carries the unresolved index in every matrix and relation
identity, so the envelope is not an equal-value collapse. -/
def selectDynamicUniformIndexedMatrixEnvelope
    (arena : OperationalExprArena)
    (binder : FamilyTemplateBinder)
    (selection : OperationalValueOrigin)
    (subject : WireRef)
    (count : Nat)
    (deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (expression : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if count = 0 then throw (.invalidCount subject.node 0)
  let selection := DynamicSelectionIdentity.fromOrigin selection count
  let root ← match expression.payload with
    | .matrix root => pure root
    | .directValue root => throw (.unsupportedOperationalExpr root)
    | .scalar _ => throw (.operandNotMatrix subject.node subject)
  let arena ← arena.rememberIndexedExpr expression
  let (arena, representative) ← mapOperationalExpr
    s!"dynamic-uniform-indexed:{subject.node}:{subject.port}:{reprStr binder}" .instantiationMap
    arena root (indexMatrixFact binder selection subject)
  let (schemaFact, _) ← deriveSchema arena [] representative
    (OperationalExprEvaluationState.empty arena)
  let (arena, root) ← arena.pushSharedSelection
    selection count representative (selectedMatrixSummary #[schemaFact])
  let expression ← arena.indexedExpr root
  let arena ← arena.rememberIndexedExpr expression
  pure (arena, expression)

/-- Close one parallel-loop matrix output in the producer's loop coordinate.  An already-indexed
body output is reindexed onto that coordinate so its existing Exact alternatives remain correlated
with the zipped input selector.  Only a genuinely nonindexed body output needs a new Shared
selection envelope. -/
def exactlyOneIndexedBinder
    (context : IndexContext)
    (root : Nat) : Except OperationalError IndexVariable := do
  if context.binders.size != 1 then throw (.unsupportedOperationalExpr root)
  match context.binders[0]? with
  | some binder => pure binder
  | none => throw (.unsupportedOperationalExpr root)

/-- Exact owner-bearing binder for a matrix `familyPack`.  This is distinct from a dynamic
selection identity: packing constructs a finite function, while later static/dynamic gets are
only capture-free substitutions into this binder. -/
def packedDirectFamilyBinder
    (scope : ScopeTemplateKey)
    (node : Nat)
    (count : IntExpr) : IndexVariable := {
  owner := {
    stage := ⟨s!"operational-family-pack:{reprStr scope}"⟩
    scope := ⟨[]⟩
    node := ⟨node⟩
  }
  slot := 0
  count
}

/-- Resolve the lane binder introduced by a direct matrix-family producer.  A selected family
retains its independent branch-choice binder, so callers must substitute this exact lane binder
rather than assuming that the carrier context has a single dimension. -/
def directFamilyLaneBinder
    (scope : ScopeTemplateKey)
    (producerNode : Nat)
    (producer : Node)
    (familyWire : WireRef)
    (countExpression : IntExpr)
    (count : Nat) : Except OperationalError IndexVariable := do
  if familyWire.node != producerNode || count == 0 then
    throw (.loopInputModeMismatch producerNode familyWire.port)
  match producer.kind with
  | .familyPack => pure (packedDirectFamilyBinder scope producerNode countExpression)
  | .parallelLoop _ _ indexSlot _ _ =>
      let selection := DynamicSelectionIdentity.fromDeclaredCount
        (.loopInstance indexSlot (.constant 0) (.local scope familyWire)) countExpression
      match selection.expression with
      | .variable binder => pure binder
      | _ => throw (.loopInputModeMismatch producerNode familyWire.port)
  | .select =>
      let selection := DynamicSelectionIdentity.fromDeclaredCount (.local scope familyWire) countExpression
      match selection.expression with
      | .variable binder => pure binder
      | _ => throw (.loopInputModeMismatch producerNode familyWire.port)
  | _ => throw (.loopInputModeMismatch producerNode familyWire.port)

/-- Pack matrix lanes entirely inside the authoritative direct indexed carrier.  Closed fixed
lanes take the compact fixed-reference table; delayed/mapped lanes retain their exact direct IDs
in an equally explicit ordered table.  Neither path creates a legacy selection node. -/
def packDirectMatrixFamily
    (scope : ScopeTemplateKey)
    (node : Nat)
    (environment : ParamEnvironment)
    (count : IntExpr)
    (arena : OperationalExprArena)
    (elements : Array OperationalFact) : Except OperationalError
      (OperationalExprArena × OperationalFact) := do
  let binder := packedDirectFamilyBinder scope node count
  let ids ← elements.mapM fun element => match element.payload with
    | .directValue id => pure id
    | .matrix _ | .scalar _ => throw (.operandNotMatrix node { node, port := 0 })
  let values ← ids.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef id)
  let schema ← match values[0]? with
    | some value => match value.payload.schema with
      | .matrix matrixType => pure (.matrix matrixType)
      | .scalar _ => throw (.operandNotMatrix node { node, port := 0 })
    | none => throw (.invalidCount node 0)
  if values.any (fun value => value.payload.schema != schema) then
    throw (.outputTypeMismatch node)
  let explicitReferences := values.mapM fun value => match value with
    | { context, payload := .shared (.matrix _) reference, .. } =>
        if context.binders.isEmpty then some reference else none
    | _ => none
  let (direct, result) ← match explicitReferences with
    | some references => do
      match arena.direct.pushExplicit environment { binders := #[binder] } binder schema references with
      | some result => pure result
      | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
    | none => do
      match arena.direct.pushExplicitValues environment binder ids with
      | some result => pure result
      | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })

/-- Pack scalar family lanes in the same direct carrier used by matrix families.  In particular,
integer index families stay ordered direct tables instead of becoming legacy scalar selections. -/
def packDirectScalarFamily
    (scope : ScopeTemplateKey)
    (node : Nat)
    (environment : ParamEnvironment)
    (count : IntExpr)
    (arena : OperationalExprArena)
    (elements : Array OperationalFact) : Except OperationalError
      (OperationalExprArena × OperationalFact) := do
  let binder := packedDirectFamilyBinder scope node count
  let ids ← elements.mapM fun element => match element.payload with
    | .directValue id => pure id
    | .matrix _ | .scalar _ => throw (.operandNotInteger node { node, port := 0 })
  let values ← ids.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef id)
  let first ← match values[0]? with
    | some value => pure value
    | none => throw (.invalidCount node 0)
  let schema ← match first.payload.schema with
      | .scalar schema => pure (.scalar schema)
      | .matrix _ => throw (.operandNotInteger node { node, port := 0 })
  if values.any (fun value => value.payload.schema != schema) then throw (.outputTypeMismatch node)
  let explicitReferences := values.mapM fun value => match value with
    | { context, payload := .shared (.scalar _) reference, .. } =>
        if context.binders.isEmpty then some reference else none
    | _ => none
  let (direct, result) ← match explicitReferences with
    | some references => match arena.direct.pushExplicit environment { binders := #[binder] } binder schema references with
      | some result => pure result
      | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
    | none => match arena.direct.pushExplicitValues environment binder ids with
      | some result => pure result
      | none => throw (.unsupportedOperationalExpr arena.direct.values.size)
  let value ← match direct.valueAt? result with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef result)
  pure ({ arena with direct }, {
    context := value.context
    payload := .directValue result
    storage := value.storage
  })

/-- Evaluate an executable matrix `select` as application of one ordered direct family table.
The fresh table binder is substituted by the selector variable through an `IndexMap`; this keeps
two uses of the same executable selector correlated without expanding alternatives into indicator
products.  Matrix branches must already be direct values: mixed or legacy storage is rejected at
this graph boundary instead of being converted back to the selection arena. -/
def selectDirectMatrixBranches
    (scope : ScopeTemplateKey)
    (node : Nat)
    (selection : OperationalIntegerFact)
    (subject : WireRef)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (arena : OperationalExprArena)
    (branches : Array OperationalFact) : Except OperationalError
      (OperationalExprArena × OperationalFact) := do
  if branches.isEmpty then throw (.invalidCount node 0)
  let count := branches.size
  let ids ← branches.mapM fun branch => match branch.payload with
    | .directValue id => pure id
    | .matrix id | .scalar id => throw (.unsupportedOperationalExpr id)
  let values ← ids.mapM fun id => match arena.direct.valueAt? id with
    | some value => pure value
    | none => throw (.invalidOperationalExprRef id)
  if values.any fun value => value.payload.schema != .matrix matrixType then
    throw (.outputTypeMismatch node)
  let familyCount := IntExpr.constant (Int.ofNat count)
  let binder := packedDirectFamilyBinder scope node familyCount
  let (arena, family) ← packDirectMatrixFamily scope node environment familyCount arena branches
  if family.context.binders.contains binder == false then
    throw (.unsupportedOperationalExpr node)
  let map ←
    if selection.lower == selection.upper then
      match closedStaticIndexMap environment family.context binder selection.lower.toNat with
      | some map => pure map
      | none => throw (.unsupportedOperationalExpr node)
    else
      let dynamicSelection := DynamicSelectionIdentity.fromOrigin selection.origin count
      match dynamicIndexMap family.context binder dynamicSelection.expression with
      | some map => pure map
      | none => throw (.unsupportedOperationalExpr node)
  let (arena, selected) ← arena.reindexDirectMatrixFact map family
  rebindOperationalFact subject arena selected

def selectionIndexedContext
    (selection : DynamicSelectionIdentity)
    (root : Nat) : Except OperationalError IndexContext := do
  match selection.expression with
  | .variable binder => pure { binders := #[binder] }
  | _ => throw (.unsupportedOperationalExpr root)

def closeParallelDirectMatrixOutput
    (scope : ScopeTemplateKey)
    (node indexSlot port : Nat)
    (declaredCount : IntExpr)
    (arena : OperationalExprArena)
    (output : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let root ← match output.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  let subject : WireRef := { node, port }
  let binder : FamilyTemplateBinder := { owner := scope, producerNode := node, binderSlot := indexSlot }
  let selection := DynamicSelectionIdentity.fromDeclaredCount
    (.loopInstance indexSlot (.constant 0) (.local scope subject)) declaredCount
  if output.context.binders.isEmpty then
    let (direct, indexed) ← arena.direct.mapMatrixValue root
      (fun fact => pure (indexMatrixFact binder selection subject fact))
    let indexedValue ← match direct.valueAt? indexed with
      | some value => pure value
      | none => throw (.invalidOperationalExprRef indexed)
    if indexedValue.context != emptyContext then throw (.unsupportedOperationalExpr indexed)
    let destination ← selectionIndexedContext selection indexed
    let map : IndexMap := { source := emptyContext, destination, assignments := #[] }
    let expression : OperationalFact := {
      context := emptyContext, payload := .directValue indexed, storage := indexedValue.storage }
    let (arena, closed) ← ({ arena with direct }).reindexDirectMatrixFact map expression
    pure (arena, closed)
  else
    let sourceBinder ← exactlyOneIndexedBinder output.context root
    let map ← match dynamicIndexMap output.context sourceBinder selection.expression with
      | some map => pure map
      | none => throw (.unsupportedOperationalExpr root)
    arena.reindexDirectMatrixFact map output

def parallelLoopIndexedMatrixOutput
    (scope : ScopeTemplateKey)
    (node indexSlot port : Nat)
    (declaredCount : IntExpr)
    (count : Nat)
    (environment : ParamEnvironment)
    (deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (output : OperationalFact) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  if count = 0 then throw (.invalidCount node 0)
  let subject : WireRef := { node, port }
  let binder : FamilyTemplateBinder := { owner := scope, producerNode := node, binderSlot := indexSlot }
  let selection := DynamicSelectionIdentity.fromDeclaredCount
    (.loopInstance indexSlot (.constant 0) (.local scope subject)) declaredCount
  match output.payload with
  | .directValue _ => closeParallelDirectMatrixOutput scope node indexSlot port declaredCount arena output
  | .scalar root => throw (.unsupportedOperationalExpr root)
  | .matrix _ =>
    match arena.get? output.payload with
    | some { node := .concrete _, .. } => do
      let (arena, representative) ← mapOperationalExpr
        s!"parallel-loop-indexed:{node}:{port}:{indexSlot}" .instantiationMap arena output.payload
        (indexMatrixFact binder selection subject)
      let (schemaFact, _) ← deriveSchema arena environment representative
        (OperationalExprEvaluationState.empty arena)
      let summary := selectedMatrixSummary #[schemaFact]
      let (arena, familyRoot) ← arena.pushSharedSelection selection count representative summary
      let expression ← arena.indexedExpr familyRoot
      let arena ← arena.rememberIndexedExpr expression
      pure (arena, expression)
    | some _ => do
        let sourceBinder ← exactlyOneIndexedBinder output.context output.payload
        let map ← match dynamicIndexMap output.context sourceBinder selection.expression with
          | some value => pure value
          | none => throw (.unsupportedOperationalExpr output.payload)
        let (arena, expression) ← reindexIndexedOperationalFact map arena output
        pure (arena, expression)
    | none => throw (.invalidOperationalExprRef output.payload)

/-- Arena-backed parallel-loop input preparation.  Matrix families retain their exact selected
expression (or one checked schema envelope) instead of materializing an indicator polynomial or
the removed fact-level selected-family representation. -/
def loopTemplateArgumentExprWithDirectLaneBinder
    (arena : OperationalExprArena)
    (node argument : Nat)
    (declaredCount : IntExpr)
    (count : Nat)
    (mode : LoopInputMode)
    (directLaneBinder : Option IndexVariable)
    (_environment : ParamEnvironment)
    (_deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (fact : OperationalFact) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  match mode with
  | .broadcast =>
      match fact with
      | expression@{ payload := .directValue root, .. } =>
          if expression.context.binders.isEmpty then
            let consumer := DynamicSelectionIdentity.fromDeclaredCount
              (.local temporaryScope { node, port := argument }) declaredCount
            let destination ← selectionIndexedContext consumer root
            let map : IndexMap := { source := emptyContext, destination, assignments := #[] }
            arena.reindexDirectMatrixFact map expression
          else
            /- A broadcast aggregate is constant in the loop coordinate, not scalar-valued.
            Keep its source family binder so a body `FamilyGetDynamic` can apply its exact gather
            map instead of silently replacing the aggregate by one loop-lane template. -/
            pure (arena, expression)
      | { payload := .matrix _, .. } =>
          throw (.loopInputModeMismatch node argument)
      | expression@{ payload := .scalar _, .. } => pure (arena, expression)
  | .zip | .zipOffset _ =>
      match fact with
      | expression@{ payload := .directValue _, .. } =>
          let sourceBinder ← match directLaneBinder with
            | some binder => pure binder
            | none => throw (.loopInputModeMismatch node argument)
          if !expression.context.binders.contains sourceBinder then
            throw (.loopInputModeMismatch node argument)
          let offset := match mode with | .zipOffset value => value | _ => 0
          let sourceCount ← match sourceBinder.count.evaluate _environment with
            | some value => if value > 0 then pure value.toNat else throw (.loopInputModeMismatch node argument)
            | none => throw (.loopInputModeMismatch node argument)
          if count + offset > sourceCount then throw (.loopInputModeMismatch node argument)
          let consumer := DynamicSelectionIdentity.fromDeclaredCount
            (.local temporaryScope { node, port := argument }) declaredCount
          let assignment := .offset consumer.expression (Int.ofNat offset)
          let map ← match dynamicIndexMap expression.context sourceBinder assignment with
            | some map => pure map
            | none => throw (.loopInputModeMismatch node argument)
          arena.reindexDirectMatrixFact map expression
      | { payload := .matrix _, .. } =>
          throw (.loopInputModeMismatch node argument)
      | expression@{ payload := .scalar _, .. } =>
          let (domain, _) ← arena.scalarSelectionDomain expression
          let offset := match mode with | .zipOffset value => value | _ => 0
          if count + offset > domain.count then throw (.loopInputModeMismatch node argument)
          let sourceBinder ← match domain.identity.expression with
            | .variable binder => pure binder
            | _ => throw (.loopInputModeMismatch node argument)
          let consumer := DynamicSelectionIdentity.fromOrigin
            (.local temporaryScope { node, port := argument }) count
          let assignment := .offset consumer.expression (Int.ofNat offset)
          let map ← match dynamicIndexMap expression.context sourceBinder assignment with
            | some map => pure map
            | none => throw (.loopInputModeMismatch node argument)
          let (arena, mapped) ← reindexIndexedScalarFact map arena expression
          pure (arena, mapped)

/-- Re-express one construction-uniform family element in the template coordinate of a newly
constructed family.  This is a binder substitution, not a claim that two family lanes have equal
values.  A family without an explicit construction coordinate remains an unresolved selection by
the new lane identity. -/
def reindexUniformMatrixFamilyElement
    (arena : OperationalExprArena)
    (node : Nat)
    (outputLane : OperationalValueOrigin)
    (binder : FamilyTemplateBinder)
    (coordinate : Option LoopCoordinate)
    (element : OperationalFact)
    (count : Nat)
    (deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState)) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  let replacement := match coordinate with
    | some (.loopBinder _ _ _) => outputLane
    | some (.loopBinderOffset _ _ slot offset) =>
        .loopInstance slot (.constant offset) outputLane
    | none => outputLane
  let expression ← match element with
    | expression@{ payload := .matrix _, .. } => pure expression
    | _ => throw (.operandNotMatrix node { node, port := 0 })
  let root := expression.payload
  match coordinate with
  | some _ =>
      let mapFact := substituteLoopTemplateMatrixFact binder replacement
      let mapSelection (selection : DynamicSelectionIdentity) := selection.withOrigin
        (substituteLoopTemplateValueOrigin binder replacement selection.index)
      let mapSummary := substituteLoopTemplateSummary binder replacement
      let cacheNamespace :=
        s!"family-select-reindex:{node}:{reprStr binder}:{reprStr coordinate}"
      let (arena, mapped) ← mapOperationalExpr cacheNamespace .instantiationMap arena root
        mapFact mapSelection none (some mapSummary)
      pure (arena, ← arena.indexedExpr mapped)
  | none =>
      selectDynamicUniformIndexedMatrixEnvelope arena binder replacement
        { node, port := 0 } count deriveSchema expression

/-- Select an exact-table indexed matrix family with a runtime selector.  The table is storage for
one indexed value; this is function application, not a Cartesian expansion of family lanes. -/
def selectIndexedMatrixFamilyDynamic
    (node : Nat)
    (selection : OperationalIntegerFact)
    (subject : WireRef)
    (matrixType : MatrixTypeExpr)
    (environment : ParamEnvironment)
    (deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena)
    (family : IndexedOperationalFact) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact) := do
  let root := family.payload
  let storage ← match arena.get? root with
    | some { node := .select domain (.exact branches), .. } =>
        if domain.count == 0 || branches.size != domain.count then
          throw (.loopInputModeMismatch node 0)
        else pure (some (domain, branches))
    | some { node := .select domain (.shared _ _), .. } =>
        if domain.count == 0 then throw (.loopInputModeMismatch node 0)
        else pure none
    | _ => throw (.loopInputModeMismatch node 0)
  let binder : FamilyTemplateBinder := {
    owner := dynamicSelectionScope selection.origin
    producerNode := node
    binderSlot := 0
  }
  let count := match storage with
    | some (domain, _) => domain.count
    | none => match arena.get? root with
      | some { node := .select domain _, .. } => domain.count
      | _ => 0
  let dynamicSelection := DynamicSelectionIdentity.fromOrigin selection.origin count
  let mapFact := indexMatrixFact binder dynamicSelection subject
  let mapSelection (nested : DynamicSelectionIdentity) := nested.withOrigin
    (indexValueOrigin binder dynamicSelection nested.index)
  match storage with
  | none => do
      let cacheNamespace :=
        s!"indexed-family-shared-dynamic:{node}:{reprStr selection.origin}:{reprStr binder}"
      let (arena, selected) ← mapOperationalExpr cacheNamespace .instantiationMap arena root
        mapFact mapSelection
      let expression ← arena.indexedExpr selected
      let arena ← arena.rememberIndexedExpr expression
      pure (arena, expression)
  | some (_, branches) => do
      let mut arena := arena
      let mut selectedBranches : Array OperationalExprId := #[]
      for branch in branches do
        let cacheNamespace :=
          s!"indexed-family-dynamic:{node}:{reprStr selection.origin}:{reprStr binder}"
        let (nextArena, selected) ← mapOperationalExpr cacheNamespace
          .instantiationMap arena branch mapFact mapSelection
        arena := nextArena
        selectedBranches := selectedBranches.push selected
      let (finalArena, selected) ← arena.pushPrimitiveSelection dynamicSelection matrixType environment
        deriveSchema selectedBranches
      let expression ← finalArena.indexedExpr selected
      let finalArena ← finalArena.rememberIndexedExpr expression
      pure (finalArena, expression)

/-- Select one matrix family as application of an ordered direct family table.  Each branch family
is first reindexed onto the output lane selector; the branch-table binder is then substituted by
the executable selector, preserving both dimensions without a legacy choice node. -/
def selectUniformMatrixFamiliesWithLaneBinders
    (scopeKey : ScopeTemplateKey)
    (node : Nat)
    (selection : OperationalIntegerFact)
    (matrixType : MatrixTypeExpr)
    (declaredCount : IntExpr)
    (expectedCount : Nat)
    (branches : List OperationalFact)
    (branchLaneBinders : List IndexVariable)
    (environment : ParamEnvironment)
    (_deriveSchema : OperationalExprArena → ParamEnvironment → OperationalExprId →
      OperationalExprEvaluationState →
      Except OperationalError (OperationalMatrixFact × OperationalExprEvaluationState))
    (arena : OperationalExprArena) :
    Except OperationalError (OperationalExprArena × OperationalFact) := do
  if expectedCount = 0 || branches.isEmpty || branches.length != branchLaneBinders.length then
    throw (.invalidCount node expectedCount)
  let outputLane : OperationalValueOrigin := .local scopeKey { node, port := 0 }
  let outputSelection := DynamicSelectionIdentity.fromDeclaredCount outputLane declaredCount
  let mut arena := arena
  let mut normalizedBranches : Array OperationalFact := #[]
  for (branch, sourceBinder) in branches.zip branchLaneBinders do
    let expression ← match branch with
      | expression@{ payload := .directValue _, .. } => pure expression
      | { payload := .matrix root, .. } => throw (.unsupportedOperationalExpr root)
      | { payload := .scalar _, .. } => throw (.operandNotMatrix node { node, port := 0 })
    let value ← match arena.direct.valueAt? expression.payload.root with
      | some value => pure value
      | none => throw (.invalidOperationalExprRef expression.payload.root)
    if value.context != expression.context || value.payload.schema != .matrix matrixType then
      throw (.outputTypeMismatch node)
    if !expression.context.binders.contains sourceBinder then
      throw (.loopInputModeMismatch node 1)
    let map ← match dynamicIndexMap expression.context sourceBinder outputSelection.expression with
      | some map => pure map
      | none => throw (.loopInputModeMismatch node 1)
    let (nextArena, normalized) ← arena.reindexDirectMatrixFact map expression
    arena := nextArena
    normalizedBranches := normalizedBranches.push normalized
  let choiceCount := normalizedBranches.size
  let choiceCountExpression := IntExpr.constant (Int.ofNat choiceCount)
  let choiceBinder := packedDirectFamilyBinder scopeKey node choiceCountExpression
  let (nextArena, table) ← packDirectMatrixFamily scopeKey node environment choiceCountExpression arena
    normalizedBranches
  arena := nextArena
  if !table.context.binders.contains choiceBinder then
    throw (.unsupportedOperationalExpr node)
  let choiceMap ←
    if selection.lower == selection.upper then
      match closedStaticIndexMap environment table.context choiceBinder selection.lower.toNat with
      | some map => pure map
      | none => throw (.unsupportedOperationalExpr node)
    else
      let choiceSelection := DynamicSelectionIdentity.fromOrigin selection.origin choiceCount
      match dynamicIndexMap table.context choiceBinder choiceSelection.expression with
      | some map => pure map
      | none => throw (.unsupportedOperationalExpr node)
  let (finalArena, selected) ← arena.reindexDirectMatrixFact choiceMap table
  rebindOperationalFact { node, port := 0 } finalArena selected

def joinOperationalScalarFacts
    (node : Nat) : OperationalScalarFact → OperationalScalarFact →
    Except OperationalError OperationalScalarFact
  | .integer left, .integer right => pure (.integer {
      left with
      lower := min left.lower right.lower
      upper := max left.upper right.upper
      lowerExpression := .minimum left.lowerExpression right.lowerExpression
      upperExpression := .maximum left.upperExpression right.upperExpression
    })
  | .boolean, .boolean => pure .boolean
  | .real, .real => pure .real
  | .trapdoor left, .trapdoor right =>
      if left == right then pure (.trapdoor left) else throw (.unsupportedOperationalExpr node)
  | .bytes left, .bytes right =>
      if left == right then pure (.bytes left) else throw (.unsupportedOperationalExpr node)
  | .typedBlob left, .typedBlob right =>
      if left == right then pure (.typedBlob left) else throw (.unsupportedOperationalExpr node)
  | .unknown left, .unknown right =>
      if left == right then pure (.unknown left) else throw (.unsupportedOperationalExpr node)
  | _, _ => throw (.unsupportedOperationalExpr node)

def OperationalExprArena.scalarAbstract
    (arena : OperationalExprArena) (root : Nat) : Nat →
    Except OperationalError OperationalScalarFact
  | 0 => throw (.unsupportedOperationalExpr root)
  | fuel + 1 =>
      match arena.scalarNodes[root]? with
      | none => throw (.invalidOperationalExprRef root)
      | some (.concrete fact) => pure fact
      | some (.primitive _ _ result) => pure result
      | some (.selectShared _ _ _ representative) => arena.scalarAbstract representative fuel
      | some (.selectExact _ branches) => do
          let first ← match branches[0]? with
            | some branch => arena.scalarAbstract branch fuel
            | none => throw (.unsupportedOperationalExpr root)
          branches.extract 1 branches.size |>.foldlM (fun accumulated branch => do
            let next ← arena.scalarAbstract branch fuel
            joinOperationalScalarFacts root accumulated next) first

def scalarFactRoot
    (arena : OperationalExprArena)
    (fact : OperationalFact) : Except OperationalError (OperationalExprArena × Nat) :=
  arena.pushOperationalScalarFact fact

partial def installSharedScalarProvenance
    (arena : OperationalExprArena)
    (binder : FamilyTemplateBinder)
    (selection : DynamicSelectionIdentity)
    (subject : WireRef)
    (root fuel : Nat) : Except OperationalError (OperationalExprArena × Nat) := do
  if fuel = 0 then throw (.unsupportedOperationalExpr root)
  match arena.scalarNodes[root]? with
  | none => throw (.invalidOperationalExprRef root)
  | some (.concrete fact) =>
      pure (arena.pushScalarConcrete
        (indexScalarFact binder selection subject fact))
  | some (.primitive kind arguments result) =>
      pure (arena.pushScalar (.primitive kind arguments
        (indexScalarFact binder selection subject result)))
  | some (.selectExact domain branches) => do
      let (arena, branches) ← branches.foldlM (fun (arena, mapped) branch => do
        let (arena, branch) ← installSharedScalarProvenance arena binder selection subject branch
          (fuel - 1)
        pure (arena, mapped.push branch)) (arena, #[])
      pure (arena.pushScalar (.selectExact domain branches))
  | some (.selectShared domain innerBinder innerSubject representative) => do
      let (arena, representative) ← installSharedScalarProvenance arena binder selection subject
        representative (fuel - 1)
      pure (arena.pushScalar (.selectShared domain innerBinder innerSubject representative))

/-- Rebuild a Shared scalar only after installing the domain's indexed provenance on the produced
representative and checking the same invariant used at family construction. -/
def pushCheckedSharedScalar
    (arena : OperationalExprArena)
    (domain : SelectionDomainId)
    (binder : FamilyTemplateBinder)
    (subject : WireRef)
    (representative : Nat) : Except OperationalError (OperationalExprArena × Nat) := do
  let (arena, representative) ← installSharedScalarProvenance arena binder domain.identity subject
    representative (arena.scalarNodes.size + 1)
  if !scalarHasCheckedSharedTemplate arena binder domain.identity subject representative
      (arena.scalarNodes.size + 1) then
    throw (.unsupportedOperationalExpr representative)
  pure (arena.pushScalar (.selectShared domain binder subject representative))

partial def mapScalarExprPointwise
    (kind : OperationalScalarPrimitiveKind)
    (transfer : OperationalScalarFact → Except OperationalError OperationalScalarFact)
    (arena : OperationalExprArena)
    (root : Nat)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × Nat) := do
  if fuel = 0 then throw (.unsupportedOperationalExpr root)
  match arena.scalarNodes[root]? with
  | none => throw (.invalidOperationalExprRef root)
  | some (.concrete fact) => pure (arena.pushScalarConcrete (← transfer fact))
  | some (.primitive ..) =>
      let result ← transfer (← arena.scalarAbstract root fuel)
      pure (arena.pushScalar (.primitive kind #[root] result))
  | some (.selectExact domain branches) => do
      let (arena, branches) ← branches.foldlM (fun (arena, mapped) branch => do
        let (arena, branch) ← mapScalarExprPointwise kind transfer arena branch (fuel - 1)
        pure (arena, mapped.push branch)) (arena, #[])
      pure (arena.pushScalar (.selectExact domain branches))
  | some (.selectShared domain binder subject representative) => do
      let (arena, representative) ←
        mapScalarExprPointwise kind transfer arena representative (fuel - 1)
      pushCheckedSharedScalar arena domain binder subject representative

partial def zipScalarExprPointwise
    (kind : OperationalScalarPrimitiveKind)
    (transfer : OperationalScalarFact → OperationalScalarFact →
      Except OperationalError OperationalScalarFact)
    (arena : OperationalExprArena)
    (left right : Nat)
    (fuel : Nat) : Except OperationalError (OperationalExprArena × Nat) := do
  if fuel = 0 then throw (.unsupportedOperationalExpr left)
  let leftNode ← match arena.scalarNodes[left]? with
    | some node => pure node
    | none => throw (.invalidOperationalExprRef left)
  let rightNode ← match arena.scalarNodes[right]? with
    | some node => pure node
    | none => throw (.invalidOperationalExprRef right)
  match leftNode, rightNode with
  | .selectExact leftDomain leftBranches, .selectExact rightDomain rightBranches =>
      if leftDomain == rightDomain && leftBranches.size == rightBranches.size then do
        let (arena, branches) ← leftBranches.zip rightBranches |>.foldlM
          (fun (arena, mapped) pair => do
            let (arena, branch) ← zipScalarExprPointwise kind transfer arena pair.1 pair.2 (fuel - 1)
            pure (arena, mapped.push branch)) (arena, #[])
        pure (arena.pushScalar (.selectExact leftDomain branches))
      else
        let result ← transfer (← arena.scalarAbstract left fuel)
          (← arena.scalarAbstract right fuel)
        pure (arena.pushScalar (.primitive kind #[left, right] result))
  | .selectShared leftDomain leftBinder leftSubject leftRepresentative,
      .selectShared rightDomain _ _ rightRepresentative =>
      if leftDomain == rightDomain then do
        let (arena, representative) ← zipScalarExprPointwise kind transfer arena
          leftRepresentative rightRepresentative (fuel - 1)
        pushCheckedSharedScalar arena leftDomain leftBinder leftSubject representative
      else do
        let (arena, representative) ← zipScalarExprPointwise kind transfer arena
          leftRepresentative right (fuel - 1)
        pushCheckedSharedScalar arena leftDomain leftBinder leftSubject representative
  | .selectExact leftDomain leftBranches, .selectShared rightDomain _ _ rightRepresentative =>
      if leftDomain == rightDomain then do
        if leftBranches.size != leftDomain.count then
          throw (.unsupportedOperationalExpr left)
        let (arena, branches) ← leftBranches.foldlM (fun (arena, mapped) branch => do
          let (arena, branch) ← zipScalarExprPointwise kind transfer arena branch
            rightRepresentative (fuel - 1)
          pure (arena, mapped.push branch)) (arena, #[])
        pure (arena.pushScalar (.selectExact leftDomain branches))
      else do
        let (arena, branches) ← leftBranches.foldlM (fun (arena, mapped) branch => do
          let (arena, branch) ← zipScalarExprPointwise kind transfer arena branch right
            (fuel - 1)
          pure (arena, mapped.push branch)) (arena, #[])
        pure (arena.pushScalar (.selectExact leftDomain branches))
  | .selectShared leftDomain leftBinder leftSubject leftRepresentative,
      .selectExact rightDomain rightBranches =>
      if leftDomain == rightDomain then do
        if rightBranches.size != rightDomain.count then
          throw (.unsupportedOperationalExpr right)
        let (arena, branches) ← rightBranches.foldlM (fun (arena, mapped) branch => do
          let (arena, branch) ← zipScalarExprPointwise kind transfer arena leftRepresentative
            branch (fuel - 1)
          pure (arena, mapped.push branch)) (arena, #[])
        pure (arena.pushScalar (.selectExact rightDomain branches))
      else do
        let (arena, representative) ← zipScalarExprPointwise kind transfer arena
          leftRepresentative right (fuel - 1)
        pushCheckedSharedScalar arena leftDomain leftBinder leftSubject representative
  | .selectExact domain branches, _ => do
      let (arena, branches) ← branches.foldlM (fun (arena, mapped) branch => do
        let (arena, branch) ← zipScalarExprPointwise kind transfer arena branch right (fuel - 1)
        pure (arena, mapped.push branch)) (arena, #[])
      pure (arena.pushScalar (.selectExact domain branches))
  | .selectShared domain binder subject representative, _ => do
      let (arena, representative) ← zipScalarExprPointwise kind transfer arena representative right
        (fuel - 1)
      pushCheckedSharedScalar arena domain binder subject representative
  | _, .selectExact domain branches => do
      let (arena, branches) ← branches.foldlM (fun (arena, mapped) branch => do
        let (arena, branch) ← zipScalarExprPointwise kind transfer arena left branch (fuel - 1)
        pure (arena, mapped.push branch)) (arena, #[])
      pure (arena.pushScalar (.selectExact domain branches))
  | _, .selectShared domain binder subject representative => do
      let (arena, representative) ← zipScalarExprPointwise kind transfer arena left representative
        (fuel - 1)
      pushCheckedSharedScalar arena domain binder subject representative
  | _, _ =>
      let result ← transfer (← arena.scalarAbstract left fuel) (← arena.scalarAbstract right fuel)
      pure (arena.pushScalar (.primitive kind #[left, right] result))

def finishIndexedScalar
    (arena : OperationalExprArena)
    (root : Nat) : Except OperationalError (OperationalExprArena × OperationalFact) := do
  let expression ← arena.indexedScalar root
  let arena ← arena.rememberIndexedScalar expression
  pure (arena, expression)


end Mxx.Certificate
