import Mxx.Certificate.Identity

namespace Mxx.Certificate

/-- A free family, loop, or runtime-selection binder.  The exact owner and cardinality prevent
accidental positional alignment of equally numbered binders from different scopes. -/
structure IndexVariable where
  owner : CoreNodeRef
  slot : Nat
  count : IntExpr
  deriving BEq, DecidableEq, Repr

/-- Immutable identity of a gather lookup.  `indices` is the exact executable integer-family
producer occurrence, so two consumers of that one producer share the identity, while equal
values emitted by different producer wires do not.  Selected source families remain in matrix
origins and provenance, not lookup correlation identity. -/
inductive GatherProgramInstanceKey where
  | temporary
  | workflowStage (stage : StageId)
  | ideal
  | requirement (index : Nat)
  | standalone (checkedProgramOrdinal : Nat)
  deriving BEq, DecidableEq, Repr

/-- Structural scope path for a runtime gather producer.  This deliberately lives below the
operational checker layer so an index expression can retain an exact producer identity without
serializing a scope into a stage name or relying on a hash. -/
inductive GatherScopeTemplateKey where
  | root (program : GatherProgramInstanceKey)
  | callBody (parent : GatherScopeTemplateKey) (callNode : Nat)
  | parallelBody (parent : GatherScopeTemplateKey) (loopNode : Nat)
  | sequentialBody (parent : GatherScopeTemplateKey) (loopNode : Nat)
  deriving BEq, DecidableEq, Repr

structure GatherLookupWire where
  scope : GatherScopeTemplateKey
  node : Nat
  port : Nat
  deriving BEq, DecidableEq, Repr

structure GatherLookupOwner where
  indices : GatherLookupWire
  deriving BEq, DecidableEq, Repr

/-- Structural hash for owner-keyed executable gather lookup registries.  This follows the full
prepared-program scope path and producer wire, so numeric node/slot coincidences cannot merge
owners from different nested scopes. -/
private def gatherProgramInstanceHash : GatherProgramInstanceKey → UInt64
  | .temporary => 1
  | .workflowStage stage => 3 ^^^ hash stage.name
  | .ideal => 5
  | .requirement index => 7 ^^^ UInt64.ofNat index
  | .standalone ordinal => 11 ^^^ UInt64.ofNat ordinal

private def gatherScopeHash : GatherScopeTemplateKey → UInt64
  | .root program => 13 ^^^ gatherProgramInstanceHash program
  | .callBody parent node => (17 ^^^ gatherScopeHash parent) ^^^ UInt64.ofNat node
  | .parallelBody parent node => (19 ^^^ gatherScopeHash parent) ^^^ UInt64.ofNat node
  | .sequentialBody parent node => (23 ^^^ gatherScopeHash parent) ^^^ UInt64.ofNat node

instance : Hashable GatherLookupOwner where
  hash owner := ((29 ^^^ gatherScopeHash owner.indices.scope) ^^^ UInt64.ofNat owner.indices.node) ^^^
    UInt64.ofNat owner.indices.port

/-- Symbolic selection indices.  Dynamic selection is function application, never an indicator
sum over every lane.  A gather retains its immutable lookup-producer identity in the correlation
key itself: `sourceCount` carries the gathered result's codomain, while `position` carries the
lookup-lane domain independently. -/
inductive IndexExpr where
  | constant (value : Nat)
  | variable (value : IndexVariable)
  | offset (base : IndexExpr) (amount : Int)
  | gather (owner : GatherLookupOwner) (sourceCount : IntExpr) (position : IndexExpr)
  deriving BEq, DecidableEq, Repr

/-- Optional request-local cache of a *closed* gather lookup.  This is deliberately separate
from `IndexExpr`: the expression carries correlation identity, while a table only caches a
fully materialized integer-family producer for closed evaluation.  It is not the general
semantic payload for a runtime-dependent integer computation; that remains owned by the indexed
scalar carrier.  Repeated source lanes are retained in order. -/
structure GatherLookupTable where
  owner : GatherLookupOwner
  sourceCount : IntExpr
  positionCount : IntExpr
  sourceIndices : Array Nat
  deriving BEq, DecidableEq, Repr

/-- A cache may be structurally nonempty before executable indexed-integer registration verifies
its declared domains and entries.  This predicate is intentionally not an admission proof and is
never used to select an owner from the registry. -/
def GatherLookupTable.cacheShapeValid (table : GatherLookupTable) : Bool :=
  !table.sourceIndices.isEmpty

abbrev GatherLookupRegistry := Array GatherLookupTable

/-- Lookup entries are selected only when the exact owner occurs once in the whole registry.
Even an invalid duplicate is a collision: filtering it out would let a forged stale cache alter
which producer identity the checker accepts.  Registration and numeric evaluation remain
unavailable until the executable indexed-integer carrier validates the domain expressions. -/
def GatherLookupRegistry.lookupExact
    (registry : GatherLookupRegistry)
    (owner : GatherLookupOwner) : Option GatherLookupTable :=
  match registry.toList.filter (fun table => table.owner == owner) with
  | [table] => some table
  | _ => none

structure IndexContext where
  binders : Array IndexVariable
  deriving BEq, DecidableEq, Repr

private def distinctVariables : List IndexVariable → Bool
  | [] => true
  | binder :: rest => !rest.contains binder && distinctVariables rest

private def sameBinderSlot (left right : IndexVariable) : Bool :=
  left.owner == right.owner && left.slot == right.slot

private def hasConflictingBinderCount : List IndexVariable → Bool
  | [] => false
  | binder :: rest =>
      rest.any (fun candidate => sameBinderSlot binder candidate && binder.count != candidate.count) ||
        hasConflictingBinderCount rest

private def staticallyPositiveCount : IntExpr → Bool
  | .constant value => 0 < value
  | _ => true

def emptyContext : IndexContext := { binders := #[] }

def extendContext (context : IndexContext) (binder : IndexVariable) : Option IndexContext :=
  if context.binders.any (sameBinderSlot binder) then none
  else some { binders := context.binders.push binder }

def validateContext (context : IndexContext) : Bool :=
  distinctVariables context.binders.toList && !hasConflictingBinderCount context.binders.toList &&
    context.binders.all (staticallyPositiveCount ·.count)

private def IndexContext.contains (context : IndexContext) (binder : IndexVariable) : Bool :=
  context.binders.contains binder

/-- All binders on which an index expression depends.  This is structural: it neither
normalizes arithmetic nor identifies independently-owned selectors. -/
def IndexExpr.freeVariables : IndexExpr → List IndexVariable
  | .constant _ => []
  | .variable value => [value]
  | .offset base _ => base.freeVariables
  /- `sourceCount` is the result-domain witness of the exact gathered family, not an independently
  supplied runtime selector.  The owner carries its producer identity; only `position` must be
  available in the destination context. -/
  | .gather _ _ position => position.freeVariables

def indexExpressionInBounds (context : IndexContext) (expression : IndexExpr) : Bool :=
  expression.freeVariables.all context.contains

/-- The statically known half-open range of an index expression.  A variable ranges over its full
declared domain, so an offset is valid only when every possible lane remains in bounds. -/
private def staticIndexRange : IndexExpr → Option (Int × Int)
  | .constant value => some (Int.ofNat value, Int.ofNat value + 1)
  | .variable binder => match binder.count with
      | .constant count => if 0 < count then some (0, count) else none
      | _ => none
  | .offset base amount => do
      let (lower, upper) ← staticIndexRange base
      some (lower + amount, upper + amount)
  | .gather _ sourceCount _ => match sourceCount with
      | .constant count => if 0 < count then some (0, count) else none
      | _ => none

/-- The exact result domain retained by an index expression without evaluating parameters.
Offsets other than zero do not preserve a complete source domain.  Gather results retain the
source domain exactly; the position is checked separately. -/
private def exactIndexDomain : IndexExpr → Option IntExpr
  | .constant _ => none
  | .variable binder => some binder.count
  | .offset base amount => if amount == 0 then exactIndexDomain base else none
  | .gather _ sourceCount _ => some sourceCount

/-- Split an additive integer expression into its non-constant structural part and its accumulated
constant.  This is deliberately not an arithmetic normalizer: it proves only offsets visible in
the `IntExpr` syntax and rejects all other symbolic inequalities. -/
private def splitAdditiveConstant : IntExpr → IntExpr × Int
  | .constant value => (.constant 0, value)
  | .add left right =>
      let (leftBase, leftConstant) := splitAdditiveConstant left
      let (rightBase, rightConstant) := splitAdditiveConstant right
      match leftBase, rightBase with
      | .constant 0, base => (base, leftConstant + rightConstant)
      | base, .constant 0 => (base, leftConstant + rightConstant)
      | _, _ => (.add leftBase rightBase, leftConstant + rightConstant)
  | expression => (expression, 0)

/-- Prove the complete interval `[offset, offset + sourceCount)` fits within `limit` using only
explicit additive constants.  This accepts symbolic ZipOffset-style maps such as
`N -> N + 2`, but never assumes an unproved relationship between symbolic parameters. -/
private def symbolicOffsetWithin (limit sourceCount : IntExpr) (offset : Int) : Bool :=
  if offset < 0 then false
  else
    let (limitBase, limitConstant) := splitAdditiveConstant limit
    let (sourceBase, sourceConstant) := splitAdditiveConstant sourceCount
    limitBase == sourceBase && sourceConstant + offset <= limitConstant

private def variableOffset : IndexExpr → Option (IndexVariable × Int)
  | .variable binder => some (binder, 0)
  | .offset base amount => do
      let (binder, baseAmount) ← variableOffset base
      some (binder, baseAmount + amount)
  | _ => none

/-- Check every statically decidable lane, not just one representative.  For symbolic limits the
expression must retain an exact structural domain; an arbitrary non-variable expression is never
accepted merely because its runtime range is unknown. -/
private def staticallyWithin (limit : IntExpr) (expression : IndexExpr) : Bool :=
  match limit, staticIndexRange expression with
  | .constant bound, some (lower, upper) => 0 < bound && 0 ≤ lower && upper ≤ bound
  | .constant _, none => false
  | _, _ => match expression with
      | .variable binder => binder.count == limit
      | .offset _ _ => match variableOffset expression with
          | some (binder, offset) => symbolicOffsetWithin limit binder.count offset
          | none => false
      | .gather _ sourceCount _ => sourceCount == limit
      | .constant _ => false

private structure ClosedIndexWitness where
  binder : IndexVariable
  assignment : IndexExpr
  bound : Nat
  deriving BEq, DecidableEq, Repr

/-- Capture-free substitution from source-context binders to target-context expressions. -/
structure IndexMap where
  source : IndexContext
  destination : IndexContext
  assignments : Array IndexExpr
  /-- A fixed lane admitted only after its symbolic binder count was closed in a concrete
  parameter environment.  Ordinary maps leave this absent and remain subject to `validate`. -/
  closedIndex : Option ClosedIndexWitness := none
  deriving BEq, DecidableEq, Repr

private def lookupAssignment : List IndexVariable → List IndexExpr → IndexVariable → Option IndexExpr
  | [], _, _ => none
  | _, [], _ => none
  | binder :: remainingBinders, expression :: remainingExpressions, sought =>
      if binder == sought then some expression
      else lookupAssignment remainingBinders remainingExpressions sought

private def IndexMap.lookup? (map : IndexMap) (binder : IndexVariable) : Option IndexExpr :=
  lookupAssignment map.source.binders.toList map.assignments.toList binder

/-- Retrieve a validated map assignment by its exact source binder.  Consumers that transport
indexed payloads must not reconstruct this association positionally. -/
def IndexMap.assignmentFor (map : IndexMap) (binder : IndexVariable) : Option IndexExpr :=
  map.lookup? binder

private def assignmentsInBounds : List IndexVariable → List IndexExpr → IndexContext → Bool
  | [], [], _ => true
  | binder :: binders, expression :: expressions, destination =>
      indexExpressionInBounds destination expression && staticallyWithin binder.count expression &&
        assignmentsInBounds binders expressions destination
  | _, _, _ => false

def IndexMap.validate (map : IndexMap) : Bool :=
  validateContext map.source && validateContext map.destination &&
    map.assignments.size == map.source.binders.size &&
    assignmentsInBounds map.source.binders.toList map.assignments.toList map.destination

private def closedIndexAssignmentsInBounds
    (closedBinder : IndexVariable)
    (closedAssignment : IndexExpr)
    (closedBound : Nat) : List IndexVariable → List IndexExpr → IndexContext → Bool
  | [], [], _ => true
  | binder :: binders, expression :: expressions, destination =>
      (if binder == closedBinder then
        expression == closedAssignment &&
          match expression with
          | .constant lane => lane < closedBound
          | .variable _ => true
          | _ => false
      else indexExpressionInBounds destination expression && staticallyWithin binder.count expression) &&
        closedIndexAssignmentsInBounds closedBinder closedAssignment closedBound binders expressions destination
  | _, _, _ => false

/-- Validate one explicitly environment-closed substitution.  This does not relax
`IndexMap.validate`: the private witness fixes its exact source binder and assignment, while every
other assignment remains subject to the ordinary static source-domain proof. -/
private def closedIndexTransportValid (map : IndexMap) : Bool :=
  match map.closedIndex with
  | some { binder, assignment, bound } =>
      validateContext map.source && validateContext map.destination && 0 < bound &&
        map.source.binders.contains binder &&
        map.assignments.size == map.source.binders.size &&
        closedIndexAssignmentsInBounds binder assignment bound map.source.binders.toList
          map.assignments.toList map.destination
  | none => false

def IndexMap.transportValid (map : IndexMap) : Bool := map.validate || closedIndexTransportValid map

/-- A direct-carrier context lift introduces free destination binders but has no source binder to
substitute in a fixed leaf.  It must survive adjacent-map composition so a later get can still
specialize the owner-bearing identity installed by loop closure. -/
def IndexMap.isDirectCarrierContextLift (map : IndexMap) : Bool :=
  map.source.binders.isEmpty && map.assignments.isEmpty

private def reindexUnchecked (map : IndexMap) : IndexExpr → Option IndexExpr
  | .constant value => some (.constant value)
  | .variable binder => map.lookup? binder
  | .offset base amount => return .offset (← reindexUnchecked map base) amount
  | .gather owner sourceCount position =>
      return .gather owner sourceCount (← reindexUnchecked map position)

def reindex (map : IndexMap) (expression : IndexExpr) : Option IndexExpr :=
  if map.transportValid then reindexUnchecked map expression else none

def composeIndexMap (first second : IndexMap) : Option IndexMap := do
  if !first.validate || !second.validate || first.destination != second.source then none
  let assignments ← first.assignments.toList.mapM (reindex second)
  let composed : IndexMap := {
    source := first.source
    destination := second.destination
    assignments := assignments.toArray
  }
  if composed.validate then some composed else none

/-- Runtime values for free index atoms.  Keys are complete `IndexExpr` values, rather than
numeric slots, so separately owned selectors cannot be conflated during evaluation. -/
abbrev IndexValueEnvironment := List (IndexExpr × Int)

private def lookupIndexValue (expression : IndexExpr) : IndexValueEnvironment → Option Int
  | [] => none
  | (candidate, value) :: remaining =>
      if candidate == expression then some value else lookupIndexValue expression remaining

private def evaluatedIndexRange
    (parameters : Mxx.Ir.ParamEnvironment) : IndexExpr → Option (Int × Int)
  | .constant value => some (value, value + 1)
  | .variable binder => do
      let count ← binder.count.evaluate parameters
      if count <= 0 then none else some (0, count)
  | .offset base amount => do
      let (lower, upper) ← evaluatedIndexRange parameters base
      some (lower + amount, upper + amount)
  | .gather _ sourceCount _ => do
      let count ← sourceCount.evaluate parameters
      if count <= 0 then none else some (0, count)

private def valueInRange (value lower upper : Int) : Bool :=
  lower <= value && value < upper

private def evaluateIndexExprUnchecked
    (parameters : Mxx.Ir.ParamEnvironment)
    (environment : IndexValueEnvironment) : IndexExpr → Option Int
  | .constant value => some value
  | .variable binder => do
      let value ← lookupIndexValue (.variable binder) environment
      let (lower, upper) ← evaluatedIndexRange parameters (.variable binder)
      if valueInRange value lower upper then some value else none
  | .offset base amount => return (← evaluateIndexExprUnchecked parameters environment base) + amount
  | .gather _ _ _ => none

/-- Evaluate an index expression only in a context that owns every free index atom.  A gather is
looked up by its full structural identity and position; it is
never lowered to an `IntExpr` or identified with a similarly numbered foreign selector. -/
def evaluateIndexExpr
    (parameters : Mxx.Ir.ParamEnvironment)
    (context : IndexContext)
    (environment : IndexValueEnvironment)
    (expression : IndexExpr) : Option Int :=
  if !validateContext context || !indexExpressionInBounds context expression then none
  else evaluateIndexExprUnchecked parameters environment expression

/-- Lossless integer parameters used by indexed operational facts.  `index` retains the complete
owner-bearing `IndexExpr`, including gathers, instead of projecting it into the executable
`IntExpr` language. -/
inductive IndexedParameterExpr where
  | ir (value : IntExpr)
  | index (value : IndexExpr)
  | add (left right : IndexedParameterExpr)
  | subtract (left right : IndexedParameterExpr)
  | multiply (left right : IndexedParameterExpr)
  | divide (left right : IndexedParameterExpr)
  | roundDivide (left right : IndexedParameterExpr)
  | log2Ceil (value : IndexedParameterExpr)
  deriving BEq, DecidableEq, Repr

private def reindexIndexedParameterExprUnchecked
    (map : IndexMap) : IndexedParameterExpr → Option IndexedParameterExpr
  | .ir value => some (.ir value)
  | .index value => return .index (← reindexUnchecked map value)
  | .add left right => do
      let left ← reindexIndexedParameterExprUnchecked map left
      let right ← reindexIndexedParameterExprUnchecked map right
      return .add left right
  | .subtract left right => do
      let left ← reindexIndexedParameterExprUnchecked map left
      let right ← reindexIndexedParameterExprUnchecked map right
      return .subtract left right
  | .multiply left right => do
      let left ← reindexIndexedParameterExprUnchecked map left
      let right ← reindexIndexedParameterExprUnchecked map right
      return .multiply left right
  | .divide left right => do
      let left ← reindexIndexedParameterExprUnchecked map left
      let right ← reindexIndexedParameterExprUnchecked map right
      return .divide left right
  | .roundDivide left right => do
      let left ← reindexIndexedParameterExprUnchecked map left
      let right ← reindexIndexedParameterExprUnchecked map right
      return .roundDivide left right
  | .log2Ceil value => return .log2Ceil (← reindexIndexedParameterExprUnchecked map value)

/-- Capture-free indexed-parameter substitution.  This traverses every index atom, including
both source and position of gathers, and accepts only a validated source-to-destination map. -/
def IndexedParameterExpr.reindex
    (map : IndexMap)
    (expression : IndexedParameterExpr) : Option IndexedParameterExpr :=
  if map.transportValid then reindexIndexedParameterExprUnchecked map expression else none

def IndexedParameterExpr.evaluate
    (parameters : Mxx.Ir.ParamEnvironment)
    (context : IndexContext)
    (indices : IndexValueEnvironment) : IndexedParameterExpr → Option Int
  | .ir value => value.evaluate parameters
  | .index value => evaluateIndexExpr parameters context indices value
  | .add left right => return (← left.evaluate parameters context indices) +
    (← right.evaluate parameters context indices)
  | .subtract left right => return (← left.evaluate parameters context indices) -
    (← right.evaluate parameters context indices)
  | .multiply left right => return (← left.evaluate parameters context indices) *
    (← right.evaluate parameters context indices)
  | .divide left right => do
      let denominator ← right.evaluate parameters context indices
      if denominator = 0 then none else return (← left.evaluate parameters context indices) / denominator
  | .roundDivide left right => do
      let denominator ← right.evaluate parameters context indices
      if denominator = 0 then none
      else return Mxx.Ir.roundDiv (← left.evaluate parameters context indices) denominator
  | .log2Ceil value => return Mxx.Ir.log2Ceil (← value.evaluate parameters context indices)

/-- Substitute one source binder by a fixed lane and retain every other binder unchanged.  Static
family access uses this map instead of erasing a context position by convention. -/
def closedStaticIndexMap
    (environment : Mxx.Ir.ParamEnvironment)
    (source : IndexContext)
    (binder : IndexVariable)
    (lane : Nat) : Option IndexMap := do
  if !validateContext source || !source.binders.contains binder then none
  let bound ← binder.count.evaluate environment
  if bound <= 0 || lane >= bound.toNat then none
  let destination : IndexContext := {
    binders := source.binders.filter (· != binder)
  }
  let assignments := source.binders.map fun candidate =>
    if candidate == binder then .constant lane else .variable candidate
  let map : IndexMap := {
    source
    destination
    assignments
    closedIndex := some { binder, assignment := .constant lane, bound := bound.toNat }
  }
  if map.transportValid then some map else none

def sameIndexExpression (left right : IndexExpr) : Bool := left == right

private def extendAll (context : IndexContext) : List IndexVariable → Option IndexContext
  | [] => some context
  | binder :: remaining => do
      let next ← if context.binders.contains binder then some context
        else extendContext context binder
      extendAll next remaining

/-- Substitute one source binder by a runtime selector while retaining all unrelated source
binders.  Selector dependencies become part of the destination context, so dynamic family access
does not erase correlation with an enclosing loop or selection. -/
def dynamicIndexMap
    (source : IndexContext)
    (binder : IndexVariable)
    (selector : IndexExpr) : Option IndexMap := do
  if !validateContext source || !source.binders.contains binder then none
  let destination ← extendAll { binders := source.binders.filter (· != binder) }
    selector.freeVariables
  let assignments := source.binders.map fun candidate =>
    if candidate == binder then selector else .variable candidate
  let map : IndexMap := { source, destination, assignments }
  if map.validate then some map else none

/-- Substitute a symbolic family lane by a canonical variable selector only after both domains
close to the same positive size in the caller's parameter environment. -/
def closedDynamicIndexMap
    (environment : Mxx.Ir.ParamEnvironment)
    (source : IndexContext)
    (binder : IndexVariable)
    (selector : IndexExpr) : Option IndexMap := do
  if !validateContext source || !source.binders.contains binder then none
  let selectorBinder ← match selector with
    | .variable value => some value
    | _ => none
  let bound ← binder.count.evaluate environment
  let selectorBound ← selectorBinder.count.evaluate environment
  if bound <= 0 || selectorBound != bound then none
  let destination ← extendAll { binders := source.binders.filter (· != binder) } selector.freeVariables
  let assignments := source.binders.map fun candidate =>
    if candidate == binder then selector else .variable candidate
  let map : IndexMap := {
    source
    destination
    assignments
    closedIndex := some { binder, assignment := selector, bound := bound.toNat }
  }
  if map.transportValid then some map else none

/-- The pointwise context of two operands.  Binders retain first-occurrence order, so equal
selector variables remain correlated rather than being renamed into independent dimensions. -/
def mergeIndexContexts (left right : IndexContext) : Option IndexContext := do
  if !validateContext left || !validateContext right then none
  extendAll left right.binders.toList

/-- Merge any number of pointwise operand contexts in first-occurrence order.  This is the sole
context construction used by n-ary lifting, preserving a shared selector binder exactly once. -/
def mergeIndexContextsN (contexts : List IndexContext) : Option IndexContext :=
  contexts.foldlM mergeIndexContexts emptyContext

inductive IndexedStorage where
  | sharedTemplate
  | mappedTemplate
  | explicitTable
  deriving BEq, DecidableEq, Repr

/-- Storage is only an optimization representation; context and payload determine semantics. -/
structure IndexedFact (α : Type) where
  context : IndexContext
  payload : α
  storage : IndexedStorage
  deriving BEq, Repr

/-- Merge the semantic context and storage representation for a pointwise indexed operation. -/
def mergeIndexedFactShape {α β : Type}
    (left : IndexedFact α) (right : IndexedFact β) : Option (IndexContext × IndexedStorage) := do
  let context ← mergeIndexContexts left.context right.context
  let storage := match left.storage, right.storage with
    | .sharedTemplate, .sharedTemplate => .sharedTemplate
    | .explicitTable, .explicitTable =>
        if left.context == right.context then .explicitTable else .mappedTemplate
    | _, _ => .mappedTemplate
  some (context, storage)

/-- Merge the semantic context and storage tag for any arity.  The empty arity denotes a constant
template over the empty context; otherwise storage remains a representation-only join. -/
def mergeIndexedFactShapeN {α : Type}
    (facts : List (IndexedFact α)) : Option (IndexContext × IndexedStorage) := do
  let context ← mergeIndexContextsN (facts.map (·.context))
  let storage := match facts with
    | [] => .sharedTemplate
    | first :: remaining =>
        if first.storage == .sharedTemplate &&
            remaining.all (fun fact => fact.storage == .sharedTemplate) then
          .sharedTemplate
        else if first.storage == .explicitTable &&
            remaining.all (fun fact =>
              fact.storage == .explicitTable && fact.context == first.context) then
          .explicitTable
        else
          .mappedTemplate
  some (context, storage)

def IndexedFact.reindex {α : Type}
    (map : IndexMap)
    (mapPayload : IndexMap → α → Option α)
    (fact : IndexedFact α) : Option (IndexedFact α) := do
  if !map.transportValid || fact.context != map.source then none
  let storage := if fact.storage == .explicitTable && map.source != map.destination then
    .mappedTemplate
  else
    fact.storage
  some { context := map.destination, payload := ← mapPayload map fact.payload, storage }

/-- Lift a fixed-assignment primitive over two indexed templates.  The storage tag controls only
memoization representation; the primitive sees exactly the two payloads at the merged context. -/
def liftPointwise {α β γ : Type}
    (primitive : α → β → Option γ)
    (left : IndexedFact α)
    (right : IndexedFact β) : Option (IndexedFact γ) := do
  let (context, storage) ← mergeIndexedFactShape left right
  let payload ← primitive left.payload right.payload
  some { context, payload, storage }

/-- Lift one fixed-assignment primitive over an arbitrary number of indexed operands.  Primitive
semantics receives only payloads; all correlation is retained by the one merged context. -/
def liftPointwiseN {α β : Type}
    (primitive : List α → Option β)
    (facts : List (IndexedFact α)) : Option (IndexedFact β) := do
  let (context, storage) ← mergeIndexedFactShapeN facts
  let payload ← primitive (facts.map (·.payload))
  some { context, payload, storage }

private def fixtureOwner (node : Nat) : CoreNodeRef := {
  stage := ⟨"indexed-fixture"⟩
  scope := ⟨[]⟩
  node := ⟨node⟩
}

private def fixtureWire (node port : Nat := 0) : CoreWireRef := {
  stage := ⟨"indexed-fixture"⟩
  scope := ⟨[]⟩
  node := ⟨node⟩
  port
}

private def fixtureGatherWire (node port : Nat := 0) : GatherLookupWire := {
  scope := .root (.standalone 0)
  node
  port
}

private def fixtureGatherOwner : GatherLookupOwner := {
  indices := fixtureGatherWire 401
}

private def fixtureGather (source position : IndexExpr) : IndexExpr :=
  let sourceCount := match source with
    | .variable binder => binder.count
    | _ => .constant 1
  .gather fixtureGatherOwner sourceCount position

private def fixtureVariable (node slot count : Nat) : IndexVariable := {
  owner := fixtureOwner node
  slot
  count := .constant count
}

private def fixtureContext (binders : List IndexVariable) : IndexContext := {
  binders := binders.toArray
}

private def fixtureMap
    (source destination : IndexContext) (assignments : List IndexExpr) : IndexMap := {
  source
  destination
  assignments := assignments.toArray
}

private def fixturePayloadReindex (map : IndexMap) (value : IndexExpr) : Option IndexExpr :=
  reindex map value

example : validateContext emptyContext = true := by native_decide

example :
    let lane4 := fixtureVariable 0 0 4
    let lane8 := fixtureVariable 0 0 8
    validateContext (fixtureContext [lane4, lane8]) = false := by native_decide

example :
    let emptyLane := fixtureVariable 0 0 0
    validateContext (fixtureContext [emptyLane]) = false := by native_decide

example :
    let lane4 := fixtureVariable 0 0 4
    let lane8 := fixtureVariable 0 0 8
    extendContext (fixtureContext [lane4]) lane8 = none := by native_decide

example :
    let lane := fixtureVariable 0 0 4
    let source := fixtureContext [lane]
    let map := fixtureMap source emptyContext [.constant 3]
    reindex map (.variable lane) = some (.constant 3) := by native_decide

example :
    let lane := fixtureVariable 0 0 4
    let source := fixtureContext [lane]
    let map := fixtureMap source emptyContext [.constant 4]
    map.validate = false := by native_decide

example :
    let selector := fixtureVariable 1 0 8
    sameIndexExpression (.variable selector) (.variable selector) = true := by rfl

example :
    let left := fixtureVariable 1 0 8
    let right := fixtureVariable 2 0 8
    sameIndexExpression (.variable left) (.variable right) = false := by rfl

example :
    let lane := fixtureVariable 0 0 8
    let selector := fixtureVariable 1 0 8
    let source := fixtureContext [lane]
    let destination := fixtureContext [selector]
    let map := fixtureMap source destination [.variable selector]
    reindex map (.variable lane) = some (.variable selector) := by native_decide

example :
    let lane := fixtureVariable 0 0 8
    let selector := fixtureVariable 1 0 8
    let source := fixtureContext [lane]
    dynamicIndexMap source lane (.variable selector) =
      some (fixtureMap source (fixtureContext [selector]) [.variable selector]) := by native_decide

example :
    let lane := fixtureVariable 0 0 8
    let loop := fixtureVariable 1 0 6
    let source := fixtureContext [lane]
    let destination := fixtureContext [loop]
    let map := fixtureMap source destination [.offset (.variable loop) 2]
    reindex map (.variable lane) = some (.offset (.variable loop) 2) := by native_decide

example :
    let lane := fixtureVariable 0 0 8
    let loop := fixtureVariable 1 0 8
    let source := fixtureContext [lane]
    let destination := fixtureContext [loop]
    let map := fixtureMap source destination [.offset (.variable loop) 2]
    map.validate = false := by native_decide

example :
    let lane : IndexVariable := {
      owner := fixtureOwner 0
      slot := 0
      count := .parameter "param4"
    }
    let sourceIndex : IndexVariable := {
      owner := fixtureOwner 1
      slot := 0
      count := .parameter "param4"
    }
    let position : IndexVariable := {
      owner := fixtureOwner 2
      slot := 0
      count := .parameter "param4"
    }
    let map := fixtureMap (fixtureContext [lane]) (fixtureContext [sourceIndex, position]) [
      fixtureGather (.variable sourceIndex) (.variable position)
    ]
    map.validate = true := by
  native_decide

example :
    let lane : IndexVariable := {
      owner := fixtureOwner 0
      slot := 0
      count := .parameter "param4"
    }
    let sourceIndex : IndexVariable := {
      owner := fixtureOwner 1
      slot := 0
      count := .parameter "param8"
    }
    let position : IndexVariable := {
      owner := fixtureOwner 2
      slot := 0
      count := .parameter "param4"
    }
    let map := fixtureMap (fixtureContext [lane]) (fixtureContext [sourceIndex, position]) [
      fixtureGather (.variable sourceIndex) (.variable position)
    ]
    map.validate = false := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let loop := fixtureVariable 1 0 6
    let source := fixtureContext [lane]
    let destination := fixtureContext [loop]
    let map := fixtureMap source destination [.offset (.variable loop) 2]
    map.validate = true := by native_decide

example :
    let lane := fixtureVariable 0 0 8
    let sourceIndex := fixtureVariable 1 0 8
    let source := fixtureContext [lane]
    let destination := fixtureContext [sourceIndex]
    let gathered := fixtureGather (.variable sourceIndex) (.variable sourceIndex)
    let map := fixtureMap source destination [gathered]
    reindex map (.variable lane) = some gathered := by native_decide

example :
    let lane := fixtureVariable 0 0 4
    let sourceIndex := fixtureVariable 1 0 4
    let position := fixtureVariable 2 0 8
    let source := fixtureContext [lane]
    let destination := fixtureContext [sourceIndex, position]
    let gathered := fixtureGather (.variable sourceIndex) (.variable position)
    let map := fixtureMap source destination [gathered]
    map.validate = true := by native_decide

/-- Generic index environments cannot forge a gather result.  The lookup payload belongs to an
exact request-local owner rather than to a caller-supplied `(IndexExpr, Int)` binding. -/
example :
    let sourceIndex := fixtureVariable 1 0 4
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := fixtureGather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable position, 7),
      (gathered, 3)
    ] = none := by
  native_decide

/-- Gather result values are still bounded by the source-family codomain, even when their lookup
position is from a larger lane domain. -/
example :
    let sourceIndex := fixtureVariable 1 0 4
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := fixtureGather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable position, 7),
      (gathered, 4)
    ] = none := by
  native_decide

/-- Closed gather caches are keyed by the integer-family producer, not by a consumer-local
number.  Until executable indexed-integer registration validates the domain expressions, caches
are not evaluable; nevertheless duplicate owners (including stale invalid entries) are rejected
and the owner remains part of the symbolic correlation key. -/
example :
    let position := fixtureVariable 2 0 3
    let owner : GatherLookupOwner := {
      indices := fixtureGatherWire 11
    }
    let table : GatherLookupTable := {
      owner
      sourceCount := .constant 4
      positionCount := .constant 3
      sourceIndices := #[2, 2, 0]
    }
    let stale : GatherLookupTable := { table with sourceIndices := #[] }
    table.cacheShapeValid && !stale.cacheShapeValid &&
      (GatherLookupRegistry.lookupExact #[table] owner == some table) &&
      (GatherLookupRegistry.lookupExact #[table, stale] owner).isNone &&
      (IndexExpr.gather owner (.constant 3) (IndexExpr.variable position) !=
        IndexExpr.gather { owner with indices := fixtureGatherWire 12 }
          (.constant 3) (IndexExpr.variable position)) := by
  native_decide

example :
    let owner : GatherLookupOwner := {
      indices := fixtureGatherWire 11
    }
    let foreign : GatherLookupOwner := { owner with indices := fixtureGatherWire 12 }
    let table : GatherLookupTable := {
      owner
      sourceCount := .constant 2
      positionCount := .constant 2
      sourceIndices := #[0, 1]
    }
    (GatherLookupRegistry.lookupExact #[table] foreign).isNone := by
  native_decide

/-- Symbolic ZipOffset is accepted only when additive `IntExpr` structure proves the whole shifted
source interval is contained in the target binder domain. -/
example :
    let lane : IndexVariable := {
      owner := fixtureOwner 0
      slot := 0
      count := .add (.parameter "lane-count") (.constant 2)
    }
    let loop : IndexVariable := {
      owner := fixtureOwner 1
      slot := 0
      count := .parameter "lane-count"
    }
    let map := fixtureMap (fixtureContext [lane]) (fixtureContext [loop]) [
      .offset (.variable loop) 2
    ]
    map.validate = true := by native_decide

/-- A symbolic ZipOffset without a structural proof of the final upper bound is rejected. -/
example :
    let lane : IndexVariable := {
      owner := fixtureOwner 0
      slot := 0
      count := .add (.parameter "lane-count") (.constant 1)
    }
    let loop : IndexVariable := {
      owner := fixtureOwner 1
      slot := 0
      count := .parameter "lane-count"
    }
    let map := fixtureMap (fixtureContext [lane]) (fixtureContext [loop]) [
      .offset (.variable loop) 2
    ]
    map.validate = false := by native_decide

example :
    let lane := fixtureVariable 0 0 8
    let loop := fixtureVariable 1 0 6
    let source := fixtureContext [lane]
    let intermediate := fixtureContext [loop]
    let first := fixtureMap source intermediate [.offset (.variable loop) 2]
    let second := fixtureMap intermediate emptyContext [.constant 3]
    composeIndexMap first second = some (fixtureMap source emptyContext [.offset (.constant 3) 2]) := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let left : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .sharedTemplate
    }
    let right : IndexedFact Bool := {
      context := fixtureContext [lane]
      payload := true
      storage := .sharedTemplate
    }
    mergeIndexedFactShape left right =
      some (fixtureContext [lane], .sharedTemplate) := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let selector := fixtureVariable 1 0 4
    let left : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .explicitTable
    }
    let right : IndexedFact Bool := {
      context := fixtureContext [lane, selector]
      payload := true
      storage := .sharedTemplate
    }
    mergeIndexedFactShape left right =
      some (fixtureContext [lane, selector], .mappedTemplate) := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let other := fixtureVariable 1 0 8
    let source := fixtureContext [lane, other]
    (closedStaticIndexMap [] source lane 3).map (fun map =>
      map.destination == fixtureContext [other] &&
        reindex map (.variable lane) == some (.constant 3) &&
        reindex map (.variable other) == some (.variable other)) = some true := by
  native_decide

example :
    let lane : IndexVariable := {
      owner := fixtureOwner 20
      slot := 0
      count := .parameter "lane_count"
    }
    let selector : IndexVariable := {
      owner := fixtureOwner 21
      slot := 0
      count := .parameter "selector_count"
    }
    let environment : Mxx.Ir.ParamEnvironment := [
      ("lane_count", .integer 2), ("selector_count", .integer 2)
    ]
    (closedDynamicIndexMap environment (fixtureContext [lane]) lane (.variable selector)).map (fun map =>
      map.validate == false && map.transportValid &&
        reindex map (.variable lane) == some (.variable selector)) = some true := by
  native_decide

example :
    let lane : IndexVariable := {
      owner := fixtureOwner 22
      slot := 0
      count := .parameter "lane_count"
    }
    let outer : IndexVariable := {
      owner := fixtureOwner 23
      slot := 0
      count := .parameter "lane_count"
    }
    let environment : Mxx.Ir.ParamEnvironment := [("lane_count", .integer 2)]
    let first := fixtureMap (fixtureContext [outer]) (fixtureContext [lane]) [.variable lane]
    (closedStaticIndexMap environment (fixtureContext [lane]) lane 1).map (fun second =>
      match composeIndexMap first second with | none => true | some _ => false) = some true := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let other := fixtureVariable 1 0 4
    let fact : IndexedFact IndexExpr := {
      context := fixtureContext [lane]
      payload := .variable lane
      storage := .sharedTemplate
    }
    let map := fixtureMap (fixtureContext [other]) emptyContext [.constant 0]
    fact.reindex map fixturePayloadReindex = none := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let loop := fixtureVariable 1 0 6
    let source := fixtureContext [lane]
    let intermediate := fixtureContext [loop]
    let first := fixtureMap source intermediate [.offset (.variable loop) 2]
    let second := fixtureMap intermediate emptyContext [.constant 3]
    (match composeIndexMap first second with
    | some composed =>
        reindex composed (.variable lane) ==
          (reindex first (.variable lane)).bind (reindex second)
    | none => false) = true := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let selector := fixtureVariable 1 0 4
    let left : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .sharedTemplate
    }
    let right : IndexedFact Nat := {
      context := fixtureContext [lane, selector]
      payload := 3
      storage := .mappedTemplate
    }
    (liftPointwise (fun a b => some (a + b)) left right).map (fun result =>
      result.context == fixtureContext [lane, selector] && result.payload == 5 &&
      result.storage == .mappedTemplate) = some true := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let source := fixtureContext [lane]
    let expression : IndexedParameterExpr := .add (.ir (.constant 4)) (.index (.variable lane))
    let map := fixtureMap source emptyContext [.constant 3]
    expression.reindex map = some (.add (.ir (.constant 4)) (.index (.constant 3))) := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let selector := fixtureVariable 1 0 8
    let source := fixtureContext [lane]
    let expression : IndexedParameterExpr := .index (.variable lane)
    (dynamicIndexMap source lane (.variable selector)).bind (expression.reindex) =
      some (.index (.variable selector)) := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let loop := fixtureVariable 1 0 6
    let source := fixtureContext [lane]
    let destination := fixtureContext [loop]
    let map := fixtureMap source destination [.offset (.variable loop) 2]
    let expression : IndexedParameterExpr := .index (.variable lane)
    expression.reindex map = some (.index (.offset (.variable loop) 2)) := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 8
    let source := fixtureContext [lane]
    let destination := fixtureContext [sourceIndex, position]
    let gathered := fixtureGather (.variable sourceIndex) (.variable position)
    let map := fixtureMap source destination [gathered]
    let expression : IndexedParameterExpr := .multiply (.index (.variable lane)) (.ir (.constant 2))
    expression.reindex map = some (.multiply (.index gathered) (.ir (.constant 2))) := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 8
    let outer := fixtureVariable 3 0 8
    let source := fixtureContext [lane]
    let intermediate := fixtureContext [sourceIndex, position]
    let destination := fixtureContext [outer, position]
    let first := fixtureMap source intermediate [fixtureGather (.variable sourceIndex) (.variable position)]
    let second := fixtureMap intermediate destination [.variable outer, .variable position]
    let expression : IndexedParameterExpr := .index (.variable lane)
    (composeIndexMap first second).bind (expression.reindex) =
      some (.index (fixtureGather (.variable outer) (.variable position))) := by
  native_decide

example :
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := fixtureGather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable sourceIndex, 1),
      (.variable position, 2),
      (gathered, 6)
    ] = none := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let context := fixtureContext [lane]
    (IndexedParameterExpr.index (.variable lane)).evaluate [] context [
      (.variable lane, -1)
    ] = none := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let context := fixtureContext [lane]
    (IndexedParameterExpr.index (.variable lane)).evaluate [] context [
      (.variable lane, 8)
    ] = none := by
  native_decide

example :
    let lane : IndexVariable := {
      owner := fixtureOwner 0
      slot := 0
      count := .parameter "lane-count"
    }
    let context := fixtureContext [lane]
    (IndexedParameterExpr.index (.variable lane)).evaluate [
      ("lane-count", .integer 5)
    ] context [(.variable lane, 4)] = some 4 := by
  native_decide

example :
    let lane : IndexVariable := {
      owner := fixtureOwner 0
      slot := 0
      count := .parameter "lane-count"
    }
    let context := fixtureContext [lane]
    (IndexedParameterExpr.index (.variable lane)).evaluate [
      ("lane-count", .integer 5)
    ] context [(.variable lane, 5)] = none := by
  native_decide

example :
    let lane := fixtureVariable 0 0 8
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 10
    let source := fixtureContext [lane]
    let destination := fixtureContext [sourceIndex, position]
    let map := fixtureMap source destination [fixtureGather (.variable sourceIndex) (.variable position)]
    (IndexedParameterExpr.index (.variable lane)).reindex map =
      some (.index (fixtureGather (.variable sourceIndex) (.variable position))) := by
  native_decide

example :
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := fixtureGather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable sourceIndex, 1),
      (.variable position, -1),
      (gathered, 6)
    ] = none := by
  native_decide

example :
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := fixtureGather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable sourceIndex, 1),
      (.variable position, 2),
      (gathered, -1)
    ] = none := by
  native_decide

example :
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := fixtureGather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable sourceIndex, 1),
      (.variable position, 2),
      (gathered, 8)
    ] = none := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let other := fixtureVariable 1 0 4
    let left : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .explicitTable
    }
    let right : IndexedFact Nat := {
      context := fixtureContext [other]
      payload := 3
      storage := .explicitTable
    }
    (mergeIndexedFactShape left right).map (fun result => result.2) = some .mappedTemplate := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let left : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .explicitTable
    }
    let right : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 3
      storage := .explicitTable
    }
    (mergeIndexedFactShape left right).map (fun result => result.2) = some .explicitTable := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let other := fixtureVariable 1 0 4
    let first : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .explicitTable
    }
    let second : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 3
      storage := .explicitTable
    }
    let independent : IndexedFact Nat := {
      context := fixtureContext [other]
      payload := 5
      storage := .explicitTable
    }
    (mergeIndexedFactShapeN [first, second, independent]).map (fun result => result.2) =
      some .mappedTemplate := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let source := fixtureContext [lane]
    let fact : IndexedFact IndexExpr := {
      context := source
      payload := .variable lane
      storage := .explicitTable
    }
    let map := fixtureMap source emptyContext [.constant 3]
    (fact.reindex map fixturePayloadReindex).map (fun result =>
      result.context == emptyContext && result.payload == .constant 3 &&
        result.storage == .mappedTemplate) = some true := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let context := fixtureContext [lane]
    let fact : IndexedFact IndexExpr := {
      context
      payload := .variable lane
      storage := .explicitTable
    }
    let identityMap := fixtureMap context context [.variable lane]
    (fact.reindex identityMap fixturePayloadReindex).map (fun result =>
      result.context == context && result.payload == .variable lane &&
        result.storage == .explicitTable) = some true := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let first : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .explicitTable
    }
    let second : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 3
      storage := .explicitTable
    }
    (mergeIndexedFactShapeN [first, second]).map (fun result => result.2) =
      some .explicitTable := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let other := fixtureVariable 1 0 4
    let first : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .sharedTemplate
    }
    let second : IndexedFact Nat := {
      context := fixtureContext [other]
      payload := 3
      storage := .sharedTemplate
    }
    (mergeIndexedFactShapeN [first, second]).map (fun result => result.2) =
      some .sharedTemplate := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let first : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 2
      storage := .mappedTemplate
    }
    let second : IndexedFact Nat := {
      context := fixtureContext [lane]
      payload := 3
      storage := .mappedTemplate
    }
    (mergeIndexedFactShapeN [first, second]).map (fun result => result.2) =
      some .mappedTemplate := by
  native_decide

example :
    let left := fixtureVariable 0 0 4
    let selector := fixtureVariable 1 0 4
    let first : IndexedFact Nat := {
      context := fixtureContext [left]
      payload := 2
      storage := .sharedTemplate
    }
    let second : IndexedFact Nat := {
      context := fixtureContext [selector]
      payload := 3
      storage := .mappedTemplate
    }
    let third : IndexedFact Nat := {
      context := fixtureContext [left, selector]
      payload := 5
      storage := .explicitTable
    }
    (liftPointwiseN (fun values => some values.sum) [first, second, third]).map (fun result =>
      result.context == fixtureContext [left, selector] && result.payload == 10 &&
        result.storage == .mappedTemplate) = some true := by
  native_decide

example :
    let lane := fixtureVariable 0 0 4
    let source := fixtureContext [lane]
    let map := fixtureMap source emptyContext [.constant 3]
    let fact : IndexedFact IndexExpr := {
      context := source
      payload := .variable lane
      storage := .mappedTemplate
    }
    (fact.reindex map fixturePayloadReindex).map (fun result =>
      result.context == emptyContext && result.payload == .constant 3 &&
        result.storage == .mappedTemplate) = some true := by
  native_decide

end Mxx.Certificate
