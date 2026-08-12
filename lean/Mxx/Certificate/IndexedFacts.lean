import Mxx.Certificate.Identity

namespace Mxx.Certificate

/-- A free family, loop, or runtime-selection binder.  The exact owner and cardinality prevent
accidental positional alignment of equally numbered binders from different scopes. -/
structure IndexVariable where
  owner : CoreNodeRef
  slot : Nat
  count : IntExpr
  deriving BEq, DecidableEq, Repr

/-- Symbolic selection indices.  Dynamic selection is function application, never an indicator
sum over every lane.  In `gather source position`, `source` carries the gathered result's
codomain, while `position` carries the lookup-lane domain independently. -/
inductive IndexExpr where
  | constant (value : Nat)
  | variable (value : IndexVariable)
  | offset (base : IndexExpr) (amount : Int)
  | gather (source position : IndexExpr)
  deriving BEq, DecidableEq, Repr

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
  | .gather source position => source.freeVariables ++ position.freeVariables

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
  | .gather source _ => staticIndexRange source

/-- The exact result domain retained by an index expression without evaluating parameters.
Offsets other than zero do not preserve a complete source domain.  Gather results retain the
source domain exactly; the position is checked separately. -/
private def exactIndexDomain : IndexExpr → Option IntExpr
  | .constant _ => none
  | .variable binder => some binder.count
  | .offset base amount => if amount == 0 then exactIndexDomain base else none
  | .gather source _ => exactIndexDomain source

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
      | .gather source _ => exactIndexDomain source == some limit
      | .constant _ => false

/-- Capture-free substitution from source-context binders to target-context expressions. -/
structure IndexMap where
  source : IndexContext
  destination : IndexContext
  assignments : Array IndexExpr
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

private def reindexUnchecked (map : IndexMap) : IndexExpr → Option IndexExpr
  | .constant value => some (.constant value)
  | .variable binder => map.lookup? binder
  | .offset base amount => return .offset (← reindexUnchecked map base) amount
  | .gather source position =>
      return .gather (← reindexUnchecked map source) (← reindexUnchecked map position)

def reindex (map : IndexMap) (expression : IndexExpr) : Option IndexExpr :=
  if map.validate then reindexUnchecked map expression else none

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
  | .gather source _ => evaluatedIndexRange parameters source

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
  | .gather source position => do
      let _ ← evaluateIndexExprUnchecked parameters environment position
      let (sourceLower, sourceUpper) ← evaluatedIndexRange parameters source
      let value ← lookupIndexValue (.gather source position) environment
      if valueInRange value sourceLower sourceUpper then some value else none

/-- Evaluate an index expression only in a context that owns every free index atom.  A gather is
looked up by its full structural identity after both source and position have been evaluated; it is
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
  if map.validate then reindexIndexedParameterExprUnchecked map expression else none

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
def staticIndexMap (source : IndexContext) (binder : IndexVariable) (lane : Nat) : Option IndexMap := do
  if !validateContext source || !source.binders.contains binder then none
  let destination : IndexContext := {
    binders := source.binders.filter (· != binder)
  }
  let assignments := source.binders.map fun candidate =>
    if candidate == binder then .constant lane else .variable candidate
  let map : IndexMap := { source, destination, assignments }
  if map.validate then some map else none

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
  if !map.validate || fact.context != map.source then none
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
      .gather (.variable sourceIndex) (.variable position)
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
      .gather (.variable sourceIndex) (.variable position)
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
    let gathered := .gather (.variable sourceIndex) (.variable sourceIndex)
    let map := fixtureMap source destination [gathered]
    reindex map (.variable lane) = some gathered := by native_decide

example :
    let lane := fixtureVariable 0 0 4
    let sourceIndex := fixtureVariable 1 0 4
    let position := fixtureVariable 2 0 8
    let source := fixtureContext [lane]
    let destination := fixtureContext [sourceIndex, position]
    let gathered := .gather (.variable sourceIndex) (.variable position)
    let map := fixtureMap source destination [gathered]
    map.validate = true := by native_decide

/-- A gather's lookup lane domain may exceed its result codomain: positions 4 through 7 select
from an eight-lane source family while the gathered result remains in the four-element codomain. -/
example :
    let sourceIndex := fixtureVariable 1 0 4
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := .gather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable position, 7),
      (gathered, 3)
    ] = some 3 := by
  native_decide

/-- Gather result values are still bounded by the source-family codomain, even when their lookup
position is from a larger lane domain. -/
example :
    let sourceIndex := fixtureVariable 1 0 4
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := .gather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable position, 7),
      (gathered, 4)
    ] = none := by
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
    (staticIndexMap source lane 3).map (fun map =>
      map.destination == fixtureContext [other] &&
        reindex map (.variable lane) == some (.constant 3) &&
        reindex map (.variable other) == some (.variable other)) = some true := by
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
    let gathered := .gather (.variable sourceIndex) (.variable position)
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
    let first := fixtureMap source intermediate [.gather (.variable sourceIndex) (.variable position)]
    let second := fixtureMap intermediate destination [.variable outer, .variable position]
    let expression : IndexedParameterExpr := .index (.variable lane)
    (composeIndexMap first second).bind (expression.reindex) =
      some (.index (.gather (.variable outer) (.variable position))) := by
  native_decide

example :
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := .gather (.variable sourceIndex) (.variable position)
    (IndexedParameterExpr.index gathered).evaluate [] context [
      (.variable sourceIndex, 1),
      (.variable position, 2),
      (gathered, 6)
    ] = some 6 := by
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
    let map := fixtureMap source destination [.gather (.variable sourceIndex) (.variable position)]
    (IndexedParameterExpr.index (.variable lane)).reindex map =
      some (.index (.gather (.variable sourceIndex) (.variable position))) := by
  native_decide

example :
    let sourceIndex := fixtureVariable 1 0 8
    let position := fixtureVariable 2 0 8
    let context := fixtureContext [sourceIndex, position]
    let gathered := .gather (.variable sourceIndex) (.variable position)
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
    let gathered := .gather (.variable sourceIndex) (.variable position)
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
    let gathered := .gather (.variable sourceIndex) (.variable position)
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
