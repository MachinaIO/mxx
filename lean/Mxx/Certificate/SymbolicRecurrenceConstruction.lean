import Mxx.Certificate.Rules.LoopRecurrence
import Mxx.Certificate.SymbolicRecurrence

namespace Mxx.Certificate

/-! # Analyzer-owned symbolic recurrence construction

This module constructs the three schema-indexed components of a symbolic recurrence transfer
from the analyzer's actual typed initial facts, one-step body templates, and recursively registered
family-element templates.
-/

structure ValidatedSequentialRecurrenceSource where
  source : SequentialRecurrenceSource
  schemas : List CarriedValueSchema
  schemaValidation : source.validateCoarseCarriedSchemas = .ok schemas

inductive SymbolicRecurrenceConstructionError where
  | schema (error : SymbolicRecurrenceError)
  | componentArityMismatch
  | componentKindMismatch (slot : Nat)
  | escapedInitialBound (slot : Nat)
  | unsupportedNaturalTransition (slot : Nat)
  | invalidIntegerCarriedPath (slot : Nat) (path : IntBoundFactPath)
  | invalidMatrixCarriedPath (slot : Nat) (path : BoundFactPath)
  | invalidExternalNaturalBound
      (slot : Nat)
      (recurrence : SequentialRecurrenceInstanceRef)
      (path : BoundFactPath)
  | invalidExternalIntegerBound
      (slot : Nat)
      (recurrence : SequentialRecurrenceInstanceRef)
      (path : IntBoundFactPath)
  | loopIndexedBound (slot : Nat)
  | invalidExpressionReference (slot : Nat)
  | invalidSymbolicForm (slot : Nat)
  | missingFamilyTemplate (slot : Nat) (aggregate : FamilyAggregateRef)
  | transfer (error : SymbolicRecurrenceError)
  deriving BEq, DecidableEq, Repr

structure ConstructedSymbolicRecurrenceTransfer where
  expressionArena : ExpressionArena
  symbolicFormArena : SymbolicMatrixFormArena
  transfer : SymbolicRecurrenceTransfer

private def intExprHasLoopIndex : IntExpr → Bool
  | .loopIndex _ => true
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .roundDivide left right => intExprHasLoopIndex left || intExprHasLoopIndex right
  | .log2Ceil value => intExprHasLoopIndex value
  | .constant _ | .parameter _ => false

private def lookupDependency
    (identity : SequentialRecurrenceInstanceRef)
    (dependencies : List SymbolicRecurrenceTransfer) : Option SymbolicRecurrenceTransfer :=
  uniqueLaneUniformRecurrenceMatch? identity dependencies (·.identity)

private def naturalDependencyValid
    (dependencies : List SymbolicRecurrenceTransfer)
    (identity : SequentialRecurrenceInstanceRef)
    (path : BoundFactPath) : Bool :=
  match lookupDependency identity dependencies with
  | some transfer => path.valid transfer.source.bodyOutputs
  | none => false

private def integerDependencyValid
    (dependencies : List SymbolicRecurrenceTransfer)
    (identity : SequentialRecurrenceInstanceRef)
    (path : IntBoundFactPath) : Bool :=
  match lookupDependency identity dependencies with
  | some transfer => path.valid transfer.source.bodyOutputs
  | none => false

def SequentialRecurrenceSource.validateForSymbolicConstruction
    (source : SequentialRecurrenceSource) :
    Except SymbolicRecurrenceConstructionError ValidatedSequentialRecurrenceSource :=
  if intExprHasLoopIndex source.count then
    .error (.loopIndexedBound 0)
  else
  match validation : source.validateCoarseCarriedSchemas with
  | .error error => .error (.schema error)
  | .ok schemas => .ok { source, schemas, schemaValidation := validation }

private def boundIsDependencyClosed
    (dependencies : List SymbolicRecurrenceTransfer) : BoundExpr → Bool
  | .add left right | .multiply left right | .maximum left right | .minimum left right =>
      boundIsDependencyClosed dependencies left && boundIsDependencyClosed dependencies right
  | .floorDivide value _ => boundIsDependencyClosed dependencies value
  | .matrixProduct ring inner left right =>
      !intExprHasLoopIndex ring && !intExprHasLoopIndex inner &&
        boundIsDependencyClosed dependencies left && boundIsDependencyClosed dependencies right
  | .recurrenceResult identity path => naturalDependencyValid dependencies identity path
  | .carriedInput _ => false
  | .parameter value | .absolute value => !intExprHasLoopIndex value
  | .constant _ => true

private def intBoundIsDependencyClosed
    (dependencies : List SymbolicRecurrenceTransfer) : IntBoundExpr → Bool
  | .natural value => boundIsDependencyClosed dependencies value
  | .negate value => intBoundIsDependencyClosed dependencies value
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .minimum left right | .maximum left right =>
      intBoundIsDependencyClosed dependencies left && intBoundIsDependencyClosed dependencies right
  | .carriedInput _ => false
  | .recurrenceResult identity path => integerDependencyValid dependencies identity path
  | .integer value => !intExprHasLoopIndex value

private def signalPresence (form : AffineForm) : SignalPresence :=
  if form.terms.isEmpty then .none else .present

private def coefficientL1Bound (form : AffineForm) : BoundExpr :=
  form.terms.foldl (fun total term => .add total term.coefficient.normBound) (.constant 0)

private def matrixBoundSummary (fact : MatrixFact) : MatrixBoundSummary :=
  match fact.primary with
  | .exact _ => .exactLarge fact.totalNormBound
  | .affine form => {
      signal := signalPresence form
      coefficientL1Bound := coefficientL1Bound form
      noiseBound := form.noiseBound
      totalBound := fact.totalNormBound
    }

private def sameFamilyTemplateKey
    (left right : FamilyAggregateRef) : Bool :=
  match left, right with
  | .joint leftFamily leftSlot _, .joint rightFamily rightSlot _ =>
      leftFamily == rightFamily && leftSlot == rightSlot
  | _, _ => left == right

/-- Family element templates are uniform across parallel lanes and sequential iterations.  The
instance path remains part of value identity, but it does not select a different element schema
or hard-bound transition for one frozen parallel-family output. -/
private def lookupFamilyTemplate
    (source : SequentialRecurrenceSource)
    (aggregate : FamilyAggregateRef) : Option ValueFactTemplate :=
  (source.familyElementTemplates.find? fun entry =>
    sameFamilyTemplateKey entry.1 aggregate).map (·.2)

private def buildInitialState
    (source : SequentialRecurrenceSource)
    (dependencies : List SymbolicRecurrenceTransfer)
    (slot : Nat) :
    (schema : CarriedValueSchema) → ValueFactTemplate →
      Except SymbolicRecurrenceConstructionError
        (CarriedBoundTemplateState schema.boundSchema)
  | .matrix _ _, { fact := .matrix fact, .. } => do
      let summary := matrixBoundSummary fact
      unless boundIsDependencyClosed dependencies summary.coefficientL1Bound &&
          boundIsDependencyClosed dependencies summary.noiseBound &&
          boundIsDependencyClosed dependencies summary.totalBound do
        throw (.escapedInitialBound slot)
      return .matrix summary
  | .integer, { fact := .integer fact, .. } => do
      unless intBoundIsDependencyClosed dependencies fact.lower &&
          intBoundIsDependencyClosed dependencies fact.upper do
        throw (.escapedInitialBound slot)
      return .integer fact.lower fact.upper
  | .boolean, { fact := .boolean _, .. } => .ok .boolean
  | .bytes, { fact := .bytes _, .. } => .ok .bytes
  | .family _ element, { fact := .family family, .. } => do
      let template ← match lookupFamilyTemplate source family.aggregate with
        | some template => pure template
        | none => throw (.missingFamilyTemplate slot family.aggregate)
      let actual ← template.toCarriedValueSchema |>.mapError fun error =>
        .schema (.initialSchema slot error)
      if actual == element then
        return .familyEnvelope (← buildInitialState source dependencies slot element template)
      else throw (.componentKindMismatch slot)
  | _, _ => .error (.componentKindMismatch slot)

private def buildInitialBounds
    (source : SequentialRecurrenceSource) :
    (dependencies : List SymbolicRecurrenceTransfer) →
    (slot : Nat) →
    (schemas : List CarriedValueSchema) →
    List ValueFactTemplate →
    Except SymbolicRecurrenceConstructionError
      (CarriedBoundTemplateVector (schemas.map CarriedValueSchema.boundSchema))
  | _, _, [], [] => .ok .nil
  | dependencies, slot, schema :: schemas, fact :: facts => do
      let head ← buildInitialState source dependencies slot schema fact
      return .cons head
        (← buildInitialBounds source dependencies (slot + 1) schemas facts)
  | _, _, [], _ | _, _, _ :: _, [] => .error .componentArityMismatch

private def boundPathSlot : BoundFactPath → Nat
  | .affineCoefficientBound slot _ | .affineNoiseBound slot | .matrixTotalBound slot => slot
  | .familyElement slot _ _ => slot

private def intBoundPathSlot : IntBoundFactPath → Nat
  | .lower slot | .upper slot => slot
  | .familyElement slot _ _ => slot

private def nestedMatrixPath? :
    (schema : CarriedBoundSchema) → BoundFactPath →
      Option (CarriedBoundNestedPath schema .matrixSummary)
  | .matrixSummary, .affineCoefficientBound _ _ => some .here
  | .matrixSummary, .affineNoiseBound _ => some .here
  | .matrixSummary, .matrixTotalBound _ => some .here
  | .family _ element, .familyElement _ _ nested =>
      (nestedMatrixPath? element nested).map .familyElement
  | _, _ => none

private def matrixStatePath? :
    (schemas : List CarriedBoundSchema) → Nat → BoundFactPath →
      Option (CarriedBoundStatePath schemas .matrixSummary)
  | [], _, _ => none
  | schema :: _, 0, path => (nestedMatrixPath? schema path).map .head
  | _ :: tail, slot + 1, path => (matrixStatePath? tail slot path).map .tail

private def nestedIntegerPath? :
    (schema : CarriedBoundSchema) → IntBoundFactPath →
      Option (CarriedBoundNestedPath schema .integerInterval)
  | .integerInterval, .lower _ => some .here
  | .integerInterval, .upper _ => some .here
  | .family _ element, .familyElement _ _ nested =>
      (nestedIntegerPath? element nested).map .familyElement
  | _, _ => none

private def integerBoundStatePath? :
    (schemas : List CarriedBoundSchema) → Nat → IntBoundFactPath →
      Option (CarriedBoundStatePath schemas .integerInterval)
  | [], _, _ => none
  | schema :: _, 0, path => (nestedIntegerPath? schema path).map .head
  | _ :: tail, slot + 1, path => (integerBoundStatePath? tail slot path).map .tail

private def matrixField : BoundFactPath → MatrixBoundField
  | .affineCoefficientBound _ _ => .coefficientL1
  | .affineNoiseBound _ => .noise
  | .matrixTotalBound _ => .total
  | .familyElement _ _ nested => matrixField nested

private def integerField : IntBoundFactPath → IntegerBoundField
  | .lower _ => .lower
  | .upper _ => .upper
  | .familyElement _ _ nested => integerField nested

private def translateNaturalBound
    (outputSlot : Nat)
    (dependencies : List SymbolicRecurrenceTransfer)
    (previous : List CarriedBoundSchema) :
    BoundExpr →
      Except SymbolicRecurrenceConstructionError (NatBoundTransitionExpr previous)
  | .constant value => .ok (.constant value)
  | .parameter value =>
      if intExprHasLoopIndex value then .error (.loopIndexedBound outputSlot)
      else .ok (.parameter value)
  | .absolute value =>
      if intExprHasLoopIndex value then .error (.loopIndexedBound outputSlot)
      else .ok (.absolute value)
  | .add left right => return (.add
      (← translateNaturalBound outputSlot dependencies previous left)
      (← translateNaturalBound outputSlot dependencies previous right))
  | .multiply left right => return (.multiply
      (← translateNaturalBound outputSlot dependencies previous left)
      (← translateNaturalBound outputSlot dependencies previous right))
  | .minimum left right => return (.minimum
      (← translateNaturalBound outputSlot dependencies previous left)
      (← translateNaturalBound outputSlot dependencies previous right))
  | .maximum left right => return (.maximum
      (← translateNaturalBound outputSlot dependencies previous left)
      (← translateNaturalBound outputSlot dependencies previous right))
  | .matrixProduct ring inner left right => do
      if intExprHasLoopIndex ring || intExprHasLoopIndex inner then
        throw (.loopIndexedBound outputSlot)
      return .matrixProduct ring inner
        (← translateNaturalBound outputSlot dependencies previous left)
        (← translateNaturalBound outputSlot dependencies previous right)
  | .floorDivide value divisor => return (.floorDivide
      (← translateNaturalBound outputSlot dependencies previous value) divisor)
  | .carriedInput path =>
      match matrixStatePath? previous (boundPathSlot path) path with
      | some typedPath => .ok (.previousState typedPath (matrixField path))
      | none => .error (.invalidMatrixCarriedPath outputSlot path)
  | .recurrenceResult identity path =>
      if naturalDependencyValid dependencies identity path then
        .ok (.externalRecurrence identity path)
      else .error (.invalidExternalNaturalBound outputSlot identity path)

private def translateIntegerBound
    (outputSlot : Nat)
    (dependencies : List SymbolicRecurrenceTransfer)
    (previous : List CarriedBoundSchema) :
    IntBoundExpr →
      Except SymbolicRecurrenceConstructionError (IntBoundTransitionExpr previous)
  | .integer value =>
      if intExprHasLoopIndex value then .error (.loopIndexedBound outputSlot)
      else .ok (.parameter value)
  | .natural _ => .error (.unsupportedNaturalTransition outputSlot)
  | .negate value => return (.negate
      (← translateIntegerBound outputSlot dependencies previous value))
  | .add left right => return (.add
      (← translateIntegerBound outputSlot dependencies previous left)
      (← translateIntegerBound outputSlot dependencies previous right))
  | .subtract left right => return (.subtract
      (← translateIntegerBound outputSlot dependencies previous left)
      (← translateIntegerBound outputSlot dependencies previous right))
  | .multiply left right => return (.multiply
      (← translateIntegerBound outputSlot dependencies previous left)
      (← translateIntegerBound outputSlot dependencies previous right))
  | .divide left right => return (.divide
      (← translateIntegerBound outputSlot dependencies previous left)
      (← translateIntegerBound outputSlot dependencies previous right))
  | .minimum left right => return (.minimum
      (← translateIntegerBound outputSlot dependencies previous left)
      (← translateIntegerBound outputSlot dependencies previous right))
  | .maximum left right => return (.maximum
      (← translateIntegerBound outputSlot dependencies previous left)
      (← translateIntegerBound outputSlot dependencies previous right))
  | .carriedInput path =>
      match integerBoundStatePath? previous (intBoundPathSlot path) path with
      | none => .error (.invalidIntegerCarriedPath outputSlot path)
      | some typedPath => .ok (.previousState typedPath (integerField path))
  | .recurrenceResult identity path =>
      if integerDependencyValid dependencies identity path then
        .ok (.externalRecurrence identity path)
      else .error (.invalidExternalIntegerBound outputSlot identity path)

private def internMatrixBodyForm
    (slot : Nat)
    (carriedArity : Nat)
    (matrixType : MatrixTypeExpr)
    (fact : MatrixFact)
    (expressionArena : ExpressionArena)
    (symbolicFormArena : SymbolicMatrixFormArena) :
    Except SymbolicRecurrenceConstructionError
      (ExpressionArena × SymbolicMatrixFormArena × SymbolicMatrixFormRef) := do
  let (expressionArena, form) ← match fact.primary with
    | .exact expression =>
        match expressionArena.internMatrix expression with
        | some (arena, reference) => pure (arena, SymbolicMatrixForm.signalAtom reference)
        | none => throw (.invalidExpressionReference slot)
    | .affine form => pure (expressionArena, .affineLeaf form)
  let context : SymbolicFormWFContext := {
    expressionArena
    preimageRelationCount := 0
    gadgetRelationCount := 0
    carriedArity
    recurrences := []
    allowCarriedInputs := true
  }
  let (symbolicFormArena, reference) ← match symbolicFormArena.intern context {
      matrixType
      form
    } with
    | some result => pure result
    | none => throw (.invalidSymbolicForm slot)
  return (expressionArena, symbolicFormArena, reference)

private def buildBodyOutputs
    (carriedArity : Nat) :
    (slot : Nat) →
    (schemas : List CarriedValueSchema) →
    List ValueFactTemplate →
    ExpressionArena →
    SymbolicMatrixFormArena →
    Except SymbolicRecurrenceConstructionError
      (ExpressionArena × SymbolicMatrixFormArena × SymbolicCarriedOutputVector schemas)
  | _, [], [], expressionArena, symbolicFormArena =>
      .ok (expressionArena, symbolicFormArena, .nil)
  | slot, .matrix matrixType _ :: schemas, { fact := .matrix fact, .. } :: facts,
      expressionArena, symbolicFormArena => do
      let (expressionArena, symbolicFormArena, reference) ←
        internMatrixBodyForm slot carriedArity matrixType fact expressionArena symbolicFormArena
      let (expressionArena, symbolicFormArena, tail) ←
        buildBodyOutputs carriedArity (slot + 1) schemas facts expressionArena symbolicFormArena
      return (expressionArena, symbolicFormArena, .cons (.matrix reference) tail)
  | slot, .integer :: schemas, { fact := .integer fact, .. } :: facts,
      expressionArena, symbolicFormArena => do
      let (nextArena, reference) ← match expressionArena.internInteger fact.expression with
        | some result => pure result
        | none => throw (.invalidExpressionReference slot)
      let (finalArena, symbolicFormArena, tail) ← buildBodyOutputs carriedArity
        (slot + 1) schemas facts nextArena symbolicFormArena
      return (finalArena, symbolicFormArena, .cons (.integer reference) tail)
  | slot, .boolean :: schemas, { fact := .boolean fact, .. } :: facts,
      expressionArena, symbolicFormArena => do
      let (nextArena, reference) ← match expressionArena.internBoolean fact.expression with
        | some result => pure result
        | none => throw (.invalidExpressionReference slot)
      let (finalArena, symbolicFormArena, tail) ← buildBodyOutputs carriedArity
        (slot + 1) schemas facts nextArena symbolicFormArena
      return (finalArena, symbolicFormArena, .cons (.boolean reference) tail)
  | slot, .bytes :: schemas, { fact := .bytes value, .. } :: facts,
      expressionArena, symbolicFormArena => do
      let (expressionArena, symbolicFormArena, tail) ← buildBodyOutputs carriedArity
        (slot + 1) schemas facts expressionArena symbolicFormArena
      return (expressionArena, symbolicFormArena, .cons (.bytes value) tail)
  | slot, .family _ _ :: schemas, { fact := .family family, .. } :: facts,
      expressionArena, symbolicFormArena => do
      let (expressionArena, symbolicFormArena, tail) ← buildBodyOutputs carriedArity
        (slot + 1) schemas facts expressionArena symbolicFormArena
      return (expressionArena, symbolicFormArena, .cons (.family family.aggregate) tail)
  | _, [], _, _, _ | _, _ :: _, [], _, _ => .error .componentArityMismatch
  | slot, _ :: _, _ :: _, _, _ => .error (.componentKindMismatch slot)

private def statePathAt?
    (schemas : List CarriedBoundSchema)
    (target : CarriedBoundSchema) :
    Nat → Option (CarriedBoundStatePath schemas target) :=
  match schemas with
  | [] => fun _ => none
  | schema :: tail => fun
      | 0 =>
          if same : schema = target then
            some (same ▸ (CarriedBoundStatePath.head (.here) :
              CarriedBoundStatePath (schema :: tail) schema))
          else none
      | slot + 1 => (statePathAt? tail target slot).map .tail

private def descendNestedFamily
    {root count element}
    (path : CarriedBoundNestedPath root (.family count element)) :
    CarriedBoundNestedPath root element := by
  cases path with
  | here => exact .familyElement .here
  | familyElement nested => exact .familyElement (descendNestedFamily nested)

private def descendStateFamily
    {schemas count element}
    (path : CarriedBoundStatePath schemas (.family count element)) :
    CarriedBoundStatePath schemas element :=
  match path with
  | .head nested => .head (descendNestedFamily nested)
  | .tail tail => .tail (descendStateFamily tail)

private def copyPreviousTransition
    (previous : List CarriedBoundSchema) :
    (schema : CarriedBoundSchema) →
    CarriedBoundStatePath previous schema →
      CarriedBoundTransition previous schema
  | .matrixSummary, path => .matrix {
      signal := .previousState path
      coefficientL1Bound := .previousState path .coefficientL1
      noiseBound := .previousState path .noise
      totalBound := .previousState path .total
    }
  | .integerInterval, path => .integer
      (.previousState path .lower) (.previousState path .upper)
  | .boolean, _ => .boolean
  | .bytes, _ => .bytes
  | .family _ element, path =>
      .familyEnvelope (copyPreviousTransition previous element (descendStateFamily path))

private def signalTransition {previous : List CarriedBoundSchema}
    (summary : MatrixBoundSummary) : SignalTransitionExpr previous :=
  .constant (match summary.signal with | .none => false | .present => true)

private def buildTransitionState
    (source : SequentialRecurrenceSource)
    (dependencies : List SymbolicRecurrenceTransfer)
    (slot : Nat)
    (previous : List CarriedBoundSchema) :
    (schema : CarriedValueSchema) → ValueFactTemplate →
      Except SymbolicRecurrenceConstructionError
        (CarriedBoundTransition previous schema.boundSchema)
  | .matrix _ _, { fact := .matrix fact, .. } => do
      let summary := matrixBoundSummary fact
      return .matrix {
        signal := signalTransition summary
        coefficientL1Bound := ← translateNaturalBound slot dependencies previous
          summary.coefficientL1Bound
        noiseBound := ← translateNaturalBound slot dependencies previous summary.noiseBound
        totalBound := ← translateNaturalBound slot dependencies previous summary.totalBound
      }
  | .integer, { fact := .integer fact, .. } => do
      return .integer
        (← translateIntegerBound slot dependencies previous fact.lower)
        (← translateIntegerBound slot dependencies previous fact.upper)
  | .boolean, { fact := .boolean _, .. } => .ok .boolean
  | .bytes, { fact := .bytes _, .. } => .ok .bytes
  | .family count element, { fact := .family family, .. } =>
      match family.aggregate with
      | .carriedInput sourceSlot =>
          match statePathAt? previous (.family count element.boundSchema) sourceSlot with
          | some path => .ok
              (copyPreviousTransition previous (.family count element.boundSchema) path)
          | none => .error (.componentKindMismatch slot)
      | aggregate => do
          let template ← match lookupFamilyTemplate source aggregate with
            | some template => pure template
            | none => throw (.missingFamilyTemplate slot aggregate)
          let actual ← template.toCarriedValueSchema |>.mapError fun error =>
            .schema (.bodySchema slot error)
          if actual == element then
            return .familyEnvelope
              (← buildTransitionState source dependencies slot previous element template)
          else throw (.componentKindMismatch slot)
  | _, _ => .error (.componentKindMismatch slot)

private def buildBoundTransition
    (source : SequentialRecurrenceSource) :
    (dependencies : List SymbolicRecurrenceTransfer) →
    (slot : Nat) →
    (previous : List CarriedBoundSchema) →
    (schemas : List CarriedValueSchema) →
    List ValueFactTemplate →
    Except SymbolicRecurrenceConstructionError
      (CarriedBoundTransitionVector previous
        (schemas.map CarriedValueSchema.boundSchema))
  | _, _, _, [], [] => .ok .nil
  | dependencies, slot, previous, schema :: schemas, fact :: facts => do
      let head ← buildTransitionState source dependencies slot previous schema fact
      return .cons head
        (← buildBoundTransition source dependencies (slot + 1) previous schemas facts)
  | _, _, _, [], _ | _, _, _, _ :: _, [] => .error .componentArityMismatch

def ValidatedSequentialRecurrenceSource.constructTransfer
    (validated : ValidatedSequentialRecurrenceSource)
    (identity : SequentialRecurrenceInstanceRef)
    (dependencies : List SymbolicRecurrenceTransfer)
    (expressionArena : ExpressionArena := {})
    (symbolicFormArena : SymbolicMatrixFormArena := {}) :
    Except SymbolicRecurrenceConstructionError ConstructedSymbolicRecurrenceTransfer := do
  let initialBounds ← buildInitialBounds validated.source dependencies 0 validated.schemas
    validated.source.initial.toList
  let (expressionArena, symbolicFormArena, bodyOutputs) ← buildBodyOutputs
    validated.source.carriedArity 0 validated.schemas validated.source.bodyOutputs.toList
    expressionArena symbolicFormArena
  let previous := validated.schemas.map CarriedValueSchema.boundSchema
  let boundTransition ← buildBoundTransition validated.source dependencies 0 previous
    validated.schemas validated.source.bodyOutputs.toList
  let transfer ← SymbolicRecurrenceTransfer.build identity validated.source initialBounds
    bodyOutputs boundTransition |>.mapError .transfer
  return { expressionArena, symbolicFormArena, transfer }

def SequentialRecurrenceSource.constructSymbolicTransfer
    (source : SequentialRecurrenceSource)
    (identity : SequentialRecurrenceInstanceRef)
    (dependencies : List SymbolicRecurrenceTransfer)
    (expressionArena : ExpressionArena := {})
    (symbolicFormArena : SymbolicMatrixFormArena := {}) :
    Except SymbolicRecurrenceConstructionError ConstructedSymbolicRecurrenceTransfer := do
  let validated ← source.validateForSymbolicConstruction
  validated.constructTransfer identity dependencies expressionArena symbolicFormArena

/-! ## Focused fixtures -/

private def constructionFixtureSite : CoreNodeRef := {
  stage := ⟨"construction-fixture"⟩
  scope := ⟨[]⟩
  node := ⟨0⟩
}

private def constructionFixtureSource : SequentialRecurrenceSource where
  loop := ⟨constructionFixtureSite⟩
  count := .constant 1
  carriedArity := 2
  initial := ⟨#[
    {
      fact := .integer {
        expression := .intConstant 1
        lower := .integer (.constant 1)
        upper := .integer (.constant 2)
      }
      schema := .integer
    },
    {
      fact := .integer {
        expression := .intConstant 10
        lower := .integer (.constant 10)
        upper := .integer (.constant 20)
      }
      schema := .integer
    }
  ], rfl⟩
  bodyInputs := ⟨#[
    {
      definition := {
        stage := ⟨"construction-fixture"⟩
        name := "body"
      }
      bodyScope := ⟨[]⟩
      node := ⟨0⟩
      port := 0
    },
    {
      definition := {
        stage := ⟨"construction-fixture"⟩
        name := "body"
      }
      bodyScope := ⟨[]⟩
      node := ⟨0⟩
      port := 1
    }
  ], rfl⟩
  bodyOutputs := ⟨#[
    {
      fact := .integer {
        expression := .carriedInput (.integerValue 1)
        lower := .carriedInput (.lower 1)
        upper := .carriedInput (.upper 1)
      }
      schema := .integer
    },
    {
      fact := .integer {
        expression := .carriedInput (.integerValue 0)
        lower := .carriedInput (.lower 0)
        upper := .carriedInput (.upper 0)
      }
      schema := .integer
    }
  ], rfl⟩
  invariantInputs := []
  iterationVariable := ⟨0⟩

private def constructionFixtureIdentity : SequentialRecurrenceInstanceRef := {
  recurrence := ⟨constructionFixtureSite⟩
  path := []
}

example :
    match constructionFixtureSource.constructSymbolicTransfer constructionFixtureIdentity [] with
    | .ok result => result.transfer.resolveBounds [] {} = .ok {
        identity := constructionFixtureIdentity
        schemas := [.integerInterval, .integerInterval]
        values := .cons (.integer 10 20) (.cons (.integer 1 2) .nil)
      }
    | .error _ => False := by
  rfl

private def unsupportedTransitionFixture : SequentialRecurrenceSource := {
  constructionFixtureSource with
  carriedArity := 1
  initial := ⟨#[{
    fact := .integer {
      expression := .intConstant 1
      lower := .integer (.constant 0)
      upper := .integer (.constant 1)
    }
    schema := .integer
  }], rfl⟩
  bodyInputs := ⟨#[constructionFixtureSource.bodyInputs[0]], by simp⟩
  bodyOutputs := ⟨#[{
    fact := .integer {
      expression := .carriedInput (.integerValue 0)
      lower := .natural (.carriedInput (.matrixTotalBound 0))
      upper := .integer (.constant 1)
    }
    schema := .integer
  }], rfl⟩
}

example :
    unsupportedTransitionFixture.constructSymbolicTransfer constructionFixtureIdentity [] =
      .error (.unsupportedNaturalTransition 0) := by
  rfl

private def matrixFixtureType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 1
  columns := .constant 1

private def matrixFixtureBound : BoundExpr :=
  .floorDivide (.absolute matrixFixtureType.modulus) 2

private def matrixFixtureInitial : ValueFactTemplate := {
  fact := .matrix {
    subject := .protocolInput ⟨"matrix-initial"⟩
    primary := .exact (.zero matrixFixtureType)
    relations := []
    totalNormBound := matrixFixtureBound
  }
  schema := .matrix matrixFixtureType .exact [] .unknown
}

private def matrixFixtureBody : ValueFactTemplate := {
  fact := .matrix {
    subject := .protocolInput ⟨"matrix-body"⟩
    primary := .exact (.carriedInput matrixFixtureType (.exactExpression 0))
    relations := []
    totalNormBound := matrixFixtureBound
  }
  schema := .matrix matrixFixtureType .exact [] .unknown
}

private def matrixFixtureSource : SequentialRecurrenceSource where
  loop := ⟨constructionFixtureSite⟩
  count := .constant 1
  carriedArity := 1
  initial := ⟨#[matrixFixtureInitial], rfl⟩
  bodyInputs := ⟨#[constructionFixtureSource.bodyInputs[0]], by simp⟩
  bodyOutputs := ⟨#[matrixFixtureBody], rfl⟩
  invariantInputs := []
  iterationVariable := ⟨0⟩

example :
    (matrixFixtureSource.constructSymbolicTransfer constructionFixtureIdentity []).map
      (fun result =>
        (result.transfer.carriedSchemas, result.symbolicFormArena.entries.length)) =
      .ok ([.matrix matrixFixtureType .unknown], 1) := by
  native_decide

private def matrixFixtureResolved : Option (Bool × Nat × Nat × Nat) :=
  match matrixFixtureSource.constructSymbolicTransfer constructionFixtureIdentity [] with
  | .error _ => none
  | .ok result =>
      match result.transfer.resolveBounds [] {} with
      | .error _ => none
      | .ok { schemas := [.matrixSummary], values := .cons (.matrix signal coefficient noise total) .nil, .. } =>
          some (signal, coefficient, noise, total)
      | .ok _ => none

example : matrixFixtureResolved = some (true, 1, 0, 8) := by
  native_decide

private def matrixFixtureDependency : SymbolicRecurrenceTransfer :=
  ((matrixFixtureSource.constructSymbolicTransfer constructionFixtureIdentity []).toOption.get
    (by native_decide)).transfer

private def matrixFixtureLaneReference : SequentialRecurrenceInstanceRef :=
  constructionFixtureIdentity.appendPath [
    .parallelLane { constructionFixtureSite with node := ⟨9⟩ } ⟨11⟩,
    .parallelLane { constructionFixtureSite with node := ⟨10⟩ } ⟨12⟩
  ]

private def laneDependentMatrixInitial : ValueFactTemplate := {
  matrixFixtureInitial with
  fact := .matrix {
    subject := .protocolInput ⟨"lane-dependent-initial"⟩
    primary := .exact (.zero matrixFixtureType)
    relations := []
    totalNormBound := .recurrenceResult matrixFixtureLaneReference (.matrixTotalBound 0)
  }
}

private def laneDependentMatrixSource : SequentialRecurrenceSource := {
  matrixFixtureSource with
  loop := ⟨{ constructionFixtureSite with node := ⟨13⟩ }⟩
  initial := ⟨#[laneDependentMatrixInitial], rfl⟩
}

private def laneDependentMatrixIdentity : SequentialRecurrenceInstanceRef := {
  recurrence := ⟨laneDependentMatrixSource.loop.site⟩
  path := []
}

/-- Phase A accepts a full family-lane identity only through the unique analyzer-owned base
transfer. -/
example :
    (laneDependentMatrixSource.constructSymbolicTransfer laneDependentMatrixIdentity
      [matrixFixtureDependency]).isOk = true := by
  native_decide

/-- A sequential suffix is not a lane-uniform dependency. -/
example :
    let sequentialReference := constructionFixtureIdentity.appendPath [
      .sequentialIteration { constructionFixtureSite with node := ⟨9⟩ } ⟨11⟩
    ]
    let initial : ValueFactTemplate := {
      laneDependentMatrixInitial with
      fact := .matrix {
        subject := .protocolInput ⟨"sequential-dependent-initial"⟩
        primary := .exact (.zero matrixFixtureType)
        relations := []
        totalNormBound := .recurrenceResult sequentialReference (.matrixTotalBound 0)
      }
    }
    let source : SequentialRecurrenceSource := {
      laneDependentMatrixSource with initial := ⟨#[initial], rfl⟩
    }
    (match source.constructSymbolicTransfer laneDependentMatrixIdentity
      [matrixFixtureDependency] with
      | .error _ => true
      | .ok _ => false) = true := by
  native_decide

private def familyFixtureAggregate : FamilyAggregateRef :=
  .joint ⟨"family-fixture"⟩ 0 []

private def familyFixtureElement : ValueFactTemplate := {
  fact := .matrix {
    subject := .familyElement familyFixtureAggregate ⟨0⟩
    primary := .affine { terms := [], noiseBound := .constant 3 }
    relations := []
    totalNormBound := .constant 3
  }
  schema := .matrix matrixFixtureType (.affine []) [] .unknown
}

private def familyFixtureSchema : ValueFactSchema :=
  .family (.constant 2) familyFixtureElement.schema

private def familyFixtureSource : SequentialRecurrenceSource where
  loop := ⟨constructionFixtureSite⟩
  count := .constant 1
  carriedArity := 1
  initial := ⟨#[{
    fact := .family {
      aggregate := familyFixtureAggregate
      count := .constant 2
      elementSchema := familyFixtureElement.schema
    }
    schema := familyFixtureSchema
  }], rfl⟩
  bodyInputs := ⟨#[constructionFixtureSource.bodyInputs[0]], by simp⟩
  bodyOutputs := ⟨#[{
    fact := .family {
      aggregate := .carriedInput 0
      count := .constant 2
      elementSchema := familyFixtureElement.schema
    }
    schema := familyFixtureSchema
  }], rfl⟩
  familyElementTemplates := [(familyFixtureAggregate, familyFixtureElement)]
  invariantInputs := []
  iterationVariable := ⟨0⟩

example :
    match familyFixtureSource.constructSymbolicTransfer constructionFixtureIdentity [] with
    | .ok result => result.transfer.resolveBounds [] {} = .ok {
        identity := constructionFixtureIdentity
        schemas := [.family (.constant 2) .matrixSummary]
        values := .cons (.familyEnvelope (.matrix false 0 3 3)) .nil
      }
    | .error _ => False := by
  rfl

private def constructionError?
    (source : SequentialRecurrenceSource) : Option SymbolicRecurrenceConstructionError :=
  match source.constructSymbolicTransfer constructionFixtureIdentity [] with
  | .ok _ => none
  | .error error => some error

private def missingFamilyTemplateFixture : SequentialRecurrenceSource := {
  familyFixtureSource with familyElementTemplates := []
}

example : constructionError? missingFamilyTemplateFixture =
    some (.missingFamilyTemplate 0 familyFixtureAggregate) := by
  native_decide

example : constructionError? {
    familyFixtureSource with
    familyElementTemplates := [
      (familyFixtureAggregate, familyFixtureElement),
      (familyFixtureAggregate, familyFixtureElement)
    ]
  } = none := by
  native_decide

private def familyFixtureInstantiatedAggregate : FamilyAggregateRef :=
  .joint ⟨"family-fixture"⟩ 0 [
    .sequentialIteration constructionFixtureSite ⟨7⟩
  ]

/-- Instantiation paths distinguish values, not the uniform family element template. -/
example : constructionError? {
    familyFixtureSource with
    initial := ⟨#[{
      fact := .family {
        aggregate := familyFixtureInstantiatedAggregate
        count := .constant 2
        elementSchema := familyFixtureElement.schema
      }
      schema := familyFixtureSchema
    }], rfl⟩
  } = none := by
  native_decide

private def invalidMatrixPathBody : ValueFactTemplate := {
  matrixFixtureBody with
  fact := .matrix {
    subject := .protocolInput ⟨"invalid-matrix-path"⟩
    primary := .exact (.carriedInput matrixFixtureType (.exactExpression 0))
    relations := []
    totalNormBound := .carriedInput (.matrixTotalBound 1)
  }
}

private def invalidMatrixPathFixture : SequentialRecurrenceSource := {
  matrixFixtureSource with bodyOutputs := ⟨#[invalidMatrixPathBody], rfl⟩
}

example : constructionError? invalidMatrixPathFixture =
    some (.invalidMatrixCarriedPath 0 (.matrixTotalBound 1)) := by
  native_decide

end Mxx.Certificate
